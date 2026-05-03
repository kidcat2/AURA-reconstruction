from aura.stages.base import BaseStage

import os
import glob
import pycolmap
import random
import numpy as np
from sklearn.neighbors import KDTree # 최근접 이웃탐색
import cv2
import torch
import torch.nn.functional as F
import math

# 1. config default 수정 (iteration, learning rate 등)
# 2. workspace 준비 (output 디렉토리 생성)
# 3. COLMAP 결과 로드 (cameras, images, points3D)
# 4. 가우시안 초기화 (포인트 클라우드 → 위치, 색상, 불투명도, 공분산)
# 5. 학습 (5-5, 5-6은 반복 100회당 1번)
## 5-1. SampleTraining
## 5-2. Rasterize
## 5-3. Loss
## 5-4. Adam
## 5-5. 공분산이 너무 크거나, 투명도가 너무 작은 가우시안 제거
## 5-6. densification (분할/복제) - 가우시안이 너무 크거나, 작으면 성능 저하
# 6. .ply 저장
# 7. 결과 검증 (가우시안 수, 파일 크기 등)
# 8. context 채우기 (ply 경로)

# input: output/sfm/*/sparse/0 (cameras.bin, images.bin, points3D.bin)
# output: output/reconstruction/*/point_cloud.ply

"""
colmap

1) cameras.bin
-- camera_id
-- model (SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL 등)
-- width, height
-- params (fx, fy, cx, cy)

2) images.bin
-- image_id
-- qvec (qw, qx, qy, qz)
-- tvec (tx, ty, tz)
-- camera_id
-- name (이미지 파일명)
-- xys (2D keypoint 좌표 배열)
-- point3D_ids (각 keypoint에 대응하는 point3D id 배열)

3) point3D.bin
-- point3D_id
-- xyz (x, y, z)
-- rgb (r, g, b)
-- error (재투영 오차)
-- track (이 포인트를 관측한 image_id, point2D_idx 쌍의 배열)

"""

"""
Q.
1. View Matrix 의 구성이 R,T 인 이유
2. 초기 가우시안에서 공분산 = R * S * S_T * R_T 인 이유
3. Proj Matrix의 구성
4. 구면조화 함수 이해하기
"""

SH_C0 = 0.28209479177387814

SH_C1 = 0.4886025119029199

SH_C2 = [
    1.0925484305920792,
   -1.0925484305920792,
    0.31539156525252005,
   -1.0925484305920792,
    0.5462742152960396,
]

SH_C3 = [
   -0.5900435899266435,
    2.890611442640554,
   -0.4570457994644658,
    0.3731763325901154,
   -0.4570457994644658,
    1.445305721320277,
   -0.5900435899266435,
]

class GaussianSplatting(BaseStage):

    def __init__(self, config):

        self.iter = config["gaussian_splatting"]["iteration"]
        self.lr = config["gaussian_splatting"]["learning_rate"]

        self.near_plane = config["gaussian_splatting"]["near"]
        self.far_plane = config["gaussian_splatting"]["far"]
        self.tile_size = config["gaussian_splatting"]["tile_size"]

    def run(self, context):
        print("[GaussianSplatting] 실행")

        # input_dir : output/sfm
        # data--> output/sfm/video_00*/sparse/0
        self.input_dir = context["sparse"] 
        self.image_dir = context["frames_dir"]

    # 초기 가우시안 생성
    # (x,y,z), scale, quaternion, alpha, color 
    def init_gaussian(self, point3D):
        
        xyz = np.array([p.xyz for p in point3D.values()])     # (N, 3) float64
        rgb = np.array([p.color for p in point3D.values()])   # (N, 3) uint8
        N = xyz.shape[0]

        # scale 
        tree = KDTree(xyz)
        dist, _ = tree.query(xyz, k=4)
        scale = dist[:, 1:].mean(axis=1, keepdims=True).repeat(3, axis=1) 

        # quaternion
        q = np.zeros((N, 4))
        q[:, 0] = 1.0

        # alpha 
        alpha = np.full((N, 1), 0.5)

        # color
        # pycolmap의 color 범위 0~255, gaussian 학습 범위는 0~1
        f_dc = (rgb / 255.0 - 0.5) / 0.28209 
        f_rest = np.zeros((N, 45))

        # gaussian
        gaussians = np.concatenate([xyz, scale, q, alpha, f_dc, f_rest], axis=1) # 위치, 크기, 회전, 투명도, 기본 색상, 색상 종류 [N, 59]
        gaussians = torch.tensor(gaussians, dtype=torch.float32).cuda()
        gaussians.requires_grad_(True)
        
        return gaussians

    # 학습
    def train(self):

        video_dirs = sorted(glob.glob(f"{self.input_dir}/*/sparse/0"))

        for video_idx, video_dir in enumerate(video_dirs):
            recon = pycolmap.Reconstruction(video_dir)
            image_ids = sorted(recon.images.keys())

            gaussians = self.init_gaussian(recon.points3D)
            optimizer = torch.optim.Adam([gaussians], lr=self.lr)

            for _ in range(self.iter):
                
                # 랜덤 샘플링
                learn_id = random.choice(image_ids)
                image = recon.images[learn_id]
                camera = recon.cameras[image.camera_id]
                
                # gt
                gt = cv2.imread(f"{self.image_dir}/video_{video_idx:03d}/{image.name}")
                gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
                gt = torch.tensor(gt, dtype=torch.float32).cuda() / 255.0

                # View, Proj Matrix
                V = self.build_view(image)
                P, cam_data = self.build_proj(camera)

                # Tile Rasterization
                out = self.tile_rasterization(gaussians, V, P, cam_data)

                # Loss
                loss = torch.abs(out - gt).mean()

                # Adam
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                # Gaussian Refinement

    def tile_rasterization(self, gaussians, view, proj, cam_data):

        ### 1. Cull Gaussian
        fx, fy, cx, cy, cw, ch = cam_data # 초점거리, 주점, 이미지 크기

        # 시야각
        fov_x = 2 * math.atan(cw / (2 * fx))
        fov_y = 2 * math.atan(ch / (2 * fy))

        tan_half_fov_x = math.tan(fov_x / 2)
        tan_half_fov_y = math.tan(fov_y / 2)
        
        # 월드 좌표 → 동차좌표 → 카메라 공간
        xyz = gaussians[:, :3]
        N = xyz.shape[0]
        ones = torch.ones((N, 1), device=xyz.device, dtype=xyz.dtype)
        xyz_h = torch.cat([xyz, ones], dim=1)
        cam_xyz = xyz_h @ view.T
        
        # Frustum culling (near/far + FOV)
        x, y, z = cam_xyz[:, 0], cam_xyz[:, 1], cam_xyz[:, 2]
        in_frustum = (
            (z > self.near_plane) &
            (z < self.far_plane) &
            (torch.abs(x / z) < tan_half_fov_x) &
            (torch.abs(y / z) < tan_half_fov_y)
        )
        
        gaussians = gaussians[in_frustum]
        cam_xyz = cam_xyz[in_frustum]

        ### 2. Screen space
        
        ## 카메라 좌표 > Projection > divide w > NDC > 픽셀 좌표
        clip_xyz = cam_xyz @ proj.T # [N' X 4] [4 X 4] [4 X 4] = [N X 4]
        ndc = clip_xyz / clip_xyz[:, 3:4]
        pixel_x = (ndc[:, 0] + 1) * 0.5 * cw # N, 1
        pixel_y = (ndc[:, 1] + 1) * 0.5 * ch # N, 1
        depth = cam_xyz[:,2]

        ## cov 3d > cov 2d
        S = gaussians[:, 3:6]
        Q = gaussians[:, 6:10]
        
        S = torch.diag_embed(S) # N' X 3 X 3
        R = self.q2rot(Q) # N' X 3 X 3
        W = view[:3, :3] # 3 X 3
        J = self.jacobian(cam_xyz, fx, fy)
        
        cov = R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)
        cov_2d = J @ W @ cov @ W.T @ J.transpose(-1, -2) # N X 2 x 2

        cov_2d[:, 0, 0] += 0.3
        cov_2d[:, 1, 1] += 0.3

        
        ## 고유값 
        mid = 0.5 * (cov_2d[:, 0,0] + cov_2d[:, 1,1])
        det = cov_2d[:, 0,0] * cov_2d[:, 1,1] - (cov_2d[:, 0,1] ** 2)
        det = det.clamp(min=1e-7)
        eigen_value = mid + torch.sqrt(torch.clamp(mid**2 - det, min=0.1))
        radius = 3.0 * torch.sqrt(eigen_value) # (N',)

        # 역행렬
        inverse_cov2d = torch.stack([
            torch.stack([cov_2d[:, 1, 1] / det, -cov_2d[:, 0, 1] / det], dim=1),
            torch.stack([-cov_2d[:, 0, 1] / det, cov_2d[:, 0, 0] / det], dim=1),
        ], dim=-2)

        # 변수 정리
        opacity = torch.sigmoid(gaussians[:, 10]) #  alpha
        mu = torch.stack([pixel_x, pixel_y], dim=-1) # mean
        
        # SH
        f_dc = gaussians[:, 11:14].view(-1, 1, 3) # [N', 3] --> [N', 1, 3]
        f_rest = gaussians[:, 14:59].view(-1, 15, 3) # [N', 3] --> [N', 15, 3]
        sh = torch.cat([f_dc, f_rest], dim=1) # [N', 16, 3]

        # camera pos, direction
        cam_pos = -view[:3, :3].T @ view[:3, 3] # [3,]
        dir = xyz - cam_pos[None, :] # [N', 3]
        dir = F.normalize(dir, dim=-1)

        color = self.eval_sh(sh, dir)

        ### 3. Create Tile

        ## gaussian bounding box
        min_tile_x = torch.floor((pixel_x - radius) / self.tile_size).to(torch.int64) # min[a,] : a번 가우시안의 bounding box의 작은 x값 좌표
        min_tile_y = torch.floor((pixel_y - radius) / self.tile_size).to(torch.int64)
        max_tile_x = torch.floor((pixel_x + radius) / self.tile_size).to(torch.int64)
        max_tile_y = torch.floor((pixel_y + radius) / self.tile_size).to(torch.int64)
        
        ## tile grid
        grid_w = (cw + self.tile_size - 1) // self.tile_size
        grid_h = (ch + self.tile_size - 1) // self.tile_size

        box_w = max_tile_x - min_tile_x + 1
        box_h = max_tile_y - min_tile_y + 1
        
        ### 4. DuplicateWithKey

        # counts[i] = j , i : 가우시안 번호, j : 가우시안 i가 속한 타일의 수
        # offset[i] = j,   i : 가우시안 번호 (0~N), j : 가우시안 i의 시작 인덱스(누적합)
        counts = box_w * box_h 
        offsets = torch.cumsum(counts, dim=0) - counts 
        M = int(counts[-1] + offsets[-1]) # Key 배열의 크기 

        slot = torch.arange(M, device=offsets.device, dtype=torch.int64) # 전체 엔트리 슬롯 인덱스 [0, M)
        values = torch.searchsorted(offsets, slot, right=True) - 1 # values[i] = j,  i : 슬롯 인덱스,  j : 가우시안 번호
        # searchsorted(a,b) : b값이 정렬된 배열 a에서 어느 index에 들어가야 배열 a가 여전히 정렬된 상태를 유지하는지. (right : 값이 같은 경우 오른쪽, 왼쪽 결정)

        k = slot - offsets[values] # k[i] = j,  i : 슬롯 인덱스 , j : 타일 순번
        bw = box_w[values]

        tile_x = min_tile_x[values] + k % bw
        tile_y = min_tile_y[values] + k // bw
        tile_num = tile_y * grid_w + tile_x
        
        # depth float bit 
        depth_int = depth[values].view(torch.int32).to(torch.int64) & 0xFFFFFFFF 
        keys = (tile_num.to(torch.int64) << 32) | depth_int # [M, ]
        
        ### 5. SortByKey
        sorted_keys, key_idx = torch.sort(keys)
        sorted_values = values[key_idx]

        ### 6. IdentifyTileRanges
        T = grid_w * grid_h
        tile_boundary = torch.arange(T+1, device=sorted_keys.device, dtype=torch.int64) << 32
        tile_range = torch.searchsorted(sorted_keys, tile_boundary)

        tile_start = tile_range[:-1] # [i] = j , i번째 타일의 시작인덱스 j
        tile_end = tile_range[1:]

        ### 7. GetTileRange
        out = torch.zeros((int(ch), int(cw), 3), device=torch.device('cuda'))

        for t in range(T):
            start_pw = t % grid_w
            start_ph = t // grid_w
            
            px_start = start_pw * self.tile_size
            py_start = start_ph * self.tile_size
            px_end = min((start_pw + 1) * self.tile_size, cw)
            py_end = min((start_ph + 1) * self.tile_size, ch)

            px = torch.arange(px_start, px_end, device=torch.device('cuda'))
            py = torch.arange(py_start, py_end, device=torch.device('cuda'))

            PY, PX = torch.meshgrid(py, px, indexing='ij')
            pixels = torch.stack([PX, PY], dim=-1).reshape(-1, 2) # [256, 2]

            # g_t : 현재 타일 t에 쓰일 가우시안들의 번호를 depth를 기준으로 정렬한 배열
            G_t = sorted_values[tile_start[t]:tile_end[t]] # [G,]
            inverse_cov2d_t = inverse_cov2d[G_t] # [G, 2, 2]
            opacity_t = opacity[G_t] # [G,]
            color_t = color[G_t] # [G,3]
            mu_t = mu[G_t] # [G,2]

            ### 8. BlendInOrder

            # gaussian alpha
            ## alpha = alpha * (x-m)^T * cov * (x-m)
            ## d = x-m
            d = pixels[:, None, :] - mu_t[None, :, :]  # [256, G, 2]

            power = torch.einsum('pgi,gij,pgj->pg', d, inverse_cov2d_t, d) # [P,G]
            alpha_t = opacity_t * torch.exp(-0.5 * power)
            alpha_t = alpha_t.clamp(max=0.99)

            # gaussian alpha blending
            # color = sum(T * a * c)

            one_minus_alpha = 1 - alpha_t # [P,G]
            T_inclusive = torch.cumprod(one_minus_alpha, dim=1) # [P,G]
            T_acc = torch.cat([torch.ones_like(T_inclusive[:, :1]), T_inclusive[:, :-1]], dim=1) # [P,G]
            contrib = T_acc * alpha_t # [P,G]
            
            c = (contrib[:, :, None] * color_t[None, :, :]).sum(dim=1) # [P,G,3] -> [P,3]
            c_2d = c.reshape(py_end - py_start, px_end - px_start, 3)
            out[py_start:py_end, px_start:px_end] = c_2d

        return out
            
    def eval_sh(self, sh, dir):
        
        dir_x, dir_y, dir_z = dir[:, 0:1], dir[:, 1:2], dir[:, 2:3]

        result = SH_C0 * sh[:, 0]

        result += -SH_C1 * dir_y * sh[:, 1]
        result += SH_C1 * dir_z * sh[:, 2]
        result += -SH_C1 * dir_x * sh[:, 3]

        dir_xx, dir_yy, dir_zz = dir_x ** 2, dir_y ** 2, dir_z ** 2
        dir_xy, dir_yz, dir_xz = dir_x * dir_y, dir_y * dir_z, dir_x * dir_z

        result += SH_C2[0] * dir_xy * sh[:, 4]
        result += SH_C2[1] * dir_yz * sh[:, 5]
        result += SH_C2[2] * (3*dir_zz - 1) * sh[:, 6]
        result += SH_C2[3] * dir_xz * sh[:, 7]
        result += SH_C2[4] * (dir_xx - dir_yy) * sh[:, 8]

        result += SH_C3[0] * dir_y * (3*dir_xx - dir_yy) * sh[:, 9]
        result += SH_C3[1] * dir_xy * dir_z * sh[:, 10]
        result += SH_C3[2] * dir_y * (5*dir_zz - 1) * sh[:, 11]
        result += SH_C3[3] * dir_z * (5*dir_zz - 3) * sh[:, 12]
        result += SH_C3[4] * dir_x * (5*dir_zz - 1) * sh[:, 13]
        result += SH_C3[5] * dir_z * (dir_xx - dir_yy) * sh[:, 14]
        result += SH_C3[6] * dir_x * (dir_xx - 3*dir_yy) * sh[:, 15]

        return (result + 0.5).clamp(min=0)
    
    def jacobian(self, cam_xyz, fx, fy):
        x, y, z = cam_xyz[:, :3].unbind(-1)
        zero = torch.zeros_like(z)

        jacobian = torch.stack([
            torch.stack([fx/z, zero, (-fx * x) / z**2], dim=-1),
            torch.stack([zero, fy/z, (-fy * y) / z**2], dim=-1),
        ], dim=-2)

        return jacobian


    # Quaternion > Rotation Matrix
    def q2rot(self, q):
        q = q / q.norm(dim=-1, keepdim=True)
        qw, qx, qy, qz = q.unbind(-1)

        rot = torch.stack([
            torch.stack([1 - 2*(qy**2 + qz**2),  2*(qx*qy - qz*qw),      2*(qx*qz + qy*qw)],      dim=-1),
            torch.stack([2*(qx*qy + qz*qw),      1 - 2*(qx**2 + qz**2),  2*(qy*qz - qx*qw)],      dim=-1),
            torch.stack([2*(qx*qz - qy*qw),      2*(qy*qz + qx*qw),      1 - 2*(qx**2 + qy**2)],  dim=-1),
        ], dim=-2)

        return rot
    
    def build_view(self, image):
        R = self.q2rot(torch.tensor(image.qvec, dtype=torch.float32, device='cuda').unsqueeze(0))[0]
        T = torch.tensor(image.tvec, dtype=torch.float32, device='cuda')
        
        V = torch.eye(4, dtype=torch.float32, device='cuda')
        V[:3, :3] = R
        V[:3, 3] = T
        
        return V.requires_grad_(False)

    def build_proj(self, camera):
        fx, fy, cx, cy = camera.params
        camera_w, camera_h = camera.width, camera.height

        # Projection Matrix
        P = torch.tensor([
            [2*fx/camera_w,      0,  1 - 2*cx/camera_w,                0],
            [     0, 2*fy/camera_h,  1 - 2*cy/camera_h,                0],
            [     0,      0,  (self.far_plane + self.near_plane)/(self.far_plane - self.near_plane),  -2*self.far_plane*self.near_plane/(self.far_plane-self.near_plane)],
            [     0,      0,  1,                        0]
        ], dtype=torch.float32).cuda().requires_grad_(False)

        return P, [fx, fy, cx, cy, camera_w, camera_h]

    
        