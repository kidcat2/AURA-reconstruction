from aura.stages.base import BaseStage

import os
import glob
import pycolmap
import random
import numpy as np
from sklearn.neighbors import KDTree # 최근접 이웃탐색
import cv2
import torch
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

        # alpah 
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
            gaussians = self.init_gaussian(recon.points3D)

            image_ids = sorted(recon.images.keys())

            for _ in range(self.iter):
                
                # 랜덤 샘플링
                learn_id = random.choice(image_ids)
                image = recon.images[learn_id]
                camera = recon.cameras[image.camera_id]
                
                # gt
                gt = cv2.imread(f"{self.image_dir}/video_{video_idx:03d}/{image.name}")
                gt = torch.tensor(gt, dtype=torch.float32).cuda() / 255.0

                # View, Proj Matrix
                V = self.build_view(image)
                P, cam_data = self.build_proj(camera)

                # Tile Rasterization
                self.tile_rasterization(gaussians, V, P, cam_data)

                # Loss

                # Adam

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
        J = self.jacobian(cam_xyz[in_frustum], fx, fy)
        
        cov = R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)
        cov_2d = J @ W @ cov @ W.T @ J.transpose(-1, -2) # N X 2 x 2

        cov_2d[:, 0, 0] += 0.3
        cov_2d[:, 1, 1] += 0.3

        ## 고유값 closed-form ( = 가우시안 타원 장축)
        mid = 0.5 * (cov_2d[:, 0,0] + cov_2d[:, 1,1])
        det = cov_2d[:, 0,0] * cov_2d[:, 1,1] - (cov_2d[:, 0,1] ** 2)
        eigen_value = mid + torch.sqrt(torch.clamp(mid**2 - det, min=0.1))
        radius = 3.0 * torch.sqrt(eigen_value) # (N',)

        ## bounding box
        min_tile_x = torch.floor((pixel_x - radius) / self.tile_size).to(torch.int64) # min[a,] : a번 가우시안의 bounding box의 작은 x값 좌표
        min_tile_y = torch.floor((pixel_y - radius) / self.tile_size).to(torch.int64)
        max_tile_x = torch.floor((pixel_x + radius) / self.tile_size).to(torch.int64)
        max_tile_y = torch.floor((pixel_y + radius) / self.tile_size).to(torch.int64)

        grid_w = (cw + self.tile_size - 1) // self.tile_size
        grid_h = (ch + self.tile_size - 1) // self.tile_size

        box_w = max_tile_x - min_tile_x + 1
        box_h = max_tile_y - min_tile_y + 1

        ### 3. Create Tile
        counts = box_w * box_h # counts[i] = j , i : 가우시안 번호, j : 가우시안 i가 속한 타일의 수
        offsets = torch.cumsum(counts, dim=0) - counts # offset[i] = j,   i : 가우시안 번호 (0~N), j : 가우시안 i의 시작 인덱스(누적합)
        M = counts[-1] + offsets[-1] # Key 배열의 크기 

        keys = torch.empty(M, dtype=torch.int64)        
        
        slot = torch.arange(M, device=offsets.device, dtype=torch.int64) # 전체 엔트리 슬롯 인덱스 [0, M)
        values = torch.searchsorted(offsets, slot, right=True) - 1 # values[i] = j,  i : 슬롯 인덱스,  j : 가우시안 번호

        # fancy indexing
        k = slot - offsets[values] # k[i] = j,  i : 슬롯 인덱스 , j : 타일 순번
        bw = box_w[values]

        tile_x = min_tile_x[values] + k % bw
        tile_y = min_tile_y[values] + k // bw
        tile_num = tile_y * grid_w + tile_x
        
        # depth는 float(32비트) 이므로, 32비트 그대로 int로 변환 후 상위 비트 0으로  
        depth_int = depth[values].view(torch.int32).to(torch.int64) & 0xFFFFFFFF 

        keys = (tile_num.to(torch.int64) << 32) | depth_int
        
        # searchsorted(a,b) : b값이 정렬된 배열 a에서 어느 index에 들어가야 배열 a가 여전히 정렬된 상태를 유지하는지. (right : 값이 같은 경우 오른쪽, 왼쪽 결정)


        

        ### 4. DuplicateWithKey

        ### 5. SortByKey
        ### 6. IdentifyTileRanges
        ### 7. GetTileRange
        ### 8. BlendInOrder
    
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

    
        