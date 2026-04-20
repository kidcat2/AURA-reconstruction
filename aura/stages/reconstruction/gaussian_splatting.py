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
                P, fov = self.build_proj(camera)

                # Tile Rasterization
                self.rasterization(gaussians, V, P, fov)

                # Loss

                # Adam

                # Gaussian Refinement

            

                

    def rasterization(self, gaussians, view, proj, fov):
        gaussians, mask = self.cull_gaussian(gaussians, view, fov)

    def cull_gaussian(self, gaussians, view, fov):

        fov_x, fov_y = fov
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
        
        return gaussians[in_frustum], in_frustum
    
    # Quaternion > Rotation Matrix
    def q2rot(self, q):
        q = q / np.linalg.norm(q)
        qw, qx, qy, qz = q

        rot = np.array([
            [1 - 2*(qy**2 + qz**2),  2*(qx*qy - qz*qw),      2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw),      1 - 2*(qx**2 + qz**2),  2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw),      2*(qy*qz + qx*qw),      1 - 2*(qx**2 + qy**2)]
        ])

        return rot
    
    def build_view(self, image):
        R = self.q2rot(np.array(image.qvec))
        T = np.array(image.tvec, dtype=np.float32)
        
        V = np.eye(4, dtype=np.float32)
        V[:3, :3] = R
        V[:3, 3] = T
        
        return torch.tensor(V, dtype=torch.float32).cuda().requires_grad_(False)

    def build_proj(self, camera):
        fx, fy, cx, cy = camera.params
        camera_w, camera_h = camera.width, camera.height

        # FOV 
        fov_x = 2 * math.atan(camera_w / (2 * fx))
        fov_y = 2 * math.atan(camera_h / (2 * fy))

        # Projection Matrix
        P = torch.tensor([
            [2*fx/camera_w,      0,  1 - 2*cx/camera_w,                0],
            [     0, 2*fy/camera_h,  1 - 2*cy/camera_h,                0],
            [     0,      0,  (self.far_plane + self.near_plane)/(self.far_plane - self.near_plane),  -2*self.far_plane*self.near_plane/(self.far_plane-self.near_plane)],
            [     0,      0,  1,                        0]
        ], dtype=torch.float32).cuda().requires_grad_(False)

        return P, [fov_x, fov_y]

    
        