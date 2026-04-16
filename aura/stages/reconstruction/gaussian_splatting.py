from aura.stages.base import BaseStage

import os
import glob
import pycolmap
import random
import numpy as np
from sklearn.neighbors import KDTree # 최근접 이웃탐색
import cv2
import torch

# 1. config default 수정 (iteration, learning rate 등)
# 2. workspace 준비 (output 디렉토리 생성)
# 3. COLMAP 결과 로드 (cameras, images, points3D)
# 4. 가우시안 초기화 (포인트 클라우드 → 위치, 색상, 불투명도, 공분산)
# 5. 학습 루프 (렌더링 → 손실 계산 → 역전파 → densify/prune)
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
class GaussianSplatting(BaseStage):

    def __init__(self, config):

        self.iter = config["gaussian_splatting"]["iteration"]
        self.lr = config["gaussian_splatting"]["learning_rate"]

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
        gaussians = np.concatenate([xyz, scale, q, alpha, f_dc, f_rest], axis=1)
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


        