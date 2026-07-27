from aura.stages.base import BaseStage

import glob
import os
import subprocess

import sqlite3
import struct
import cv2

COLMAP_BIN = os.path.join(os.path.dirname(__file__), "../../../colmap/bin/colmap.exe")

# 1. config default 수정 (quality preset 등)
# 2. workspace 준비 (database, sparse 디렉토리 생성)
# 3. pycolmap.extract_features 호출
# 4. pycolmap.match_exhaustive 호출
# 5. pycolmap.incremental_mapping 호출
# 6. 결과 검증 (등록된 이미지 수, 포인트 수 확인)
# 7. context 채우기 (카메라 파라미터, sparse model 경로)

# input 파일 경로 : output/frames/*
# output 파일 경로 : output/sfm/*

class Colmap(BaseStage):

    def __init__(self, config):
        self.output_dir = config["data"]["sfm_dir"]

    def run(self, context):
        print("Colmap: start")

        self.input_dir = context["frames_dir"]

        self.run_colmap()
        self.verify()
        self.make_context(context)

        print("Colmap: done")

    def run_colmap(self):

        # self.input_dir = output/frames
        # image paths = [video_001, video_002...]
        image_paths = sorted(glob.glob(f"{self.input_dir}/*"))

        print(f"Colmap: processing {len(image_paths)} videos")

        for idx, image_path in enumerate(image_paths):

            print(f"Colmap: video_{idx:03d} ({idx+1}/{len(image_paths)})")

            ### debug code
            frames = sorted(glob.glob(f"{image_path}/*"))
            flen = len(frames)

            if  flen < 100 : 
                print("Input Frames numbers under 150")
                exit()

            sample = cv2.imread(frames[0])
            h, w = sample.shape[:2]
            
            if h < 256 or w < 256:
                print("Frame Resolution under 256")
                exit()
            ###

            output_dir = f"{self.output_dir}/video_{idx:03d}"
            database_path = f"{output_dir}/database.db"
            sparse_path = f"{output_dir}/sparse"

            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(sparse_path, exist_ok=True)

            if os.path.exists(database_path):
                os.remove(database_path)

            # 이미지 특징점 추출 (PINHOLE 강제: fx,fy,cx,cy 4개 파라미터로 고정)
            self._run([
                "feature_extractor",
                "--database_path", database_path,
                "--image_path", image_path,
                "--ImageReader.camera_model", "PINHOLE",
                "--ImageReader.single_camera", "1",
                "--SiftExtraction.max_num_features", "32768",
                "--SiftExtraction.peak_threshold", "0.0066",
                "--SiftExtraction.edge_threshold", "10.0",
            ])

            # 이미지 특징점 매칭
            self._run([
                "sequential_matcher",
                "--database_path", database_path,
            ])

            # 특징점 기반 카메라 포즈, 3d point 계산
            self._run([
                "mapper",
                "--database_path", database_path,
                "--image_path", image_path,
                "--output_path", sparse_path,
            ])
    
    def _run(self, args):
        result = subprocess.run([COLMAP_BIN] + args, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout.strip())
        if result.returncode != 0:
            print(result.stderr.strip())
            raise RuntimeError(f"COLMAP {args[0]} failed")

    def verify(self):

        # self.output_dir : output/sfm
        output_paths = sorted(glob.glob(f"{self.output_dir}/*"))

        # 필수 output
        required_file = [
            "/sparse/0/cameras.bin",
            "/sparse/0/images.bin",
            "/sparse/0/points3D.bin",
        ]

        for idx, output in enumerate(output_paths):
            
            if not all(os.path.exists(f"{output}{p}") for p in required_file):
                print(f"Colmap: video_{idx:03d} missing required files")
                continue

            with open(f"{output}/sparse/0/images.bin", "rb") as f:
                registered = struct.unpack("<Q", f.read(8))[0]

            input_count = len(os.listdir(f"{self.input_dir}/video_{idx:03d}"))

            if registered != input_count:
                print(f"Colmap: video_{idx:03d} image mismatch ({registered}/{input_count})")
            else:
                print(f"Colmap: video_{idx:03d} verified ({registered} images)")

    def verify_extract(self, database_path):
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()

        # 등록된 이미지 수 조회
        cursor.execute("SELECT count(*) FROM images")
        num_images = cursor.fetchone()[0]
        print(f"Registered images: {num_images}")

        # 각 이미지당 포인트 수 출력
        cursor.execute("SELECT image_id, rows FROM keypoints")
        for image_id, num_points in cursor.fetchall():
            print(f"Image ID {image_id}: {num_points} points")

        conn.close()

    def make_context(self, context):
        context["sparse"] = self.output_dir