from aura.stages.base import BaseStage

import subprocess
import pycolmap
import glob
import os

# 1. config default 수정 (quality preset, colmap 바이너리 경로 등)
# 2. workspace 준비 (database, sparse 디렉토리 생성)
# 3. colmap feature_extractor 호출 (subprocess)
# 4. colmap exhaustive_matcher / sequential_matcher 호출
# 5. colmap mapper 호출 (sparse reconstruction)
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

        for idx, image_path in enumerate(image_paths) :

            print(f"Colmap: video_{idx:03d} ({idx+1}/{len(image_paths)})")

            # output/sfm/video_00*
            output_dir = f"{self.output_dir}/video_{idx:03d}"

            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(f"{output_dir}/sparse", exist_ok=True)

            # 이미지 특징점 추출
            subprocess.run([
                "colmap",
                "feature_extractor",
                "--database_path",
                f"{output_dir}/database.db",
                "--image_path",
                f"{image_path}"
            ], check=True)

            # 이미지 특징점 매칭
            subprocess.run([
                "colmap",
                "exhaustive_matcher",
                "--database_path",
                f"{output_dir}/database.db"
            ], check=True)

            # 특징점 기반 카메라 포즈, 3d point 계산

            subprocess.run([
                "colmap",
                "mapper",
                "--database_path",
                f"{output_dir}/database.db",
                "--image_path",
                f"{image_path}",
                "--output_path",
                f"{output_dir}/sparse"
            ], check=True)
    
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

            recon = pycolmap.Reconstruction(f"{output}/sparse/0")

            input_count = len(os.listdir(f"{self.input_dir}/video_{idx:03d}"))
            registered = len(recon.images)

            if registered != input_count:
                print(f"Colmap: video_{idx:03d} image mismatch ({registered}/{input_count})")
            else:
                print(f"Colmap: video_{idx:03d} verified ({registered} images)")

    def make_context(self, context):
        context["sparse"] = self.output_dir