from aura.stages.base import BaseStage

import glob
import os
import subprocess

import sqlite3
import struct
import cv2

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

COLMAP_BIN = os.path.join(os.path.dirname(__file__), "../../../colmap/bin/colmap.exe")

class Colmap(BaseStage):

    OUTPUT = "pose"

    def __init__(self, config):

        self.output_dir = os.path.join(config["data"]["run_dir"], self.OUTPUT)

    def run(self, context):
        print("Colmap: start")

        self.input_dir = context["frames"]

        self.run_colmap()
        self.verify()
        self.make_context(context)

        print("Colmap: done")

    def run_colmap(self):

        """
        1) 경고 : frame < 100 , resolution < 256
        2) colmap : frame extractor → sequential matcher → mapper
        3) 저장 : camears.bin, images.bin, points3D.bin
            - 경로 : (config-default.yaml)
        
        """

        image_paths = sorted(glob.glob(f"{self.input_dir}/*"))

        print(f"Colmap: processing {len(image_paths)} videos")

        for idx, image_path in enumerate(image_paths):

            print(f"Colmap: video_{idx:03d} ({idx+1}/{len(image_paths)})")

            ### debug code
            frames = sorted(glob.glob(f"{image_path}/*"))
            flen = len(frames)

            if flen < 100 : 
                print("Input Frames numbers under 150")
                #exit()

            sample = cv2.imread(frames[0])
            h, w = sample.shape[:2]
            
            if h < 256 or w < 256:
                print("Frame Resolution under 256")
                #exit()
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

        """
        1) 필수 파일 확인 : camears.bin, images.bin, points3D.bin
        2) 경고 : 입력 프레임 수 - colmap 후 image 수 비교
        
        """

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
        context["poses"] = self.output_dir