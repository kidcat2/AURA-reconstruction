from aura.stages.base import BaseStage

import os
import glob
import shutil
import cv2
import numpy as np

from aura.contracts import Frame, FrameSet


"""
1. config default 수정
2. 영상 열기
3. 샘플링
4. 블러 점수 계산
5. 목표 장수로 균등 다운 샘플
6. 리사이즈 & 저장
7. 자원 정리 & 로깅
8. FrameSet 반환

"""

class FrameExtract(BaseStage):

    OUTPUT = "frames"

    def __init__(self, config):

        scenes_dir = config["data"]["scenes_dir"]
        scene = config["data"]["scene"]

        self.input_dir = os.path.join(scenes_dir, scene)
        self.output_dir = os.path.join(config["data"]["run_dir"], self.OUTPUT)

        ## 세팅값

        self.target_frames = config["frame_extract"]["target_frames"]

        self.blur_keep_ratio = config["frame_extract"]["blur_keep_ratio"]
        self.blur_chunk_cnt = config["frame_extract"]["blur_chunk_count"]
        self.blur_chunk_min = config["frame_extract"]["blur_min_per_chunk"]

        self.resize_long_side = config["frame_extract"]["resize_long_side"]
        self.output_format = config["frame_extract"]["output_format"]
        self.jpg_quality = config["frame_extract"]["jpg_quality"]

    def run(self, context):
        print("FrameExtract start")
        
        context["frames"] = self.sampling()
    
    def sampling(self) -> FrameSet:

        videos = glob.glob(f"{self.input_dir}/*.mp4")

        if len(videos) == 0:
            raise FileNotFoundError(f"FrameExtract: {self.input_dir} 에 mp4 없음")
        if len(videos) > 1:
            raise ValueError(f"FrameExtract: 영상 1개만 허용, {len(videos)}개 발견 ({self.input_dir})")

        video_path = videos[0]
        source_video = os.path.basename(video_path)

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():                                  # 손상·코덱 문제를 0 나눗셈 대신 여기서 잡음
            raise RuntimeError(f"FrameExtract: 영상 열기 실패 ({video_path})")

        try:
            frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if frame_total < self.target_frames:
                keep_indices = list(range(frame_total))
            else:
                scores = self.compute_blur_scores(cap)
                keep_indices = self.filter_blur(scores)

                if len(keep_indices) > self.target_frames:
                    indices = np.linspace(0, len(keep_indices) - 1, self.target_frames, dtype=int)
                    keep_indices = [keep_indices[i] for i in indices]

            frames = self.save_frames(cap, keep_indices)
        finally:
            cap.release()                                       # 예외로 빠져나가도 영상 파일 핸들 누수 방지

        print(f"FrameExtract: {source_video} → {len(frames)} frames")

        return FrameSet(frames=frames, source_video=source_video)

    def compute_blur_scores(self, cap):
        scores = []

        while True:
            ret, frame = cap.read()

            if not ret: 
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            score = cv2.Laplacian(gray, cv2.CV_64F).var() # type: ignore

            scores.append(score)

        return scores
    
    def filter_blur(self, scores):
        
        keep_indices = []

        for i in range (0, len(scores), self.blur_chunk_cnt):
            value = scores[i:i+self.blur_chunk_cnt]
            batch = [(i + j, s) for j, s in enumerate(value)]

            batch.sort(key=lambda x : x[1], reverse=True)

            if len(batch) <= self.blur_chunk_min:
                batch = batch[:self.blur_chunk_min]
            else:
                batch = batch[:int(self.blur_chunk_cnt * self.blur_keep_ratio)]

            batch.sort(key=lambda x : x[0])

            keep_indices.extend([x[0] for x in batch])

        return keep_indices
            
    def save_frames(self, cap, indices) -> list[Frame]:

        frames = []

        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        long_side = max(h,w)

        scale = self.resize_long_side / long_side
        resize_w, resize_h = int(w * scale), int(h * scale)

        # 재실행 시 이전 결과가 섞이지 않도록 폴더째 비우고 시작
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)

        for i, frame_idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:                                         # 없는 파일을 가리키는 Frame 이 목록에 섞이는 것 방지
                raise RuntimeError(f"FrameExtract: {frame_idx} 번 프레임 읽기 실패")

            if long_side > self.resize_long_side:
                frame = cv2.resize(frame, (resize_w, resize_h), interpolation=cv2.INTER_AREA)

            filename = f"{i:05d}.{self.output_format}"
            path = os.path.join(self.output_dir, filename)

            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpg_quality])

            frames.append(Frame(name=filename, image_path=path))

        return frames

                





            








                
        
             
            
            

        
            
            






            

    

    
