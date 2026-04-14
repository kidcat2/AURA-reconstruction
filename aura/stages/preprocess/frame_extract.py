from aura.stages.base import BaseStage

import os
import glob
import cv2
import numpy as np


"""
1. config default 수정
2. 영상 열기
3. 샘플링
4. 블러 점수 계산
5. 목표 장수로 균등 다운 샘플
6. 리사이즈 & 저장
7. 자원 정리 & 로깅
8. context 채우기

"""



class FrameExtract(BaseStage):

    def __init__(self, config):

        self.input_dir = config["data"]["input_dir"]
        self.output_dir = config["data"]["frames_dir"]

        self.target_frames = config["frame_extract"]["target_frames"]

        self.blur_keep_ratio = config["frame_extract"]["blur_keep_ratio"]
        self.blur_chunk_cnt = config["frame_extract"]["blur_chunk_count"]
        self.blur_chunk_min = config["frame_extract"]["blur_min_per_chunk"]

        self.resize_long_side = config["frame_extract"]["resize_long_side"]
        self.output_format = config["frame_extract"]["output_format"]
        self.jpg_quality = config["frame_extract"]["jpg_quality"]

        self.video_count = 0

    def run(self, context):
        print("[FrameExtract] 실행")
        
        self.sampling()
        self.make_context(context)
    
    def sampling(self):

        video_paths = glob.glob(f"{self.input_dir}/*.mp4")
        self.video_count = len(video_paths)

        for i, video_path in enumerate(video_paths):

            cap = cv2.VideoCapture(video_path)

            frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT
                                      ))
            # 전체 프레임 수가 target_frames 보다 부족하면 전부 넣기
            if frame_total < self.target_frames :
                self.save_frames(cap, i, [x for x in range(frame_total)])
                cap.release()
                continue 

            # 블러 심한 이미지 거르기
            scores = self.compute_blur_scores(cap)
            keep_indices = self.filter_blur(scores)

            # 균등 다운 샘플
            if(len(keep_indices) > self.target_frames):
                indices = np.linspace(0, len(keep_indices) - 1, self.target_frames, dtype=int)
                keep_indices  = [keep_indices[i] for i in indices]
                
            self.save_frames(cap, i, keep_indices )

            cap.release()

        print("sampling complete")

    def compute_blur_scores(self, cap):
        scores = []

        while True:
            ret, frame = cap.read()

            if not ret: 
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            score = cv2.Laplacian(gray, cv2.CV_64F).var()

            scores.append(score)

        return scores
    
    def filter_blur(self, scores):
        
        keep_indices = []

        for i in range (0, len(scores), self.blur_chunk_cnt):
            value = scores[i:i+self.blur_chunk_cnt]
            batch = [(i + j, s) for j, s in enumerate(value)]

            batch.sort(key=lambda x : x[1], reverse=True)
            batch = batch[:int(self.blur_chunk_cnt * self.blur_keep_ratio)]
            batch.sort(key=lambda x : x[0])

            keep_indices.extend([x[0] for x in batch])

        return keep_indices
            
    def save_frames(self, cap, video_idx, indices):

        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        long_side = max(h,w)

        scale = self.resize_long_side / long_side
        resize_w, resize_h = int(w * scale), int(h * scale)

        output_dir = f"{self.output_dir}/video_{video_idx:03d}"
        os.makedirs(output_dir, exist_ok=True)

        for i, frame_idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if long_side > self.resize_long_side:
                frame = cv2.resize(frame, (resize_w, resize_h), interpolation=cv2.INTER_AREA)

            filename = f"{i:05d}.{self.output_format}"
            path = os.path.join(output_dir, filename)

            cv2.imwrite(path, frame)


    def make_context(self, context):
        context["frames_dir"] = self.output_dir
                





            








                
        
             
            
            

        
            
            






            

    

    
