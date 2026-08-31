"""
(frames) -- [FramseSet] --> (pose) -- [camera_poses] --> (train) -- [GaussianModel] --> (compress)
"""


"""
[interface : pose → train]

좌표 규약 : opencv_w2c 
(+X 오른쪽 / +Y 아래 / +Z 앞, 행렬은 world→camera)

CameraPoses : 재구성 한 번당 하나. 안의 모든 view가 같은 world 원점을 공유한다
    - views       : 이 장면의 View 목록. (입력 이미지 수만큼)
    - points_xyz  : 복원된 3D 점 좌표 (M,3). 장면 전체에 하나
    - points_rgb  : 각 점의 색 (M,3). 0~1 범위
    - source      : 이 결과를 만든 모델 이름 (colmap, da3 ...)
    - scale_type  : "metric"(좌표 1 = 1m) 또는 "arbitrary"(단위 모름)

View : 이미지 하나당 하나
    - name        : 파일명. 이미지와 pose를 짝짓는 키
    - image_path  : 원본 이미지 경로
    - width       : 원본 이미지 가로
    - height      : 원본 이미지 세로
    - K           : 카메라 내부 파라미터 초점거리·주점, (3,3) Matrix 
    - w2c         : world→camera 변환 (4,4). 회전 + 평행이동
    - depth       : [선택] per-pixel 거리 (h,w). 원본 해상도와 달라도 된다
    - depth_conf  : [선택] depth 신뢰도 (h,w). 0~1
    - distortion  : [선택] 렌즈 왜곡 계수
"""

from dataclasses import dataclass
import numpy as np

@dataclass(eq=False)
class View:
    name: str
    image_path: str
    width: int
    height: int
    K: np.ndarray # float64
    w2c: np.ndarray  # float64
    depth: np.ndarray | None = None # float32
    depth_conf: np.ndarray | None = None # float32
    distortion: np.ndarray | None = None 


@dataclass(eq=False)
class CameraPoses: 
    views: list[View]
    points_xyz: np.ndarray # float32
    points_rgb: np.ndarray # float32
    source: str
    scale_type: str
    
