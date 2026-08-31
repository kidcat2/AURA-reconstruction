"""
(frames) -- [FrameSet] --> (pose) -- [CameraPoses] --> (train) -- [GaussianModel] --> (compress) -- [Splat] --> 산출물
"""

"""
data/
└── scenes/                      ← scenes_dir : 입력 자산 루트
    ├── seoul_station/           ← scene : 이번 실행에 쓸 씬 이름
    │   ├── video.mp4
    │   └── scene.yaml           (선택) 촬영 정보 메모
    │
    ├── input/                   ← 지금 config에 적힌 씬
    │   └── test.mp4
    │
    └── lego/                    ← 정답 포즈가 있는 데이터셋 씬
        ├── images/
        │   ├── 00000.jpg
        │   └── ...
        └── sparse/0/
"""

"""
[interface : frames → pose]

입력 영상   : config/default.yaml - scenes_dir / scene
    
출력 프레임 : runs/3dgs_org/frames/00000.jpg   ( config['data']['run_dir']/self.OUTPUT , self.OUTPUT = frames )

FrameSet : 실행 한 번당 하나. 영상 하나에서 뽑은 프레임 전체
    - frames        : 이 실행의 Frame 목록. 순서 = 촬영 순서
    - source_video  : 프레임을 뽑아낸 영상 파일 이름

Frame : 프레임 이미지 하나당 하나
    - name          : 파일명. 포즈 추정 결과와 이미지를 짝짓는 키
    - image_path    : 프레임 이미지 파일 경로. train 이 정답 이미지로 읽는다

[보류] 읽는 쪽이 생기면 추가
    - width, height : 포즈는 COLMAP 카메라 값을, train 은 View 값을 쓴다
    - frame_index   : 원본 영상에서 몇 번째 프레임이었는지
    - blur_score    : 프레임 단계 내부 판단값
"""

from dataclasses import dataclass

@dataclass(eq = False)
class Frame:
    name: str
    image_path: str

@dataclass(eq = False)
class FrameSet:
    frames: list[Frame]
    source_video: str
