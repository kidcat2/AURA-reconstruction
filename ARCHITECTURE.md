# ARCHITECTURE
- aura-reconstruction의 코드 구조 및 설계 방향
- 구성
    - **Ⅰ. 전체 설계 구조** — 유지되는 것
    - **Ⅱ. 작업 순서** — 이번 작업의 단위
    - **Ⅲ. TASK** — 작업 단위의 세부


<br>

---
---

# Ⅰ. 전체 설계 구조

---
---

## 1. config, context, contracts

1. config 
- 실행 전 사람이 정하는 값 : 경로, 하이퍼 파라미터, 외부 바이너리 위치
- 실행 중 불변

2. contracts
- stage 사이를 오가는 데이터 형식 : 필드 이름, 타입 좌표 규약
- `데이터 형식`을 고정해야, 다른 모델, 알고리즘이 사용되도 해당 형식에 맞게 만들면 문제 없이 진행된다

3. context
- 실행 중 실제로 만들어진 데이터 : contracts의 `instance`
- stage 끼리 직접 부르지 않고 이 그릇으로만 주고받는다


---

## 2. 단계 간 포맷 (contracts)

- 각 포맷은 두 단계 사이에 위치 — 앞 단계가 만들고 뒤 단계가 받음
- 정의 위치 : `aura/contracts/`

```
입력 영상
   │  frames/
   ▼
FrameSet          영상에서 뽑은 사진들
   │  pose/
   ▼
CameraPoses       각 사진의 카메라 위치·각도
   │  train/
   ▼
GaussianModel     3D 결과 (가우시안)
   │  compress/
   ▼
Splat             전달용 최종 포맷
   │
   ▼
최종 산출물
```

| 포맷 | 위치 (단계 사이) | 담는 것 | 확정 |
|------|------|------|------|
| FrameSet | frames → pose | 영상에서 뽑은 사진들 | 필드 확정 |
| CameraPoses | pose → train | 각 사진의 카메라 위치·각도 | 필드 확정 |
| GaussianModel | train → compress | 3D 결과 (가우시안) · PLY | 미정 |
| Splat | compress → 출력 | 전달용 최종 포맷 · SPZ | 미정 |

- 형식 목록은 `aura/contracts/__init__.py` 에 모아 노출

### FrameSet
- 위치 : frames/ 출력 → pose/ 입력
- 정의 : `aura/contracts/frame_set.py`

| | 필드 |
|------|------|
| Frame (이미지당) | name · image_path |
| FrameSet (실행당) | frames · source_video |

- 담는 기준 : 다음 단계가 실제로 읽는 것만 (부록 A)

### CameraPoses
- 위치 : pose/ 출력 → train/ 입력
- 정의 : `aura/contracts/camera_poses.py`
- 좌표 규약 : opencv_w2c 고정. 규약이 다른 모델은 생산자가 변환

| | 필드 |
|------|------|
| View (이미지당) | name · image_path · width · height · K (3,3) · w2c (4,4) · [선택] depth · depth_conf · distortion |
| CameraPoses (장면당) | views · points_xyz (M,3) · points_rgb (M,3) · source · scale_type |

- 담는 기준 : 다음 단계 모델이 실제로 읽는 것만 (부록 A)
- `validate()` : pose 단계 완결 시 함께 작성
- io : DA3 까지 붙고 나서

### GaussianModel
- 위치 : train/ 출력 → compress/ 입력
- 파일 포맷 : PLY (원본 3DGS 규격, 59채널 + normal 6)
- 미정

### Splat
- 위치 : compress/ 출력 → 최종 산출물
- 파일 포맷 : SPZ — SH 유지 (viewer 의 SH 평가 · §4 조명 계산에 필요). `.splat` 은 SH 없음 → 제외
- 미정


---

## 3. 폴더 구조

```
AURA-reconstruction/
├── run.py                     연구용 진입점
├── serve/                     배포용 진입점
│
├── aura/                      모델 코드
│   ├── contracts/                단계 간 포맷
│   ├── pipeline/                 orchestrator · registry
│   ├── stages/                   단계별 부품
│   │   ├── frames/                  영상 → 프레임
│   │   ├── pose/                    프레임 → 카메라 포즈
│   │   ├── train/                   포즈 → 가우시안
│   │   └── compress/                가우시안 → 전달 포맷
│   └── research/                 연구 기록
│       ├── eval/                    평가 점수
│       ├── report/                  report.json 생성
│       └── media/                   영상 · 비교 이미지
│
├── recipes/                   단계 조합
│   ├── research/                 연구용
│   └── production.lock.yaml      배포용
├── environments/              conda 환경
├── config/                    경로 · 하이퍼파라미터
├── colmap/                    COLMAP 바이너리
│
├── data/                      입력
│   └── scenes/{씬}/              영상 (또는 이미지 + GT 포즈) · scene.yaml
│
├── runs/                      연구 출력
│   └── {날짜}_{실험명}/
│       ├── frames/                  stages/frames 결과
│       ├── pose/                    stages/pose 결과
│       ├── train/                   stages/train 결과
│       ├── compress/                stages/compress 결과
│       ├── research/                research 결과
│       └── report.json
│
└── jobs/                      배포 출력
    └── {임의 id}/                frames · pose · train · compress
```

- git 추적 제외 : colmap/ · data/ · runs/ · jobs/
- 출력 폴더 이름 = stages/ 폴더 이름 — 부품(colmap/da3)이 바뀌어도 폴더 불변


---

## 4. 롤백

- 배포본 = 코드 + 환경 + 가중치 세 벌

| | 어디에 | 되돌리는 법 |
|------|------|------|
| 코드 | git | 태그 체크아웃 |
| 환경 | environments/*.yml | git 에 있어 함께 돌아감 |
| 가중치 | git 밖 (수 GB) | `production.lock.yaml` 에 적힌 버전으로 다시 받음 |

- 첫 배포부터 git 태그


<br>

---
---

# Ⅱ. 작업 순서

---
---

```
1. 폴더·파일 구조 변경                              [완료]
2. 기존 코드 이식 — import·경로 수정, 오가는 키 파악     [완료]        → TASK 001~005
3. DA3 단독 실행 — 출력 형태 확인                     [완료]
4. 공통 규약 확정 — contracts · 경로 · context 키       [진행 중]      → TASK 006~008
5. 단계별 완결 — frames → pose → train → compress                     → TASK 009~015
6. DA3 pose 부품 추가                                                → TASK 016
7. recipes 분리 — production.lock + research/
       └ lock 파일에 가중치 버전 칸
       └ recipe 가 stage 별 값을 덮어쓸 수 있게 (default → recipe → CLI)
8. [배포 시] serve/ + 스모크 테스트
       └ 첫 배포부터 git 태그
```

- 3 을 4 앞에 두는 이유 : COLMAP · DA3 두 출력을 다 본 뒤 공통 형식을 잡기 위해
- 3 에서 본 것 : DA3-BASE 22장 중 2장이 회전 44°·23° 오차 → 틀린 뷰 제외는 producer 몫 (부록 A X2)
- 5 를 단계별로 끊는 이유 : 한 단계를 열면 형식·경로·값·버그를 한 번에 끝내고 그 파일을 다시 열지 않기 위해
- 4 가 5 앞에 오는 이유 : 규약이 흔들리면 "끝낸" 파일을 다시 열게 됨


<br>

---
---

# Ⅲ. TASK

---
---

## 기존 코드 이식 (작업 순서 2)

- 원칙 : 로직은 건드리지 않음. 경로·이름만. 에러가 뒤 단계로 넘어가면 통과

### TASK 001 — 부품 목록의 import 경로 복구 [완료]
- 폴더 이름 변경으로 깨진 import 경로 교체

### TASK 002 — 레시피 파일 경로 복구 [완료]
- orchestrator 가 옮겨간 레시피 위치를 찾도록 수정

### TASK 003 — 임시로 넣어둔 context 제거 [완료]
- 디버깅용으로 채워둔 값 삭제, 빈 상태로 시작

### TASK 004 — compress 단계를 공통 규약에 맞춤 [완료]
- ply_to_splat.py 를 다른 부품과 같은 모양(생성자 · run)으로 정리 — 변환 로직은 미작성

### TASK 005 — 오가는 데이터 목록 기록 [완료]
- 부품별 입출력 값을 전부 적고 필수 · 선택 · 보류로 분류 → 부록 A


---

## 공통 규약 확정 (작업 순서 4)

- 원칙 : 코드를 열기 전에 전 단계가 공유하는 규칙부터 못 박음

### TASK 006 — CameraPoses 형식 정의 [완료]
- camera_poses.py 작성 — View · CameraPoses 필드 확정
- 폴더 경로 필드 삭제, 점 좌표 · 점 색 이름 통일

### TASK 007 — FrameSet 형식 정의 [완료]
- frame_set.py 작성 — Frame · FrameSet 필드 확정
- contracts/__init__.py 에 형식 목록 모아 노출

### TASK 008 — 경로 · context 키 규칙 확정 [완료]
- config 경로를 세 줄로 축소 — scenes_dir · scene · run_dir
- 부품마다 OUTPUT 상수 추가 — 출력 폴더 이름을 코드에 고정
- context 키를 frames · poses · gaussians · splat 으로 교체
- 입력을 data/scenes 로 이동, runs · jobs 를 git 제외에 추가


---

## 단계별 완결 (작업 순서 5)

- 원칙 : 한 단계를 열면 형식 · 경로 · 값 · 버그를 한 번에 끝내고 그 파일을 다시 열지 않음
- 순서는 파이프라인 순서와 동일

### TASK 009 — 프레임 단계 완결 [완료]
- 1실행-1영상에 맞춰 frame_extract.py 수정 — 영상 반복문 · 영상별 하위 폴더 삭제
- FrameSet 을 조립해 넘김 — 기존에는 폴더 경로
- 재실행 시 출력 폴더 비우기 (부록 B)
- 에러 · 로그 메시지 정리

### TASK 010 — 포즈 단계 완결 [진행]
- 1실행-1영상에 맞춰 colmap.py 수정 — 영상 반복문 · 영상별 하위 폴더 삭제
- 입력을 FrameSet 으로 — 이미지 목록에서 경로를 꺼내 씀
- COLMAP 결과를 CameraPoses 로 조립해 넘김 — 기존에는 결과 폴더 경로
- train 의 카메라 행렬 코드를 pose 로 이동
- COLMAP 실행 파일 위치 · 특징점 파라미터를 config 로 이동
- CameraPoses 검증 함수 작성 + 넘기기 직전 호출
- 디버그 블록 정리 (부록 B), 에러 · 로그 메시지 정리

### TASK 011 — 학습 단계 완결
- 1실행-1영상에 맞춰 gaussian_splatting.py 수정
- 입력을 CameraPoses 로 — COLMAP 파일 직접 읽기 제거
- 손실 합계 변수 초기화 버그 수정 (부록 B)
- 정답 이미지를 미리 올려두고 재사용 — 반복마다 재로드 제거 (부록 B)
- 에러 · 로그 메시지 정리

### TASK 012 — 압축 단계 완결
- GaussianModel · Splat 형식 정의
- ply_to_splat.py 변환 로직 작성 — 지금은 껍데기
- 에러 · 로그 메시지 정리

### TASK 013 — 전 구간 통과 확인
- 데이터셋 확보 후 처음부터 끝까지 실행
- 실행 폴더 하나에 네 단계 결과가 모두 생성되는지 확인

### TASK 014 — 받는 것 · 주는 것 선언 추가
- 부품마다 어떤 키를 받고 주는지 선언
- orchestrator 가 실행 전 레시피 순서를 검사 — 어긋나면 실행 없이 오류

### TASK 015 — 부록 A 갱신
- 010 · 011 에서 실제로 쓴 필드와 대조, 달라진 항목 수정


---

## DA3 pose 부품 추가 (작업 순서 6)

### TASK 016 — DA3 포즈 부품 추가
- DA3 결과를 CameraPoses 로 변환하는 부품 작성
- 부품 목록 · 레시피에 등록


<br>

---
---

# 부록

---
---

## 부록 A — 오가는 키

- TASK 005 결과. CameraPoses 필드 선정 근거
- 윗줄은 포즈 추정 쪽 의미, `사용` 은 3DGS 쪽 용도

[필수] — 없으면 3DGS 학습 불성립. 1~5는 뷰마다, 6~7은 장면에 하나

1. name        : 이 추정 결과가 어느 입력 프레임의 것인지 가리키는 식별자
   - 사용 : 이미지와 pose 를 짝지어 로드
2. image       : 촬영된 원본 프레임
   - 사용 : 렌더 결과와 비교하는 학습 정답
3. width/height: 원본 프레임의 픽셀 크기. K 가 정의된 좌표계의 크기
   - 사용 : 렌더 해상도 결정, K 해석의 기준
4. K           : 카메라 내부 파라미터 (초점거리, 주점)
   - 사용 : 3D 가우시안을 2D 화면으로 투영
5. pose        : 카메라 외부 파라미터 (회전, 평행이동, world→camera)
   - 사용 : 여러 뷰를 하나의 3D 좌표계로 묶음
6. points_xyz  : 복원한 3D 점의 좌표
   - 사용 : 가우시안 초기 배치
7. points_rgb  : 각 3D 점의 색
   - 사용 : 가우시안 초기 색 (SH 0차)

[선택] — 특정 3DGS 변종이 활용

8.  depth       : 뷰별 per-pixel 거리 추정값
    - 사용 : 깊이 정규화
9.  depth_conf  : depth 신뢰도
    - 사용 : 신뢰 낮은 픽셀을 정규화에서 제외
10. distortion  : 렌즈 왜곡 계수
    - 사용 : 투영 시 왜곡 보정

[보류] — 제외. 기준 : 소비자 코드에 읽는 줄이 없음

X1. points_conf : 점의 추정 품질 — 3DGS·gsplat 모두 xyz·rgb 만 읽음. 나쁜 점은 producer 가 걸러서 줌
X2. pose_conf   : 뷰의 pose 신뢰도 — 3DGS 는 모든 뷰 동등. 틀린 뷰 제외는 producer 몫
X3. native_dir  : 모델 고유 산출물 위치 — `run_dir/pose` 로 고정이라 유추 가능

[meta] — 데이터에 대한 정보

13. source      : 포즈 추정 모델 이름 (colmap, da3 ...)
14. scale_type  : metric / arbitrary
15. convention  : opencv_w2c 고정. 필드가 아니라 문서로 고정


## 부록 B — 기록해둘 로직 버그

- `frame_extract.save_frames()` — 출력 폴더를 비우지 않음 → TASK 009
- `colmap.py` — 프레임 100장 미만 `exit()` 디버그 블록 잔존 (주석 처리됨) → TASK 010
- `gaussian_splatting.py` — `loss_sum` 미초기화 → TASK 011
- `gaussian_splatting.py` — iteration 마다 GT 이미지를 디스크에서 재로드 (`cv2.imread` + 변환 + GPU 업로드 반복) → TASK 011
