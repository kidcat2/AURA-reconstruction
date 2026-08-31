# AURA-reconstruction

Video → 3D Gaussian Splatting


## 1. Install

```bash
conda env create -f environments/[recipe].yml
conda activate [recipe]
```

*[recipe] → [5. Recipes](#5-recipes)*

---

## 2. Run

### Research

See what you can run

```bash
python run.py --list
```

Run a verified recipe

```bash
python run.py --recipe colmap_3dgs
```

Or mix your own

```bash
python run.py --frames blur --pose da3 --train gsplat --compress spz
```

### Deploy

```bash
python -m serve
```

*Site : [AURA](http://localhost:5173)*

---

## 3. Folder Structure

```
AURA-reconstruction/
├── run.py
├── serve/
├── aura/
│   ├── contracts/
│   ├── pipeline/
│   │   ├── orchestrator.py
│   │   └── registry.py
│   ├── stages/
│   │   ├── base.py
│   │   ├── frames/
│   │   ├── pose/
│   │   ├── train/
│   │   └── compress/
│   └── research/
│       ├── eval/
│       ├── report/
│       └── media/
├── recipes/
│   ├── research/
│   └── production.lock.yaml
├── environments/
├── config/
├── colmap/
├── data/
│   └── scenes/
├── runs/
└── jobs/
```

- run.py : 진입점 (연구용). 레시피 자유 · 평가·기록 동반 · 중간 산출물 보존
- serve : 진입점 (배포용). 레시피 고정 · 평가·기록 없음 · 중간 산출물 삭제 

- aura : 모델 전체 코드

- recipes : 모델 선택
    - research/ : 연구용 (자유롭게 조합)
    - production.lock.yaml : 배포용 (고정)

- environments : 레시피별 conda 환경 (레시피 이름 = 파일 이름 = 환경 이름)

- config : 경로, 하이퍼파라미터 값

- colmap : colmap 외부 라이브러리 (git 추적 제외)

- data/scenes : 입력 씬 (씬별 영상 또는 이미지 + 정답 포즈)

- runs : 연구 출력 (실행마다 폴더 하나)

- jobs : 배포 출력

---

## 4. Model Pipeline

- Pipeline + recipes
    - 모델 선정
    - 단계(Stage) 조합

- Contracts
    - 단계별 input, output 스키마

- Stages
    - base.py : 각 단계의 공통 인터페이스
    - 단계 : 총 4단계 구성
        - frames : 영상 전처리, 프레임화
        - pose : 프레임 → 카메라 파라미터
        - train : 모델 훈련
        - compress : 결과 압축, 양자화

- Research
    - 연구용 결과 제작 코드
        - eval : 평가 점수 계산
        - report : 연구용 수치 결과
        - media : 연구용 영상, 비교 이미지

---

## 5. Recipes

- colmap_3dgs
    - pose : COLMAP
    - train : 3D Gaussian Splatting (Kerbl et al., 2023)

- da3_gsplat
    - pose : Depth Anything 3 (ByteDance, 2026)
    - train : gsplat + MCMC (Kheradmand et al., 2024)
    - compress : SPZ (Niantic)