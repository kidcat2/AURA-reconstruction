# ARCHITECTURE
- aura-reconstruction의 코드 구조 및 설계 방향.


---

## 설계 목표

연구와 프로덕션, 두 가지를 한 코드베이스에서 돌린다.
(프로덕션 = 배포되어 유저에게 서비스되는 쪽. '상업 사용 가능 라이선스'의 상업과 구분하기 위해 이 이름을 쓴다)

| 구분 | 연구 (research) | 프로덕션 (production) |
|------|------|------|
| 언제 | 내가 원할 때만 | 배포 후 항상 |
| 레시피 | 자유롭게 교체 | 하나로 고정 |
| 중간 산출물 | 남김 (분석용) | 삭제 |
| 목적 | 파이프라인 실험 | 안정적 서비스 |

- 심장부(부품 + 실행 엔진)는 공유하고, 진입점만 둘로 나눈다.
- 원칙: 연구를 아무리 굴려도 프로덕션은 항상 돌아간다.

---

## 두 개의 진입점

```
심장부 (공유) : 부품 + 단계 간 포맷 + 실행 엔진
       │
   ┌───┴───┐
 실험대       자판기
 (연구)     (프로덕션)
 run.py      serve/
```

| 진입점 | 역할 |
|------|------|
| run.py (실험대) | 수동 실행. 아무 레시피나. 평가·기록 항상 동반. 결과·중간물 보존 |
| serve/ (자판기) | 외부 호출용. 고정 레시피. 진행 상태를 파일로 기록. 끝나면 중간물 정리 · 배포 시 추가 |

둘 다 같은 심장부를 호출한다. 실행 방식만 다르다.

serve는 영상 경로를 받아 splat과 상태 파일을 내놓는 것까지만 책임진다.
누가 어떤 환경에서 호출하는지는 이 저장소의 관심사가 아니다.

### 평가·기록은 실험대에만 있다

연구는 모델을 만드는 일이라 학습·평가·손실이 항상 함께 돌고 매번 저장된다.
프로덕션은 완성된 모델로 결과만 뽑는 추론이라 점수도 기록도 필요 없고, 남기지도 않는다.

| 구분 | 실험대 (연구) | 자판기 (프로덕션) |
|------|------|------|
| 평가 (점수 산출) | 항상 | 없음 |
| 실험 기록 (report·wandb) | 항상 (실패한 실험도) | 없음 |
| 시각 자료 (궤적 영상·비교 이미지) | 항상 | 없음 |

- 켜고 끄는 옵션이 아니다. 연구 경로에는 항상 있고, 프로덕션 경로에는 코드 자체가 없다.
- 쌓인 기록은 자동으로 남기고, 공개할 것만 사람이 골라 쓴다 (기록은 전부 · 게시는 큐레이션).

---

## 현재 구조

```
AURA-reconstruction/
├── main.py
├── aura/
│   ├── pipeline/
│   │   ├── orchestrator.py
│   │   └── stages.py
│   └── stages/
│       ├── base.py
│       ├── preprocess/
│       ├── sfm/
│       ├── reconstruction/
│       └── postprocess/
├── recipes/
│   └── recipes.yaml
├── config/
│   └── default.yaml
└── data/
```

### aura/pipeline/
| 파일 | 역할 |
|------|------|
| orchestrator.py | config/recipes yaml 로딩 → Stage 순서대로 실행 |
| stages.py | 문자열 이름 → 실제 Stage 클래스 매핑 |

### aura/stages/
각 처리 단계 구현체. 역할별 폴더 분리

| 폴더/파일 | 역할 |
|------|------|
| base.py | Stage 인터페이스 (추상 클래스) |
| preprocess/ | 영상 → 프레임 추출 + 품질 필터링 |
| sfm/ | 카메라 파라미터 + 포인트 클라우드 생성 |
| reconstruction/ | 3DGS 학습 → .ply 출력 |
| postprocess/ | 포맷 변환 (.ply → .splat) |

한계: 단계끼리 데이터를 정해진 형식 없이 dict(context)로 넘기고, 그 안에 COLMAP 전용 이름(sparse 등)이 섞여 있다. 그래서 포즈 도구를 바꾸면 다음 단계가 깨진다.

---

## 목표 구조

```
AURA-reconstruction/
├── run.py                  # 실험대 (연구) — 현 main.py 자리
├── serve/                  # 자판기 (프로덕션) — 배포 시 추가
├── aura/                   # ── 심장부 (공유) ──
│   ├── contracts/          # 단계 간 포맷 (아래 표)
│   ├── pipeline/
│   │   ├── orchestrator.py # 레시피 읽어 순서대로 실행
│   │   └── registry.py     # 이름 → 부품 매핑 (현 stages.py)
│   ├── stages/             # 부품함: 자리(role)마다 구현체 여럿
│   │   ├── base.py
│   │   ├── frames/         # 영상 → 사진
│   │   ├── pose/           # 사진 → 포즈  : colmap · da3 · loftr
│   │   ├── train/          # 포즈 → 3D   : gsplat
│   │   └── compress/       # 3D → 전달포맷 : spz · ply_to_splat
│   └── research/           # ── 연구 전용 (프로덕션에 없음) ──
│       ├── eval/           # 평가 — 점수 산출
│       ├── report/         # 실험 기록 — report.json · wandb
│       └── media/          # 시각 자료 — 궤적 영상 · 비교 이미지 · 크롭
├── recipes/
│   ├── production.lock.yaml # 프로덕션 레시피 (고정)
│   └── research/            # 연구 레시피들 (자유)
├── config/
│   └── default.yaml
└── tests/
    └── smoke_production.py  # 프로덕션 레시피 통과 확인
```

현재 → 목표 차이

| 항목 | 현재 → 목표 |
|------|------|
| 단계 간 포맷 | dict(context) → contracts (정해진 형식) |
| 부품 배치 | 단계 이름(sfm 등) → 자리(pose 등) 밑 구현체 여럿 |
| 레시피 | 1개 → 프로덕션(고정) + 연구(자유) |
| 진입점 | main.py 하나 → run.py + serve/ |
| 평가·기록 | 없음 → research/ 신설 (연구 경로에만) |

---

## 단계 간 포맷 (contracts)

각 단계는 정해진 형식으로만 데이터를 주고받는다. 각 포맷은 파이프라인에서
특정 두 단계 사이에 위치한다 — 앞 단계가 만들고, 뒤 단계가 받는다.
부품(colmap → da3 등)을 바꿔도 이 형식은 그대로라, 다음 단계가 안 깨진다.
포맷 정의는 aura/contracts/ 에 둔다.

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
| FrameSet | frames → pose | 영상에서 뽑은 사진들 | 미정 |
| CameraPoses | pose → train | 각 사진의 카메라 위치·각도 | 미정 |
| GaussianModel | train → compress | 3D 결과 (가우시안) | 미정 |
| Splat | compress → 출력 | 전달용 최종 포맷 | 미정 |

→ 아래에서 하나씩 확정한다.

### FrameSet
위치: frames/ 출력 → pose/ 입력
미정

### CameraPoses
위치: pose/ 출력 → train/ 입력
미정

### GaussianModel
위치: train/ 출력 → compress/ 입력
미정

### Splat
위치: compress/ 출력 → 최종 산출물
미정

---

## 프로덕션 보호 장치

배포 후 연구하다 공유 부품을 깨도 프로덕션이 안 멈추게 하는 장치.

- 레시피 잠금 : 프로덕션 레시피는 고정. 연구는 기존 부품을 바꾸지 않고 옆에 새로 추가한다.
- 스모크 테스트 : 작은 샘플로 프로덕션 레시피 처음~끝 통과를 자동 확인 → 깨지면 배포 전에 걸린다.
- 배포본 분리 : 돌아가는 프로덕션은 얼린 완성품(배포 버전)이라, 작업 중인 코드와 별개다.
- 롤백 : 잘못된 버전이 나가면 이전 버전으로 되돌린다.

---

## 마이그레이션 순서

배포 전에는 실험대(run)만 있으면 된다. 자판기(serve)·스모크는 배포 시점.

```
1. contracts 정의 → 현재 dict(context)를 이걸로 교체   (제일 먼저)
2. stages를 자리(role)별로 재배치 + da3 추가
3. recipes 분리 (production.lock + research/)
4. main.py → run.py 정리 + 프로덕션 레시피 1회 테스트
5. research/ (평가·기록·시각자료) — 실험대에 붙임
6. [배포 시] serve/ + 스모크 테스트 추가
```

심장부(aura/)는 이 재배치 한 번이면, 이후 연구·프로덕션이 공유한다.
