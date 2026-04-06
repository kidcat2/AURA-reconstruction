# AURA-reconstruction

동영상 입력 → 3D Gaussian Splatting(.splat) 파일 출력 파이프라인

---

## 파이프라인 흐름

```
동영상 입력 → 프레임 추출 → SfM → 3DGS 학습 → .ply → .splat
```

- Stage 간 데이터는 메모리에서 직접 전달
- 파일 저장은 파이프라인 흐름과 분리, 별도 처리

---

## 실행

```bash
python main.py --recipe colmap_3dgs --input video.mp4
```

- `--recipe` 변경으로 다른 모델 조합 실행 가능

---

## 폴더 구조

```
AURA-reconstruction/
├── main.py
├── src/
│   ├── pipeline/
│   ├── stages/
│   └── common/
├── recipes/
├── config/
└── data/
```

### main.py

- 파이프라인 진입점
- recipe 이름 + 입력 파일을 받아 Orchestrator 호출

### src/pipeline/

파이프라인 실행 엔진

| 파일 | 역할 |
|------|------|
| orchestrator.py | Recipe yaml 로딩 → Stage 순서대로 실행 |
| registry.py | yaml 문자열 → 실제 Stage 클래스 매핑 |

### src/stages/

각 처리 단계 구현체. 역할별 폴더 분리

| 폴더 | 역할 |
|------|------|
| base.py | Stage 인터페이스 (ABC) |
| preprocess/ | 영상 → 프레임 추출 + 품질 필터링 |
| sfm/ | 카메라 파라미터 + 포인트 클라우드 생성 |
| reconstruction/ | 3DGS 학습 → .ply 출력 |
| postprocess/ | 포맷 변환 (.ply → .splat) |
| e2e/ | SfM 없이 영상 → .splat 직행 (향후) |

- 새 모델 추가 시: 해당 역할 폴더에 .py 추가 + registry 등록 + recipe 작성

### src/common/

- Stage 간 공유 타입 정의
- 예시: 프레임 목록, 포인트 클라우드 경로, 카메라 파라미터, 학습 결과 등 Stage 간 전달되는 데이터 구조

### recipes/

파이프라인 조합 정의 yaml. 어떤 Stage를 어떤 순서로 실행할지 결정

| 파일 | 조합 |
|------|------|
| colmap_3dgs.yaml | 프레임 추출 → COLMAP → 3DGS → ply_to_splat |

### config/

- 경로, 파라미터 등 설정값
- 환경(로컬/클라우드)에 따라 config만 교체

### data/

gitignore 대상

```
data/
├── input/          # 입력 영상
├── frames/         # 추출된 프레임 이미지
├── sfm/            # SfM 결과 (포인트 클라우드, 카메라 파라미터)
├── reconstruction/ # 3DGS 학습 출력 (.ply)
└── output/         # 최종 결과물 (.splat)
```

---

## 아키텍처

**Stage + Recipe 구조**

- **Stage** : 하나의 처리 단위. `run()`으로 이전 결과 수신 → 처리 → 리턴
- **Recipe** : Stage 조합을 yaml로 정의. 모델마다 레시피가 다름
- **Orchestrator** : Recipe 로딩 → Registry에서 Stage 조회 → 순서대로 실행

- 새 모델 대응: Stage 파일 추가 + Recipe yaml 작성으로 경로 분기
- Orchestrator, 기존 Stage 수정 불필요
