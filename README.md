# AURA-reconstruction

동영상 입력 → 3D Gaussian Splatting(.splat) 파일 출력 파이프라인

---

## 파이프라인 흐름

```
동영상 입력 → 프레임 추출 → SfM → 3DGS 학습 → .ply → .splat
```

---

## 실행

```bash
python main.py --recipe colmap_3dgs
```

- `--recipe` 인자로 `recipes/recipes.yaml` 안의 다른 조합을 골라 실행

---

## 폴더 구조

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

### main.py
- 파이프라인 진입점. recipe 이름을 받아 Orchestrator 호출

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

### recipes/
- 파이프라인 조합 정의. 하나의 yaml에 여러 recipe를 키로 묶어 관리

### config/
- 경로, 파라미터 등 설정값

### data/
- 입력 영상 (.mp4)

### output/ (자동 생성)
```
output/
├── frames/         # 추출된 프레임 이미지 (video_000/, video_001/, ...)
├── sfm/            # SfM 결과 (database.db, sparse/)
├── ply/            # 3DGS 학습 출력 (point_cloud.ply)
└── splat/          # 최종 결과물 (.splat)
```

---

## COLMAP 바이너리

SfM Stage는 프로젝트 루트의 `colmap/bin/colmap.exe`를 직접 호출한다. COLMAP 공식 배포본을 받아 `colmap/` 폴더에 두면 된다 (git 추적 제외).
