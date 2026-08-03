# AURA-reconstruction

동영상 입력 → 3D Gaussian Splatting(.splat) 파일 출력 파이프라인

> 코드 구조·설계 방향(연구/프로덕션 분리, 재구성 계획)은 [ARCHITECTURE.md](ARCHITECTURE.md) 참조

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
├── main.py             # 진입점: recipe 이름을 받아 Orchestrator 호출
├── aura/
│   ├── pipeline/       # orchestrator(실행) · stages(이름 → 클래스 매핑)
│   └── stages/         # 처리 단계 구현체 (preprocess · sfm · reconstruction · postprocess)
├── recipes/            # 파이프라인 조합 정의 (yaml)
├── config/             # 경로 · 파라미터 설정값
└── data/               # 입력 영상 (.mp4)
```

각 파일 역할·설계·목표 구조는 [ARCHITECTURE.md](ARCHITECTURE.md) 참조.

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
