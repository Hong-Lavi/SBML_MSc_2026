# PROJECT SNAPSHOT — SBML_MSc_2026

> Owner: Hong Lavi  
> Timezone: Asia/Seoul  
> Program: SBML_PreMaster (12-day)

---

## Day 1 — Canonical Repo + Reproducibility Scaffold (완료)

### Date
- 2026-03-03 (KST)

### Status
- 완료

### Day1 DoD 체크
- [x] Canonical repo 구조 구성 (`src/`, `scripts/`, `configs/`, `data/`, `results/`, `tests/`, `docs/`)
- [x] 환경 파일 구성 (`environment.yml`)
- [x] Sanity 산출물 생성
- [x] Seed reproducibility test 추가 (`tests/test_seed.py`)
- [x] 로그 파일 생성 (`docs/logs/day1_sanity.log`)

### Artifacts (경로)
- `configs/base.yaml`
- `scripts/run_sanity.sh`
- `scripts/sanity_python.py`
- `src/sbml_msc_2026/utils/{config.py,seed.py,logger.py,env.py}`
- `tests/test_seed.py`
- `results/sanity/{seed_info.json,env_info.json,sample.json}`
- `docs/logs/day1_sanity.log`

### Key Results
- Seed 설정: `seed=123`, `deterministic=true`
- Environment snapshot 생성 완료: Python/Torch/Git 상태 json 저장
- Sanity log 기준 실행 완료 시각: `2026-03-03 13:14:42`

### Blockers / Notes
- 현재 Codex sandbox 세션에서 OpenMP SHM 이슈로 `run_sanity.sh` 재실행이 실패할 수 있음.
- Day1 산출물 파일은 이미 존재하며, Day2 진행에는 직접적인 blocker 아님.

### Next (Day 2 시작 입력용)
- 목표: `RDKit -> Morgan FP(2048) -> Pair Feature -> DataLoader [B, D]` 파이프라인 구축
- 우선 작업:
  - `src/sbml_msc_2026/datasets/` 데이터셋 로더 정리
  - `src/sbml_msc_2026/features/`에 pair feature 모듈 추가
  - `data/raw/` 입력 스키마 고정 + `data/processed/` 캐시(`.npz`/`.parquet`) 저장
- 검증:
  - tiny dataset으로 shape sanity (`[B, D]`)
  - split leakage 체크 포인트 명시

---

## SESSION SNAPSHOT (copy-paste)
Date: 2026-03-03
Day: 1
Status: 완료
Artifacts:
- configs/base.yaml
- scripts/run_sanity.sh
- scripts/sanity_python.py
- src/sbml_msc_2026/utils/{config.py,seed.py,logger.py,env.py}
- tests/test_seed.py
- results/sanity/{seed_info.json,env_info.json,sample.json}
- docs/logs/day1_sanity.log
Results:
- sanity artifacts 생성 완료
- seed reproducibility test 추가
Blockers:
- Codex sandbox에서 OpenMP SHM 재실행 이슈 가능
Next:
- Day2: RDKit fingerprint + pair feature + DataLoader shape sanity
