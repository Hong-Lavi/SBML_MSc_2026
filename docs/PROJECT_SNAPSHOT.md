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

## Day 2 — DeepDDI-min (1): FP → Pair Feature → DataLoader (완료)

### Date
- 2026-03-06 (KST)

### Status
- 완료 (pipeline PASSED, 19/19 tests PASSED)

### Day2 DoD 체크
- [x] SMILES → Morgan FP (radius=2, nBits=2048) 함수 + `.npz` 캐시
- [x] Pair feature 모듈 (concat/sum/diff/symmetric)
- [x] Drug-aware split (train drug set ∩ test drug set = ∅)
- [x] Dataset/DataLoader → `[B, D]` shape sanity 통과
- [x] 19/19 pytest 통과

### Pipeline 흐름 (Day3 참조용)

```
[Raw CSV]  data/raw/dummy_ddi_pairs.csv
  │        columns: drug_a_smiles, drug_b_smiles, interaction_type (0/1/2)
  │        16 pairs, 6 unique drugs
  ▼
[Step 1]  SMILES → Morgan FP
  │        smiles_to_morgan_fp(smi, radius=2, nBits=2048) → ndarray (2048,) uint8 ∈{0,1}
  │        compute_fp_dict(smiles_list) → Dict[str, ndarray]  {6 drugs}
  │        캐시: np.savez_compressed → data/processed/fingerprints/fp_cache.npz
  ▼
[Step 2]  Drug-aware Split
  │        6 drugs → shuffle(seed) → 4 train_drugs / 2 test_drugs
  │        pair 할당: A∈train AND B∈train → train (7), otherwise → test (9)
  │        ※ random pair split 대비 data leakage 방지
  ▼
[Step 3]  Pair Feature 생성 (on-the-fly in __getitem__)
  │        fp_a (2048,) + fp_b (2048,) → pair_feat
  │        concat: np.concatenate → (4096,)  ← 이번 프로젝트 기본값
  │        sum: fp_a + fp_b → (2048,)
  │        diff: |fp_a - fp_b| → (2048,)
  │        symmetric: sorted concat → (4096,)
  ▼
[Step 4]  DDIPairDataset(Dataset) → DataLoader
  │        __getitem__(idx) → (FloatTensor (4096,), LongTensor scalar)
  │        DataLoader(batch_size=4) → torch.stack
  │        batch_feat: [B, 4096]   ← Day3 MLP 입력
  │        batch_label: [B]        ← CrossEntropyLoss 타겟 (0/1/2)
  ▼
[Output]  results/day2/day2_summary.json
```

### 핵심 설계 결정 & 근거

| 결정 | 근거 |
|---|---|
| Morgan FP (ECFP4) | 반경 2 bond 내 원형 부분구조 열거 → 분자 유사도의 de facto standard |
| nBits=2048 | 충돌률과 메모리 tradeoff; 연구용 기본값 |
| `.npz` 캐시 | `allow_pickle=False`로 보안 유지, `savez_compressed`로 디스크 절약 |
| Drug-aware split | Random pair split은 같은 drug이 train/test에 등장 → FP 패턴 암기 → 과대 성능 |
| On-the-fly pair feature | 전체 pair matrix 미리 계산 시 O(N²×D) 메모리 → 대규모 확장 불가 |

### Artifacts (경로)

- `configs/day2_ddi.yaml`
- `data/raw/dummy_ddi_pairs.csv`
- `data/processed/fingerprints/fp_cache.npz`
- `src/sbml_msc_2026/features/{fingerprint.py, pair_features.py}`
- `src/sbml_msc_2026/datasets/{ddi_dataset.py, splitter.py}`
- `scripts/{run_day2_pipeline.py, run_day2.sh}`
- `tests/{test_day2_fingerprint.py, test_day2_dataset.py}`
- `results/day2/day2_summary.json`

### 디버깅 이력 (학습 포인트)

| 에러 | 원인 | 수정 | 교훈 |
|---|---|---|---|
| `ModuleNotFoundError: sbml_msc_2026` | PYTHONPATH 미설정 | `export PYTHONPATH="$(pwd)/src:$PYTHONPATH"` | src-layout은 반드시 PYTHONPATH 또는 `pip install -e .` 필요 |
| `KeyError: 'label'` | CSV 컬럼명 = `interaction_type` | `row["interaction_type"]`으로 변경 | 실제 데이터 스키마를 반드시 확인 후 코딩 |
| `AssertionError: dtype == 'torch.int64'` | dtype 객체를 문자열과 비교 | 따옴표 제거 → `torch.int64` | dtype 비교는 항상 객체끼리 (`torch.int64`, `torch.float32`) |
| `load_fp_cache` → `None` 연쇄 버그 | `FileNotFoundError` 잡는 방식 변경 후 호출자 미수정 | try/except → if None 패턴 전환 | 하위 함수 에러 처리 변경 시 상위 호출자 반드시 점검 |

### Blockers / Notes
- `fingerprint.py`의 `get_or_compute_fp_dict`에서 `logging.info` → `logger.info`로 변경 권장 (기능 영향 없음, linting 일관성)
- dummy data 16행 / 6 drugs로 충분히 작아서 성능 평가 무의미, Day3에서 더 큰 데이터 활용 필요

### Next (Day 3 시작 입력용)
- 목표: MLP 학습/평가 + Research-grade Metrics
- Day2 산출물 중 Day3가 직접 사용하는 것:
  - `DDIPairDataset` + `build_ddi_dataloader()` → train/test DataLoader
  - `batch_feat [B, 4096]` → MLP 입력, `batch_label [B]` → CE Loss 타겟
  - `split_ddi_pairs()` → drug-aware train/test indices
  - `configs/day2_ddi.yaml` 구조를 Day3 config에서 확장
- 우선 작업:
  - `src/sbml_msc_2026/models/` MLP 정의 (input_dim=4096, hidden, num_classes=3)
  - `src/sbml_msc_2026/trainers/` 학습 루프 + early stopping
  - macro-F1 + confusion matrix + per-class classification report
  - sanity overfit test (1 batch로 loss→0 확인)

---

## SESSION SNAPSHOT (copy-paste)
Date: 2026-03-06
Day: 2
Status: 완료
Artifacts:
- configs/day2_ddi.yaml
- data/raw/dummy_ddi_pairs.csv
- data/processed/fingerprints/fp_cache.npz
- src/sbml_msc_2026/features/{fingerprint.py, pair_features.py}
- src/sbml_msc_2026/datasets/{ddi_dataset.py, splitter.py}
- scripts/{run_day2_pipeline.py, run_day2.sh}
- tests/{test_day2_fingerprint.py, test_day2_dataset.py}
- results/day2/day2_summary.json
Results:
- Pipeline sanity check PASSED (train [4,4096], test [4,4096])
- 19/19 pytest PASSED
- Drug-aware split: 4 train drugs / 2 test drugs → 7 train pairs / 9 test pairs
Blockers:
- logger.info vs logging.info 일관성 (minor)
Next:
- Day3: MLP 학습 + macro-F1 + confusion matrix + early stopping
