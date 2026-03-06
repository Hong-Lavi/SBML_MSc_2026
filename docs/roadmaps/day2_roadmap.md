# Day2 Roadmap — DDI Fingerprint → Pair Feature → DataLoader Pipeline

> 이 문서는 Day2에서 접근한 파일의 순서와 각 파일에서 수행한 작업을 정리한 것이다.
> 아래 순서대로 진행하면 **의존성 충돌 없이** bottom-up으로 파이프라인이 완성된다.

---

## Gap 분석: DoD 대비 보완 사항

Day2 DoD (SBML_12day_program.md) + ProjectPrompt.md 요구사항을 분석한 결과,
원본 DoD에 **3가지 보완**을 적용했다:

| # | 보완 사항 | 근거 | 파일 |
|---|---|---|---|
| 1 | **Drug-aware split 모듈 분리** | Random pair split은 동일 drug이 train/test에 등장 → FP 패턴 암기 → data leakage. Drug 단위 분리를 별도 모듈로 구현하여 Day3 이후 재사용 가능하게 함. | `splitter.py` TODO 8-1~8-4 |
| 2 | **FP 캐싱 (`.npz`)** | RDKit FP 계산은 molecule 수가 늘면 O(N) 비용. 한 번 계산 후 `np.savez_compressed`로 캐싱하면 재실행 시 I/O만으로 복원. `allow_pickle=False`로 보안 유지. | `fingerprint.py` TODO 3-1~4-1 |
| 3 | **4가지 pair feature method 지원** | concat만으로도 충분하지만, sum/diff/symmetric을 함께 구현하면 Day4 이후 ablation study에서 method별 성능 비교가 즉시 가능. | `pair_features.py` TODO 5-1~5-4 |

---

## 파일 접근 순서 (총 8 파일)

### 순서 1: `configs/day2_ddi.yaml` — 실험 설정 파악
```
할 일: 읽고 이해만 하면 된다. TODO 없음.
핵심: seed, paths, fingerprint(radius/n_bits), pair_feature(method),
      split(method/train_ratio), dataloader(batch_size) 섹션 구성.
      이후 모든 코드에서 cfg["key"]로 참조하는 원천.
시간: ~5min (읽기 전용)
```

### 순서 2: `src/.../features/fingerprint.py` — Morgan FP 계산 + 캐시 (TODO 8개)
```
할 일: SMILES→Mol→Morgan FP 변환, dict 일괄 계산, npz 저장/로드, 통합 함수.
핵심: smiles_to_morgan_fp() → ndarray (n_bits,) uint8 ∈{0,1}
      compute_fp_dict() → Dict[str, ndarray]
      save/load_fp_cache() → npz 직렬화/역직렬화
      get_or_compute_fp_dict() → 캐시 hit이면 로드, miss면 계산+저장+반환
함정: invalid SMILES → MolFromSmiles returns None → 반드시 None 체크.
      load 시 파일 미존재 → None 반환 (FileNotFoundError 잡기).
      load_fp_cache 에러 처리 변경 시 호출자(get_or_compute)도 동기화 필수.
시간: ~40min
```

### 순서 3: `src/.../features/pair_features.py` — Pair feature 생성 (TODO 4개)
```
할 일: fp_a + fp_b → pair vector 생성. 4가지 method 구현.
핵심: concat: np.concatenate → (2*n_bits,)  순서 구분 O
      sum: fp_a + fp_b → (n_bits,)          순서 무관 (대칭)
      diff: |fp_a - fp_b| → (n_bits,)       순서 무관 (대칭)
      symmetric: sorted concat → (2*n_bits,) 순서 무관 + 정보 보존
함정: input shape 불일치 → 명시적 assert로 조기 검출.
      diff에서 abs() vs np.abs() → 결과 동일하나 np.abs()가 명시적.
시간: ~15min
```

### 순서 4: `tests/test_day2_fingerprint.py` — FP + pair feature 즉시 테스트 (TODO 11개)
```
할 일: FP shape/값 검증, 캐시 roundtrip, pair feature shape/대칭성 테스트.
핵심: 모듈을 Dataset에 넣기 전에 단독으로 검증해야 한다.
      shape 불일치는 여기서 잡지 않으면 DataLoader에서 에러 메시지가 모호해진다.
함정: 캐시 테스트는 tmp_path fixture 사용 → 테스트 간 상태 격리.
      존재하지 않는 경로 → None 반환 검증 필수.
시간: ~20min
```

### 순서 5: `src/.../datasets/splitter.py` — Train/Test 분리 (TODO 4개)
```
할 일: random split + drug-aware split 구현.
핵심: random: 행 인덱스 shuffle → train_ratio 기준 분할.
      drug-aware: unique drug 추출 → drug shuffle → drug split →
                  pair 할당 (A∈train_drugs AND B∈train_drugs → train).
      반환값: {"train": List[int], "test": List[int]} (행 인덱스)
함정: seed 미고정 시 split 비재현 → np.random.default_rng(seed) 사용.
      drug-aware split은 pair 수가 불균형해질 수 있음 (정상 동작).
시간: ~25min
```

### 순서 6: `src/.../datasets/ddi_dataset.py` — Dataset + DataLoader (TODO 4개)
```
할 일: DDIPairDataset(Dataset) 구현 + build_ddi_dataloader() wrapper.
핵심: __init__: CSV 로드 → indices 필터링 → fp_dict 준비
      __getitem__(idx): df.iloc[idx] → fp_a, fp_b 조회 → pair_feat 생성
                        → (FloatTensor (D,), LongTensor scalar)
      DataLoader(batch_size=B): torch.stack → (B, D), (B,)
함정: indices 필터링 후 reset_index(drop=True) 빼먹으면 idx 불일치.
      interaction_type 컬럼명을 데이터에서 확인 후 사용할 것.
시간: ~30min
```

### 순서 7: `tests/test_day2_dataset.py` — Dataset/Splitter 테스트 (TODO 8개)
```
할 일: Dataset length, getitem shape, DataLoader batch shape,
       splitter coverage/overlap/reproducibility 검증.
핵심: dummy CSV fixture로 독립 테스트 환경 구성.
      split indices로 생성한 Dataset의 length가 올바른지 검증.
시간: ~15min
```

### 순서 8: `scripts/run_day2.sh` + `scripts/run_day2_pipeline.py` — 전체 실행 (TODO 없음)
```
할 일: bash scripts/run_day2.sh 실행.
핵심: (1) PYTHONPATH 자동 설정
      (2) run_day2_pipeline.py 실행 → shape sanity check
      (3) pytest 전체 실행
      (4) 결과 요약 JSON 저장
시간: ~5min (실행 + 결과 확인)
```

---

## 시간 예상 총합

| 파일 | 예상 시간 |
|---|---|
| config 읽기 | 5min |
| fingerprint.py | 40min |
| pair_features.py | 15min |
| test_fingerprint.py | 20min |
| splitter.py | 25min |
| ddi_dataset.py | 30min |
| test_dataset.py | 15min |
| run_day2.sh 실행 | 5min |
| **합계** | **~2h 35min** |

---

## 디버깅 이력 (실제 발생한 에러)

| 에러 | 원인 | 수정 | 교훈 |
|---|---|---|---|
| `ModuleNotFoundError: sbml_msc_2026` | PYTHONPATH 미설정 | `export PYTHONPATH="$(pwd)/src:$PYTHONPATH"` | src-layout은 반드시 PYTHONPATH 또는 `pip install -e .` 필요 |
| `KeyError: 'label'` | CSV 컬럼명 = `interaction_type` | `row["interaction_type"]`으로 변경 | 실제 데이터 스키마를 반드시 확인 후 코딩 |
| `AssertionError: dtype == 'torch.int64'` | dtype 객체를 문자열과 비교 | 따옴표 제거 → `torch.int64` | dtype 비교는 항상 객체끼리 |
| `load_fp_cache` → `None` 연쇄 버그 | 하위 함수 에러 처리 변경 후 호출자 미수정 | try/except → if None 패턴 전환 | 하위 함수 변경 시 상위 호출자 반드시 점검 |

---

## DoD 체크리스트 (완료 기준)

- [x] SMILES → Morgan FP (radius=2, nBits=2048) 함수 + `.npz` 캐시
- [x] Pair feature 모듈 (concat/sum/diff/symmetric)
- [x] Drug-aware split (train drug set ∩ test drug set = ∅)
- [x] Dataset/DataLoader → `[B, D]` shape sanity 통과
- [x] `bash scripts/run_day2.sh` 에러 없이 종료
- [x] `data/processed/fingerprints/fp_cache.npz` 존재
- [x] `results/day2/day2_summary.json` 존재
- [x] 19/19 pytest 통과
- [x] PROJECT_SNAPSHOT.md Day2 섹션 업데이트
