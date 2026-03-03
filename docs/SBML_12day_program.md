# SBML_PreMaster — 12-Day Practical Onboarding Program (SBML_MSc_2026_Onboarding)

> **Design goals (핵심 목표)**
> 1) SBML 스타일의 연구 파이프라인을 “돌아가는 코드(재현 가능)”로 만든다.  
> 2) 논문 Methods를 구현 체크리스트로 바꾸는 능력을 만든다.  
> 3) 12일 종료 시, 교수님/선배에게 보여줄 수 있는 **포트폴리오(코드+문서+실험로그)**를 남긴다.

---

## 운영 규칙 (Automation)

- 라비 입력은 최소화:
  - 아침: `Day k 시작`
  - 저녁: `Day k 완료` 또는 `Day k 막힘` + `PROJECT_SNAPSHOT.md` 붙여넣기
- Assistant(나)는 Snapshot을 읽고 **다음 Day 패키지**를 자동 생성:
  - 수업(필수 개념) + 실습(체크리스트) + 스켈레톤 코드(TODO) + DoD(최소 산출물)

---

## 공통 레포 구조 (권장)

```text
SBML_PreMasters_Sprint/
  README.md
  pyproject.toml (또는 requirements.txt)
  environment.yml (conda)
  configs/
  data/                # raw/processed 분리 권장
  src/
    datasets/
    features/
    models/
    trainers/
    utils/
  scripts/             # train/infer/eval entrypoints
  tests/               # 최소 sanity test
  results/
  docs/
    PROJECT_SNAPSHOT.md
    logs/
```

---

## Day-by-day Plan (12 Days)

> 각 Day의 목표는 “학습”이 아니라 **산출물(artifact)** 기준으로 정의됨.

### Day 1 — Research Engineering & Linux/CLI + Git + Reproducibility
**Why**: SBML 연구 코드/오픈소스를 가져와서 돌리려면 “환경/재현성/CLI”가 먼저다.  
**Deliverables (DoD)**  
- 레포 구조 생성 + `environment.yml`/`requirements.txt`  
- `scripts/run_sanity.sh` (환경 점검용)  
- seed 고정 유틸 + minimal test 1개  
**Key topics**: bash basics, git branch strategy, conda, reproducibility(Seed/Config/Logging)

---

### Day 2 — DeepDDI-min (1): RDKit → Fingerprint → Pair Feature + Caching
**Why**: 약물 구조 기반 파이프라인의 핵심은 feature engineering과 데이터 I/O다.  
**Deliverables (DoD)**  
- SMILES → Morgan FP(2048) 함수 + 캐시(npz/parquet)  
- pair feature 생성 모듈(concat or symmetric)  
- Dataset/DataLoader가 `[B, D]`를 안정적으로 출력  
**Key topics**: RDKit, fingerprint, caching, split & leakage pitfalls

---

### Day 3 — DeepDDI-min (2): MLP Training/Eval + Research-grade Metrics
**Deliverables (DoD)**  
- `train_ddi.py` 실행으로 checkpoint + metrics 저장  
- macro-F1 + confusion matrix + per-class report  
- early stopping + reproducible split  
**Key topics**: class imbalance(Weighted CE vs Sampler), macro-F1, sanity overfit test

---

### Day 4 — DeepDDI Robustness + Screening Packaging (Paxlovid-style scenario)
**Deliverables (DoD)**  
- (A,B)/(B,A) 처리 전략 확정(augmentation or symmetric features)  
- `infer_pairs.py` 형태의 스크리닝 스크립트  
- 결과 리포트(csv) + top-k 분석  
**Key topics**: symmetry, calibration intuition, reporting

---

### Day 5 — DeepEC-min (1): Protein Sequence Encoding + 1D-CNN Baseline
**Deliverables (DoD)**  
- sequence tokenizer/encoder(one-hot or embedding)  
- 1D-CNN 모델 forward가 정상 동작  
- train/eval skeleton 실행(작은 subset도 OK)  
**Key topics**: padding/masking, Conv1d shapes, variable length handling

---

### Day 6 — DeepEC-min (2): Hierarchical EC + (Optional) Homology Fallback “Design”
**Deliverables (DoD)**  
- EC level1(혹은 level2까지) 평가 파이프라인  
- fallback은 완전 구현 대신 “설계 문서 + 인터페이스(skeleton)”  
**Key topics**: hierarchical labels, error analysis, fallback interface design

---

### Day 7 — GEM/COBRApy Fundamentals: FBA/FVA/KO
**Deliverables (DoD)**  
- COBRApy로 FBA 1회 실행 + 결과 저장  
- single-gene knockout simulation loop + 요약 테이블  
- FVA 1회 실행  
**Key topics**: S·v=0, constraints, LP, objective choice

---

### Day 8 — MGP Workflow: Repo Reading + Minimal Execution (Environment-aware)
**Deliverables (DoD)**  
- MGP_prediction(또는 유사) repo의 I/O 다이어그램 1장  
- “최소 실행(minimal run)” 성공 또는 실패 원인 문서화  
- macOS 한계 시 Colab/Ubuntu/Docker로 이동 계획 확정  
**Key topics**: environment parity, dependency pinning, reproducible runs

---

### Day 9 — PNAS 2025-style Mini: KO Simulation → Feature → UMAP/k-means
**Deliverables (DoD)**  
- KO 결과를 feature matrix로 정리(`[n_KO, n_features]`)  
- UMAP + kmeans로 cluster 생성 + 시각화 1장  
- “왜 이게 sensitization target 탐색과 연결되는지” 1-page 정리  
**Key topics**: clustering, representation, biological interpretation

---

### Day 10 — Biology-informed ML Integration: GEM features + ML baseline
**Deliverables (DoD)**  
- GEM-derived features + omics-derived features를 합친 baseline 모델  
- leakage 방지(환자/샘플 단위 split) 문서화  
**Key topics**: feature engineering vs end-to-end, evaluation protocol

---

### Day 11 — Portfolio Packaging Day: README + 1-page × 3 + Execution Commands
**Deliverables (DoD)**  
- 1-page 요약 3개(DDI / DeepEC / GEM-KO)  
- 실행 커맨드(재현 방법) + 결과 폴더 정리  
- “교수님 미팅 질문 5개” 준비  
**Key topics**: scientific communication, reproducibility checklist

---

### Day 12 — Buffer/Polish: Debugging, Refactor, Make it Portable
**Deliverables (DoD)**  
- 다른 환경에서도 실행되도록 의존성/경로/seed 고정  
- 코드 품질(모듈화/테스트/로그) 개선  
- 최종 snapshot + 다음 4주 계획 초안  
**Key topics**: maintainability, refactoring discipline

---

## 매일 공통 “Definition of Done” 체크
- (1) 코드가 실제로 실행됨 (entrypoint 존재)  
- (2) 결과물이 파일로 저장됨 (checkpoint/metrics/plots)  
- (3) 재현 방법이 README 또는 scripts에 남아 있음  
- (4) `PROJECT_SNAPSHOT.md` 업데이트 완료
