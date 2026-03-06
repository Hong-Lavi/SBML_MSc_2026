# Day3 Roadmap — DDI MLP Training/Eval Pipeline

> 이 문서는 Day3에서 접근할 파일의 순서와 각 파일에서 수행할 작업을 정리한 것이다.
> 아래 순서대로 진행하면 **의존성 충돌 없이** bottom-up으로 파이프라인이 완성된다.

---

## Gap 분석: DoD 대비 보완 사항

Day3 DoD (SBML_12day_program.md) + ProjectPrompt.md 요구사항을 분석한 결과,
원본 DoD에 **3가지 보완**을 적용했다:

| # | 보완 사항 | 근거 | 파일 |
|---|---|---|---|
| 1 | **Val split 추가** | Day2 config에 `val_ratio: 0.0`으로 예고됨. Early stopping은 val set 기준이어야 한다. Test set으로 모델 선택 = data snooping → 논문 리뷰에서 reject 사유. | `train_ddi.py` TODO S-2 |
| 2 | **Epoch-level training log** | epoch별 train_loss / val_loss / val_macro_f1을 JSON으로 저장. 논문 Figure (learning curve) 작성에 필수. 또한 학습 불안정 디버깅에도 사용. | `ddi_trainer.py` TODO T-7 |
| 3 | **compute_class_weights 직접 구현** | sklearn 내장 함수 대신 w_c = N/(C·N_c) 수식을 직접 구현. 내부 동작을 정확히 이해하는 것이 실무 역량의 핵심. | `metrics.py` TODO E-6 |

---

## 파일 접근 순서 (총 8 파일)

### 순서 1: `configs/day3_ddi.yaml` — 실험 설정 파악
```
할 일: 읽고 이해만 하면 된다. TODO 없음.
핵심: model/training/evaluation 섹션이 Day2에서 추가된 부분.
      val_fraction=0.2, patience=10, class_weight=auto 등
      이후 모든 코드에서 cfg["key"]로 참조하는 원천.
시간: ~5min (읽기 전용)
```

### 순서 2: `src/.../models/mlp.py` — MLP 모델 정의 (TODO 6개)
```
할 일: nn.Sequential로 hidden layers 쌓기 + Xavier init + forward.
핵심: (B, input_dim) → [Linear→BN→ReLU→Dropout]×L → Linear → (B, num_classes)
      softmax를 넣지 않는 이유: CrossEntropyLoss가 내부 처리.
함정: BN이 eval 모드에서 running stats를 쓰므로 model.eval() 필수.
시간: ~30min
```

### 순서 3: `tests/test_day3_model.py` — 모델 즉시 테스트 (TODO 4개)
```
할 일: forward shape, param count, gradient flow, eval determinism 검증.
핵심: 모델을 Trainer에 넣기 전에 단독으로 검증해야 한다.
      shape 불일치는 여기서 잡지 않으면 Trainer에서 에러 메시지가 모호해진다.
시간: ~15min
```

### 순서 4: `src/.../trainers/metrics.py` — 메트릭 + class weight (TODO 6개)
```
할 일: compute_metrics() + compute_class_weights() 구현.
핵심: macro-F1 = (1/C) Σ F1_c → minority class 성능 반영.
      class weight: w_c = N/(C·N_c) → 소수 class loss 기여도 증가.
      confusion matrix labels 인자 필수 (누락 class 방지).
함정: zero_division=0 안 넣으면 UndefinedMetricWarning.
      bincount에 음수 labels 넣으면 ValueError.
시간: ~20min
```

### 순서 5: `src/.../trainers/ddi_trainer.py` — 학습 루프 + early stopping (TODO 7개)
```
할 일: train_one_epoch() + evaluate() + fit() 구현.
핵심: gradient descent loop (zero_grad → forward → loss → backward → step).
      early stopping: val macro_f1 기준, patience epoch 개선 없으면 중단.
      checkpoint: model.state_dict()만 저장 (전체 model pickle은 anti-pattern).
함정: model.train()/model.eval() 전환 빼먹으면 dropout/BN 동작 이상.
      loss.item() * feat.size(0)로 가중 누적 안 하면 마지막 batch 편향.
시간: ~60min (Day3 최대 난이도)
```

### 순서 6: `scripts/train_ddi.py` — Entry point 조립 (TODO 6개)
```
할 일: config→seed→data split→model→loss→fit→test eval→save 전체 연결.
핵심: val split을 train에서 떼온다 (test에서 떼오면 data snooping).
      input_dim은 하드코딩하지 않고 데이터에서 동적 추론.
      best checkpoint 로드 후 test 평가 → metrics.json 저장.
함정: weight tensor device 불일치 → RuntimeError.
      fit()에 test_loader 넘기지 않는다 (학습 중 test 성능 참조 금지).
시간: ~30min
```

### 순서 7: `tests/test_day3_trainer.py` — Sanity overfit + metrics 테스트 (TODO 5개)
```
할 일: (1) 1 batch 반복 학습 → loss→0 확인 (sanity overfit)
       (2) compute_metrics 반환 형식 검증
       (3) compute_class_weights 수식 검증
핵심: sanity overfit은 "학습 능력 존재 확인"이다.
      실패하면 forward/loss/gradient 중 하나에 버그가 있는 것.
      dropout=0.0으로 해야 완벽한 수렴이 가능.
시간: ~25min
```

### 순서 8: `scripts/run_day3.sh` — 전체 실행 (TODO 없음)
```
할 일: bash scripts/run_day3.sh 실행.
핵심: (1) pytest 전체 통과 → (2) 학습 실행 → (3) 산출물 존재 확인.
      PYTHONPATH 자동 설정 포함.
시간: ~5min (실행 + 결과 확인)
```

---

## 시간 예상 총합

| 파일 | 예상 시간 |
|---|---|
| config 읽기 | 5min |
| mlp.py | 30min |
| test_model.py | 15min |
| metrics.py | 20min |
| ddi_trainer.py | 60min |
| train_ddi.py | 30min |
| test_trainer.py | 25min |
| run_day3.sh 실행 | 5min |
| **합계** | **~3h 10min** |

---

## DoD 체크리스트 (완료 기준)

- [ ] `bash scripts/run_day3.sh` 가 에러 없이 종료
- [ ] `results/day3/checkpoints/best_model.pt` 존재
- [ ] `results/day3/metrics.json` 존재 (test macro_f1, confusion_matrix 포함)
- [ ] `results/day3/checkpoints/training_log.json` 존재 (epoch별 기록)
- [ ] pytest 전체 통과 (model 4개 + trainer 5개 = 9개 테스트)
- [ ] PROJECT_SNAPSHOT.md Day3 섹션 업데이트
