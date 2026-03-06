# Skeleton Code Guidelines — SBML_MSc_2026

> 이 문서는 Claude가 Day별 모의 프로젝트의 skeleton code를 생성할 때 따라야 할 규칙이다.
> 목적: 라비가 "복붙"이 아닌 "이해 후 구현"을 하도록 유도하는 것.

---

## 1. TODO 힌트의 추상화 수준

### 원칙
- **함수명/API는 알려줘도 된다.** 어떤 도구를 써야 하는지는 알아야 하니까.
- **인자값, 파일명, 컬럼명 등 구체적 리터럴은 주지 마라.** 스스로 코드/데이터를 읽고 채워야 한다.

### Good vs Bad 예시

```python
# ❌ BAD — 답 그대로 복붙 가능
# TODO: df = pd.read_csv("data/raw/dummy_ddi_pairs.csv")
# TODO: fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

# ✅ GOOD — 방향만 제시
# TODO: pd.read_csv()로 CSV를 로드하라. 경로는 config에서 가져올 것.
# TODO: AllChem.GetMorganFingerprintAsBitVect()를 사용하라.
#       radius와 nBits는 함수 인자로 전달받은 값을 쓸 것.
```

```python
# ❌ BAD — 변수명+조건까지 완성
# TODO: if indices is not None:
#           self.df = self.df.iloc[indices].reset_index(drop=True)

# ✅ GOOD — 개념 + 메서드 힌트
# TODO: indices가 주어지면 df를 필터링하라.
#       힌트: DataFrame.iloc[] + reset_index() 활용.
#       왜 reset_index가 필요한지 생각해볼 것.
```

---

## 2. 힌트 구성 3단계 (Layered Hints)

각 TODO 블록은 아래 순서로 작성한다:

```python
# TODO X-Y: [한 줄 목표] ← 무엇을 해야 하는지
#   힌트: [사용할 함수/클래스명]  ← 어떤 도구를 쓸지
#   생각: [왜 이렇게 하는지 / edge case]  ← 왜 이 방식인지
```

**금지 사항:**
- 파일 경로 리터럴 (예: `"data/raw/xxx.csv"`)
- 컬럼명 리터럴 (예: `"drug_a_smiles"`)
- 완성된 조건문/반복문 구조
- 정답 shape 값 (예: `assert shape == (2048,)` — 대신 "n_bits와 일치하는지 검증하라")

---

## 3. Shape 주석은 변수 선언부에 유지

shape 주석은 학습에 핵심이므로 유지하되, **구체적 숫자 대신 변수명으로** 표기한다.

```python
# ✅ GOOD
# fp: ndarray, shape=(n_bits,), dtype=uint8
# pair_feat: shape=(2*n_bits,) if concat, (n_bits,) if sum/diff
# batch: (B, D) where D depends on pair_method

# ❌ BAD
# fp: shape=(2048,)  ← 구체적 숫자 노출
```

---

## 4. 테스트 TODO도 동일 원칙 적용

```python
# ❌ BAD
# assert fp.shape == (2048,)
# assert result is None

# ✅ GOOD
# TODO T-X: FP shape이 config의 n_bits와 일치하는지 검증하라.
# TODO T-Y: 존재하지 않는 경로 입력 시 반환값을 검증하라.
```

---

## 5. Config 값은 skeleton에 포함하되, 코드에서는 참조만

config YAML은 완성된 형태로 제공해도 된다 (실험 설정은 "외워야 할 것"이 아니라 "참조할 것"이므로).
단, 코드에서 config 값을 직접 하드코딩하지 않고 `cfg["key"]`로 접근하도록 TODO를 구성한다.

```python
# ✅ GOOD
# TODO: config에서 fingerprint 관련 설정을 읽어 변수에 할당하라.
#       힌트: cfg["fingerprint"]["radius"] 등

# ❌ BAD
# radius = cfg["fingerprint"]["radius"]  # 2
# n_bits = cfg["fingerprint"]["n_bits"]  # 2048
```

---

## 6. Skeleton 제공 시 포함해야 할 메타 정보

모든 skeleton 파일 상단에 아래를 포함한다:

```python
"""
Day X: [모듈 목적 한 줄]

Pipeline 위치: [이전 모듈] → [이 모듈] → [다음 모듈]
Input:  [입력 타입/shape 개요]
Output: [출력 타입/shape 개요]

TODO 개수: N개 (TODO X-1 ~ X-N)
난이도: ★☆☆ / ★★☆ / ★★★
예상 소요: ~30min
"""
```

---

## 7. 요약: 제공 O / 제공 X 체크리스트

| 항목 | 제공 | 비고 |
|---|---|---|
| 사용할 함수/클래스명 | O | `pd.read_csv()`, `torch.stack()` 등 |
| 함수 시그니처 (빈 def) | O | 인자명 + type hint |
| shape 주석 (변수명 기반) | O | `(n_bits,)`, `(B, D)` |
| import 문 | O | 어떤 라이브러리를 쓰는지는 알아야 함 |
| config YAML 전체 | O | 실험 설정은 참조 대상 |
| 파일 경로 리터럴 | X | config에서 읽도록 유도 |
| 컬럼명 리터럴 | X | 데이터를 직접 확인하도록 유도 |
| 완성된 조건문/로직 | X | 힌트로만 방향 제시 |
| 구체적 shape 숫자 | X | 변수명으로 대체 |
| 정답 assert 값 | X | 검증 로직은 직접 구성 |
