# Reasoning LLM Early Stop

추론 모델이 답을 이미 굳힌 뒤에도 계속 생성하는 문제를 다룬다.
**학습된 선형 판독기(detector)** 로 "답이 굳었다"를 hidden state 에서 읽어 발화하고,
**V 캐시 섭동** 으로 그 발화를 검증한 뒤, 통과하면 멈추고 아니면 되돌려 생성을 이어 간다(가역 정지).

- 모델: Qwen3-8B (thinking mode, 그리디 생성, max_new 8192)
- 벤치마크: MATH-500
- 현재 상태와 수치: `docs/` 참고 (detector E1~E3 정리 등)

```
생성 중 문단 경계 ──▶ ① Fire     detector 점수 w·h ≥ τ ?   (아니면 계속 생성)
                       ② Close    캐시 포크 → "\n</think>\n\nFinal Answer: \boxed{" 주입
                       ③ Read     답 읽기 (발화 1회당 약 15.8 토큰)
                       ④ Verify   V 캐시 섭동으로 답이 굳었는지 확인   ← 아직 오프라인 단계
                       ⑤ Pass → 멈추고 답 출력 / Fail → 롤백(cache.crop) 후 계속 생성
```

---

## 폴더 구조

```
.
├── README.md
├── environment.yml        # conda 환경 (아래 "환경" 참고)
├── src/                   # 실행 코드 — 전부 이 폴더 안에서 서로 import 한다
│   ├── run_commit_math.py   ① 생성 + probe + hidden state 덤프 수집
│   ├── grade2.py            채점기 (eq_math2). 다른 스크립트가 import
│   ├── window_fit.py        진단: ±K 창에 들어가는 예제 수 (덤프 안 읽음)
│   ├── label_shift.py       진단: 구/신 데이터의 commit_step 비교
│   ├── probe_commit.py      ② detector 신호 검정 (판독기·특징 비교, E1~E3)
│   ├── quartile_paired.py   ③ 배포 평가 (길이 사분위 안 fixed_T 짝비교)
│   └── vnoise_pretest.py    ④ V 캐시 노이즈 섭동 검증 (GPU)
├── scripts/               # 실행 명령 모음 (.sh)
├── docs/                  # 정리 문서, 발표 자료
├── results/               # 작은 결과물만 커밋 (로그, w/traj npz, json)
│   ├── collect/  detector/  deploy/  vnoise/
├── data/                  # ★ git 에 올리지 않음 — 서버 공유 경로 사용
└── legacy/                # ★ git 에 올리지 않음 — 닫힌 축의 코드·데이터
```

**`src/` 를 하위 폴더로 나누지 않는 이유**: 스크립트들이 `from grade2 import eq_math2` 처럼
같은 폴더의 파일을 import 한다. 파이썬은 실행한 스크립트의 폴더에서 import 를 찾으므로
`python src/xxx.py` 로 실행하면 코드 수정 없이 동작한다.

---

## 환경

```bash
conda env create -f environment.yml
conda activate cot-quant
```

주요 패키지: `torch`, `transformers`, `numpy`, `scipy`, `scikit-learn`, `sympy`, `safetensors`, `huggingface_hub`
(정확한 버전은 `environment.yml`)

모델 가중치는 HF 캐시(`~/.cache/huggingface/hub/models--Qwen--Qwen3-8B`)를 쓴다.

---

## 데이터

데이터는 git 에 없다. 서버의 공유 경로에서 `data/` 로 링크해서 쓴다.

```bash
ln -s <공유 경로>/data data        # TODO: 공유 경로 기입
```

| 파일 | 내용 | 크기 |
|---|---|---|
| `sig_para500.jsonl` | 500문제 생성 결과. 예제별 `text`, `pred`, `gold`, `fin`, `think_len`, `probes_arms`(A/B/C/D), `feat.commit_step` | |
| `sig_para500.jsonl.h.f16` | hidden state 덤프 `[104676, 6, 4096]` fp16 | 5.1 GB |
| `sig_para500.jsonl.meta.jsonl` | 덤프 행마다 `ex_id, t, para, grid` | |
| `sig_para500.jsonl.dumpinfo.json` | 덤프에 담긴 모델 층 `[9, 18, 27, 31, 35, 36]` | |
| `sig_para500.jsonl.prefill.f16` | | |
| `math500_baseline.jsonl` | 베이스라인 실행 | |

- 덤프와 메타 파일은 jsonl 경로 뒤에 확장자를 붙여서 찾는다. **같은 폴더에 같이 두면 된다.**
- 덤프 행 = probe 와 무관하게 32토큰마다 + 모든 문단 경계(`\n\n`)에서 저장한 hidden state. 행 간격 중앙 약 18토큰.
- probe(답 읽기)는 직전 probe 이후 200자 이상 쌓인 뒤의 첫 `\n\n` 에서, 경계가 없으면 1,200자에서 강제로 찍는다.

---

## 실행

모든 명령은 저장소 최상위에서 실행한다. 로그를 `tee` 로 남길 때는 **`python -u`** 를 붙인다
(안 붙이면 출력이 버퍼에 쌓여 멈춘 것처럼 보인다).

### ① 수집 (GPU)

```bash
# TODO: sig_para500 을 만든 실제 명령을 results/collect/para500.log 상단에서 확인해 그대로 기입
python -u src/run_commit_math.py --model Qwen/Qwen3-8B --n 500 \
  --probe-mode para --arms A,B,C,D \
  --dump-hidden --dump-layers 9 18 27 31 35 36 --sig-every 32 \
  --out data/sig_para500.jsonl | tee results/collect/para500.log
```

자체검증 T1~T7 이 먼저 돈다. 특히 T4 는 probe 를 켜고 끈 생성의 토큰열이 같은지 확인한다(probe 가 본 생성을 오염시키지 않음).

### 진단 (CPU, 수 초)

```bash
python src/window_fit.py data/sig_para500.jsonl        # ±K 창 선택
```

### ② detector 신호 검정 (CPU)

```bash
# 기본: ridge, 세 층, matched/uniform 귀무 둘 다
python -u src/probe_commit.py data/sig_para500.jsonl --layers 18 27 36 --K 12 \
  --nnull 100 --drop-extreme 10 | tee results/detector/base.log

# 센터링(예제 안 평균 제거) + 잘 규제된 판독기
python -u src/probe_commit.py data/sig_para500.jsonl --layers 36 --K 12 \
  --nnull 100 --drop-extreme 10 --null matched --center-ex \
  --pca 256 --model ridge logreg | tee results/detector/e1c_L36.log
```

주요 옵션

| 옵션 | 뜻 |
|---|---|
| `--K` | commit 앞뒤로 쓰는 **덤프 행** 수 (토큰 수 아님). 12 = 앞 12행 y=0, 뒤 12행 y=1 |
| `--nnull` | 가짜 commit 반복 수. 경험적 p 하한 = 1/(N+1). 3층 Bonferroni 면 N ≥ 60 필요 |
| `--null` | `matched`: 다른 예제의 commit 상대위치를 빌림(위치 분포 보존). `uniform`: 균등 |
| `--center-ex` | 예제 안 24행 평균을 빼서 예제 사이 차이 제거 |
| `--drop-extreme 10` | 노름이 중앙값의 10배를 넘는 행 제외 (massive activation, fp16 오버플로) |
| `--model` | `dom ridge logreg mlp rbf` 중 여러 개. 같은 귀무 추첨 위에서 비교 |
| `--pca` | 모든 판독기에 같은 PCA 차원 (mlp/rbf 는 0 이면 256) |
| `--feat` | `snap,delta,local,global` (과거만 쓰는 인과 특징) |
| `--traj`, `--save-w` | 홀드아웃 궤적 정렬 / 학습된 w 저장 |

결과 읽는 법: **진짜 AUC 가 아니라 Δ = 진짜 − 가짜평균** 으로 비교한다. 판독기가 위치만 읽어도 가짜 AUC 가 0.5 보다 높게 나온다.

### ③ 배포 평가 (CPU)

```bash
python -u src/quartile_paired.py data/sig_para500.jsonl --layer 36 --K 12 --folds 5 \
  --targets 0.3,0.5,0.7 --drop-extreme 10 --tol 0.05 --early-t 256 --seed 0 \
  | tee results/deploy/qp_L36.log

# detector 를 ridge + PCA256 으로
python -u src/quartile_paired.py data/sig_para500.jsonl --layer 36 --K 12 --folds 5 \
  --targets 0.3,0.5,0.7 --drop-extreme 10 --tol 0.05 --early-t 256 --seed 0 \
  --pca 256 | tee results/deploy/qp_pca256_L36.log
```

네 팔을 같은 절감률에서 예제 단위로 짝지어 비교한다: fixed_T / 오라클 길이×ρ / 예측 길이×ρ / +검증기(gold, 상한).
채점은 전부 `grade2.eq_math2`. 처음 실행은 채점이 오래 걸린다.

### ④ V 캐시 섭동 (GPU)

```bash
python -u src/vnoise_pretest.py --model Qwen/Qwen3-8B \
  --out data/sig_para500.jsonl --script src/run_commit_math.py \
  --data-file <MATH-500 파일> --data-n 500 \
  --n 200 --reps 5 --cs 0.25,0.5,1.0,2.0 --layers 18 all \
  --save results/vnoise/vnoise.json | tee results/vnoise/vnoise.log
```

- `--script` 는 기본값이 옛 경로(`out/run_commit_math.py`)라 **반드시 `src/run_commit_math.py` 로 지정**한다.
- `--data-file`, `--data-n` 은 수집 때와 같아야 한다(`--n` 이 다르면 셔플 순서가 달라져 `ex_id` 가 어긋난다). TODO: 파일 경로 기입
- `--dry-run` 으로 모델 없이 접두사 복원 가능 여부만 먼저 확인할 수 있다.
- 판정은 집계 비가 아니라 **예제 안 짝지은 차이차(DiD)** 로 한다: `[log post − log pre] − [log c1 − log c0]`.

---

## 알아둘 함정 (전부 실제로 당한 것)

- **Qwen3 종료 토큰**: `tok.eos_token_id`(`<|endoftext|>`)와 실제 종료(`<|im_end|>`)가 다르다. 하나만 보면 모든 예제가 max_new 까지 간다.
- **`hidden_states[-1]` 은 이미 `model.norm()` 뒤**다. 다시 norm 하면 가중치가 두 번 곱해진다.
- **`dump[:, L]` = `model.layers[L-1]` 의 출력.** 덤프 인덱스 36(모델 층 36)은 final norm 뒤라 V 가 없다.
- **`commit_step` 은 gold 가 아니라 모델 자신의 최종 답 기준.** gold 로 잡으면 오답 예제가 전부 빠져 표본이 쉬운 문제로 쏠린다.
- **h 에 공분산·PCA·k-means 를 걸기 전에 극단 노름 행을 뺄 것.** d2276 하나가 제곱노름의 99.1%.
- **`--resume` 과 `--n`**: `--n` 이 다르면 문제 순서가 달라진다. 다른 `--out` 이름을 쓸 것.
- **저장 텍스트를 재토큰화하면 길이가 `gen_len` 보다 1 작다**(`</think>` 가 빠짐). 전체를 한 번 토큰화한 뒤 앞 t 개를 쓸 것.
- **sdpa 와 eager 는 수백 스텝 안에 토큰이 갈라진다.** probe 와 신호는 같은 실행에서 뽑을 것.

---

## 협업 규칙

- `data/`, `legacy/`, 덤프 파일(`*.f16`, `*.f32`)은 **절대 커밋하지 않는다.** 한 번 들어가면 지워도 이력에 남는다. 커밋 전에 `git status` 로 확인.
- 결과 로그는 `results/<단계>/` 에 남기고, 어떤 명령으로 만들었는지 `scripts/` 에 같이 남긴다.
- 핵심 코드(`probe_commit.py`, `quartile_paired.py`)를 바꿀 때는 **옵션을 안 주면 예전과 같은 출력**이 나오게 유지한다. 바꾼 뒤 기존 로그와 diff 해서 확인.
- 새 실험은 가설과 판정 기준을 먼저 적고 돌린다.