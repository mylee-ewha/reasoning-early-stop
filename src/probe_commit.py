"""학습된 판독기 — 정착이 원시 hidden state 에 흔적을 남기는가.

지금까지의 검정은 전부 '내가 고른 축으로 사영한 스칼라가 정착 시점에
변하는가' 였다. ‖h‖ 은 4096 좌표를 숫자 하나로 뭉개므로, 어떤 한 방향에서
정착 전후가 완벽히 갈려도 보이지 않는다. 선형 표현 가설이 보증하는 것은
요약통계가 아니라 **원시 상태 위의 선형 판독기**다. 이것이 마지막 시험이다.

    입력  h_t ∈ R^4096   (레이어 L, 체크포인트 t)
    라벨  y = 1  (t >= commit_step),  0  (t < commit_step)
    모형  선형 분류기를 학습 → 학습된 w 가 '방향'

세 가지 교란을 설계로 막는다.

 1. 위치 교란 — y=1 은 '정착 후' 이면서 동시에 '체인 뒤쪽' 이다. 위치는
    hidden state 에 당연히 들어 있으므로(RoPE, residual 누적), 아무 것도
    안 해도 AUC 가 올라간다. 정착 **직전 K 개와 직후 K 개만** 써서 두 집단의
    체인 내 위치를 맞춘다.
 2. 귀무 — 같은 체인의 **임의 지점**을 가짜 정착으로 놓고 똑같이 학습시킨다.
    위치만 읽는 판독기는 진짜와 가짜에서 같은 점수를 낸다. 판정 기준은
    AUC(진짜) − AUC(가짜) 이지 AUC(진짜) 자체가 아니다.
 3. 누수 — 같은 체인의 이웃 체크포인트가 train/test 양쪽에 들어가면 AUC 가
    공짜로 오른다. **예제 단위**로 분할한다.

 [2026-11-13 추가] 4. 극단값 행 — massive activation 으로 d2276 하나가 노름
    제곱의 99% 를 차지하는 행이 섞여 있다(check4 로 확정). StandardScaler 가
    좌표별로 눌러주긴 하나, 그 좌표의 std 자체가 이 행들 때문에 부풀려져
    일반 행은 ~0.3σ 로 눌리고 극단 행은 ~68σ 지렛점으로 남는다(추정).
    --drop-extreme 으로 제외하고 z 가 유지되는지 확인한다.

 [2026-11-17 추가] 교수님 면담 후 세 실험 (E1~E3). 옵션을 안 주면 예전과
    **완전히 같은 출력**이 나온다(--model ridge, --feat snap, 같은 난수열).

  E1 "선형이면 충분한가" — --model 에 여러 판독기를 주면 **같은 귀무 추첨**
     위에서 차례로 돌린다. 용량이 큰 판독기는 위치·예제도 더 잘 외우므로
     AUC 가 아니라 **차(진짜 − 같은 모델의 귀무)** 로 비교한다.
       dom    평균 차이 방향. 학습이 거의 없는 가장 단순한 선형
       ridge  지금 쓰는 것 (RidgeClassifierCV)
       logreg 로지스틱 (C 는 예제 단위 내부 2폴드로 고름)
       mlp    은닉 128 한 층 (PCA 후, alpha 는 내부 2폴드)
       rbf    Nystroem RBF 커널 + 릿지 (PCA 후, gamma 는 내부 2폴드)
     --pca 를 주면 모든 판독기에 같은 PCA 를 건다 → ridge 에도 걸어서
     "PCA 때문인지 비선형 때문인지" 를 가를 것.

  E2 "w 위의 궤적은 계단인가 경사인가" — --traj. 2폴드로 학습한 선형
     판독기로 **홀드아웃 예제의 체인 전체**(commit ±W 행)를 점수화하고
     commit 에 맞춰 정렬한다. 예제마다 'commit 에서의 3행 대 3행 점프' 가
     같은 예제의 다른 위치 점프들 사이에서 몇 분위인지(pct)를 본다.
     계단이면 pct → 1, 꾸준한 경사면 pct ≈ 0.5. matched 가짜 commit 으로
     같은 걸 해서 비교한다. --save-w 는 전체 데이터로 학습한 w 를 저장하고
     cos(dom, ridge) 와 층간 cos(w) 를 찍는다.

  E3 "변화량이 스냅샷에 정보를 더하는가" — --feat. 과거만 쓰는 인과 특징.
       snap    h_t (지금)
       delta   h_t − h_{t−lag}          (lag 은 **덤프 행** 단위)
       local   최근 win 행 평균
       global  체인 시작부터 t 까지 누적 평균
     쉼표로 붙인다: --feat snap,delta. 판독기를 선형으로 두면 특징의 효과만
     분리된다. 덤프 행 간격이 일정하지 않으므로(32토큰 격자 + 문단 경계)
     lag/win 은 토큰이 아니라 행 수다. 히스토리가 모자라면 있는 만큼만 쓴다
     (예제 집합을 특징 종류와 무관하게 똑같이 유지하기 위해).
     --drop-extreme 이 켜져 있으면 극단 행은 히스토리 계산에서도 뺀다 —
     안 빼면 누적 평균 하나가 그 행에 지배된다.

실행:
    python probe_commit.py sig_para500.jsonl --layers 18 27 36 --K 12 --nnull 100 --drop-extreme 10
    # E1
    python -u probe_commit.py sig_para500.jsonl --layers 36 --K 12 --nnull 100 --drop-extreme 10 \
        --null matched --model dom ridge logreg
    python -u probe_commit.py ... --null matched --pca 256 --model ridge mlp rbf
    # E2
    python -u probe_commit.py ... --model ridge --traj --traj-out traj --save-w w
    # E3
    python -u probe_commit.py ... --null matched --feat snap,delta --lag 4

메모리: 레이어 하나당 (체크포인트 수 x 4096 x 2B). 새 덤프 6층이면 5.1GB 전체.
"""
import argparse
import json
import sys
import warnings

import numpy as np

try:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
    from sklearn.metrics import roc_auc_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
except ImportError:
    sys.exit("sklearn 이 필요합니다:  pip install scikit-learn")

warnings.filterwarnings("ignore", category=UserWarning)
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except ImportError:
    pass

LINEAR = ("ridge", "dom", "logreg")
RIDGE_ALPHAS = (1e1, 1e2, 1e3, 1e4, 1e5)


# ──────────────────────── 데이터 ────────────────────────
def dump_map(out):
    """모델 층 번호 → 덤프 인덱스. 2026-11 재수집부터 덤프가 6층만 담는다.

    기존 35GB 덤프는 37층 전부라 항등사상이고 .dumpinfo.json 이 없다.
    새 덤프는 [9,18,27,31,35,36] 만 담으므로 --layers 18 이 dump[:,1] 이다.
    두 데이터셋을 같은 명령으로 돌릴 수 있게 여기서 흡수한다.
    """
    import os
    p = out + ".dumpinfo.json"
    if not os.path.exists(p):
        return None
    return json.load(open(p))["dump_layers"]


def load_layers(out, layers, chunk=2000):
    """.h.f16 을 한 번만 순차로 읽어 지정한 레이어만 메모리에 올린다."""
    meta = [json.loads(l) for l in open(out + ".meta.jsonl") if l.strip()]
    n = len(meta)
    ex = np.array([m["ex_id"] for m in meta])
    tt = np.array([m["t"] for m in meta])
    H = np.memmap(out + ".h.f16", np.float16, "r")
    nl1 = H.size // (n * 4096)
    H = H.reshape(n, nl1, 4096)

    dl = dump_map(out)
    if dl is not None:
        miss = [L for L in layers if L not in dl]
        if miss:
            raise SystemExit(f"덤프에 레이어 {miss} 가 없다. 담긴 층: {dl}")
        print(f"  덤프 {nl1}층 = 모델 층 {dl} → "
              f"{layers} 를 인덱스 {[dl.index(L) for L in layers]} 로 읽는다")
        layers = [dl.index(L) for L in layers]
    assert max(layers) < nl1, f"레이어 {max(layers)} 가 범위({nl1})를 넘는다"

    A = np.empty((n, len(layers), 4096), np.float16)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        A[s:e] = H[s:e][:, layers, :]
        if (s // chunk) % 10 == 0:
            print(f"  읽는 중 {e:,}/{n:,}", end="\r", flush=True)
    print(f"  읽기 완료 {n:,} 행, 레이어 {layers}        ")
    return A, ex, tt


# [2026-11-13 추가]
def norm_mask(A, li, k, chunk=20000):
    """노름이 중앙값의 k배를 넘는 행을 걸러낼 마스크.

    fp16 으로 제곱합을 구하면 넘친다(d2276=12416 → 1.5e8, fp16 최대 65504).
    반드시 float32 로 올려서 계산한다.
    """
    n = A.shape[0]
    nrm = np.empty(n, np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        blk = A[s:e, li, :].astype(np.float32)
        nrm[s:e] = np.sqrt((blk * blk).sum(1))
    med = float(np.median(nrm))
    return nrm <= k * med, nrm, med


def commit_of(out, use_grade2=True):
    """commit_step. grade2 가 있으면 보강 채점기로 다시 계산한다."""
    rows = [json.loads(l) for l in open(out)]
    if not use_grade2:
        return {r["ex_id"]: r.get("feat", {}).get("commit_step") for r in rows}
    try:
        from grade2 import eq_math2 as eq
    except ImportError:
        print("  grade2 없음 → jsonl 의 commit_step 을 그대로 쓴다")
        return {r["ex_id"]: r.get("feat", {}).get("commit_step") for r in rows}

    out_ = {}
    for r in rows:
        ps = r.get("probes", [])
        if not ps:
            out_[r["ex_id"]] = None
            continue
        reps, cls = [], []
        for s in [p["ans"] for p in ps] + [r["pred"]]:
            for i, q in enumerate(reps):
                if eq(s, q):
                    cls.append(i)
                    break
            else:
                reps.append(s)
                cls.append(len(reps) - 1)
        pc, fc = cls[:-1], cls[-1]
        j = len(pc)
        while j > 0 and pc[j - 1] == fc:
            j -= 1
        out_[r["ex_id"]] = ps[j]["t"] if j < len(pc) else None
    return out_


# ──────────────────────── [2026-11-17] E3: 인과 특징 ────────────────────────
def parse_feat(s):
    kinds = [k.strip() for k in s.split(",") if k.strip()]
    bad = [k for k in kinds if k not in ("snap", "delta", "local", "global")]
    if bad or not kinds:
        raise SystemExit(f"--feat 에 모르는 종류 {bad}. snap/delta/local/global")
    return kinds


def feat_rows(A, li, rs, positions, keep, kinds, lag, win):
    """예제 rs 의 positions(rs 안 인덱스)에서 인과 특징을 만든다.

    과거 행만 쓴다 — 온라인에서도 그대로 계산 가능하다. keep 이 있으면
    극단 행은 히스토리에서 뺀다(누적 평균 하나가 그 행에 지배되므로).
    positions 중 keep=False 인 행은 build 가 어차피 버리므로 값은 아무거나.
    """
    hi_pos = int(max(positions)) + 1
    H = A[rs[:hi_pos], li, :].astype(np.float32)
    mk = (keep[rs[:hi_pos]] if keep is not None
          else np.ones(hi_pos, bool))
    kp = np.flatnonzero(mk)
    if len(kp) == 0:
        kp = np.arange(hi_pos)
    Hk = H[kp]
    need_cs = ("local" in kinds) or ("global" in kinds)
    cs = np.cumsum(Hk, 0, dtype=np.float64) if need_cs else None

    out = []
    for p in positions:
        i = int(np.searchsorted(kp, p, "right")) - 1   # p 이하 마지막 유효 행
        i = max(i, 0)
        parts = []
        for k in kinds:
            if k == "snap":
                parts.append(H[p])
            elif k == "delta":
                q = kp[max(i - lag, 0)]
                parts.append(H[p] - H[q])
            elif k == "local":
                lo = i - win
                s = cs[i] - (cs[lo] if lo >= 0 else 0.0)
                parts.append((s / min(win, i + 1)).astype(np.float32))
            else:  # global
                parts.append((cs[i] / (i + 1)).astype(np.float32))
        out.append(np.concatenate(parts))
    return np.stack(out)


# ──────────────────────── 한 번의 학습 ────────────────────────
def build(A, li, spans, K, keep=None, center=False, feat=None):
    """spans: [(rows, j)]  → X, y, groups, used_rows, n_dropped.

    j 앞 K 개 = 0, 뒤 K 개 = 1.
    keep 가 주어지면 그 마스크가 False 인 행을 뺀다. 한쪽 클래스가 전부
    빠진 예제는 통째로 제외한다(AUC 를 계산할 수 없으므로).

    center=True 면 **예제 안에서** 창 24행의 평균을 빼고 남긴다.
      왜: matched 귀무는 컷의 위치 '분포'를 맞추지만, 예제 사이의 차이
      (이 예제가 길다 / 정착이 늦는 종류다)는 그대로 남는다. 2026-11-15
      실행에서 '체인 후반인가' AUC 0.773 과 정착 AUC 0.782 가 거의 같았다 —
      판독기가 '굳었나'를 읽는지 '어느 예제의 행인가'를 읽는지 구별이 안 된다.
      예제 평균을 빼면 예제 사이 분산이 0 이 되고 **같은 체인 안의 전/후
      대비만** 남는다. y=0 평균과 y=1 평균의 차는 그대로 살아남으므로,
      신호가 예제 내부에 있으면 AUC 가 유지되고 예제 사이에 있었으면
      0.5 로 주저앉는다. 귀무도 같이 센터링되니 비교는 공정하다.

    [2026-11-17] feat=(kinds, lag, win) 이고 kinds 가 ['snap'] 이 아니면
      feat_rows 로 인과 특징을 만든다. ['snap'] 이면 예전 경로 그대로.
    """
    plain = feat is None or feat[0] == ["snap"]
    Xs, ys, gs, rws = [], [], [], []
    ndrop, nskip = 0, 0
    for gi, (rs, j) in enumerate(spans):
        lo, hi = rs[j - K], rs[j + K - 1] + 1
        # [2026-11-13 추가] 연속성 검증 — --resume 재시작 구간이 섞이면
        # 슬라이스가 어긋나 조용히 다른 행을 읽는다
        if hi - lo != 2 * K:
            nskip += 1
            continue
        if plain:
            blk = A[lo:hi, li, :]
        else:
            blk = feat_rows(A, li, rs, np.arange(j - K, j + K), keep, *feat)
        yy = np.r_[np.zeros(K), np.ones(K)]
        rr = np.arange(lo, hi)
        if keep is not None:
            m = keep[lo:hi]
            d = int((~m).sum())
            if d:
                ndrop += d
                if len(np.unique(yy[m])) < 2:
                    nskip += 1
                    continue
                blk, yy, rr = blk[m], yy[m], rr[m]
        if center:
            blk = blk.astype(np.float32)
            blk = blk - blk.mean(0, keepdims=True)
        Xs.append(blk)
        ys.append(yy)
        gs.append(np.full(len(yy), gi))
        rws.append(rr)
    return (np.concatenate(Xs).astype(np.float32),
            np.concatenate(ys), np.concatenate(gs),
            np.concatenate(rws), ndrop, nskip)


# ──────────────────────── [2026-11-17] E1: 판독기 ────────────────────────
def _group_split(g, n_fold, seed):
    rg = np.random.default_rng(seed)
    uq = np.unique(g)
    rg.shuffle(uq)
    return np.array_split(uq, n_fold)


class Readout:
    """표준화 → (선택) PCA → 판독기. 하이퍼파라미터는 **예제 단위** 내부
    2폴드 AUC 로 고른다 — 행 단위로 고르면 같은 예제가 양쪽에 들어가서
    용량 큰 모델이 유리해진다. ridge 는 예전과 같게 RidgeClassifierCV 그대로.
    """
    GRID = {"logreg": (1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
            "mlp": (1e-2, 1e0, 1e2),
            "rbf": (0.25, 1.0, 4.0)}

    # [2026-11-18] 고른 하이퍼파라미터의 이름과 '가장 강한 규제' 쪽 끝값.
    #   끝값이 자주 골라지면 후보 범위가 모자란 것이다(더 강한 규제가 최적일 수 있음)
    HP_NAME = {"ridge": "alpha", "logreg": "C", "mlp": "alpha", "rbf": "gamma*d"}
    STRONG_END = {"ridge": max(RIDGE_ALPHAS), "logreg": min(GRID["logreg"]),
                  "mlp": max(GRID["mlp"]), "rbf": None}
    WEAK_END = {"ridge": min(RIDGE_ALPHAS), "logreg": max(GRID["logreg"]),
                "mlp": min(GRID["mlp"]), "rbf": None}

    def __init__(self, kind, pca=0, seed=0):
        self.kind, self.pca, self.seed = kind, pca, seed
        self.chosen = None

    def _pre_fit(self, X):
        self.sc = StandardScaler().fit(X)
        Z = self.sc.transform(X)
        self.pc = self.sc2 = None
        if self.pca:
            nc = int(min(self.pca, Z.shape[0] - 1, Z.shape[1]))
            self.pc = PCA(nc, random_state=self.seed).fit(Z)
            Z = self.pc.transform(Z)
            self.sc2 = StandardScaler().fit(Z)
            Z = self.sc2.transform(Z)
        return Z

    def _pre(self, X):
        Z = self.sc.transform(X)
        if self.pc is not None:
            Z = self.sc2.transform(self.pc.transform(Z))
        return Z

    def _make(self, hp, d, n):
        k = self.kind
        if k == "logreg":
            return LogisticRegression(C=hp, max_iter=3000)
        if k == "mlp":
            return MLPClassifier(hidden_layer_sizes=(128,), alpha=hp,
                                 max_iter=300, random_state=self.seed)
        if k == "rbf":
            return _RBF(gamma=hp / d, n_comp=int(min(1000, n)), seed=self.seed)
        raise ValueError(k)

    @staticmethod
    def _dec(clf, Z):
        if hasattr(clf, "decision_function"):
            return clf.decision_function(Z)
        return clf.predict_proba(Z)[:, 1]

    def fit(self, X, y, g):
        Z = self._pre_fit(X)
        if self.kind == "ridge":
            self.clf = RidgeClassifierCV(alphas=RIDGE_ALPHAS).fit(Z, y)
            self.chosen = float(self.clf.alpha_)
        elif self.kind == "dom":
            self.w = Z[y == 1].mean(0) - Z[y == 0].mean(0)
        else:
            best, best_auc = None, -1.0
            for hp in self.GRID[self.kind]:
                a_ = []
                for f in _group_split(g, 2, self.seed + 7):
                    te = np.isin(g, f)
                    tr = ~te
                    if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
                        continue
                    c = self._make(hp, Z.shape[1], tr.sum()).fit(Z[tr], y[tr])
                    a_.append(roc_auc_score(y[te], self._dec(c, Z[te])))
                m = float(np.mean(a_)) if a_ else -1.0
                if m > best_auc:
                    best, best_auc = hp, m
            self.hp = self.chosen = best
            self.clf = self._make(best, Z.shape[1], len(y)).fit(Z, y)
        return self

    def score(self, X):
        Z = self._pre(X)
        if self.kind == "dom":
            return Z @ self.w
        return self._dec(self.clf, Z)

    def w_raw(self):
        """원시 h 좌표계의 방향 (선형, PCA 없음일 때만)."""
        if self.kind not in LINEAR or self.pc is not None:
            return None
        if self.kind == "dom":
            w = self.w
        else:
            w = np.ravel(self.clf.coef_)
        return w / self.sc.scale_

    def w_std(self):
        if self.kind not in LINEAR or self.pc is not None:
            return None
        return self.w if self.kind == "dom" else np.ravel(self.clf.coef_)


class _RBF:
    """RBF 커널 근사(Nystroem) + 릿지.

    [2026-11-17 수정] sklearn Nystroem 은 기저 커널을 scipy svd(gesdd)로
    분해하는데, 귀무 반복 중 'SVD did not converge' 로 죽었다(L36, PCA 256).
    RBF 기저 커널은 대칭 양반정치라 eigh 로 분해하면 안정적이고, 아주 작은
    고유값(수치 잡음)은 버린다. 계산하는 특징은 Nystroem 과 같다.
    """

    def __init__(self, gamma, n_comp, seed):
        self.gamma, self.n_comp, self.seed = gamma, n_comp, seed

    def _k(self, X, Y):
        X = X.astype(np.float64); Y = Y.astype(np.float64)
        d2 = ((X * X).sum(1)[:, None] + (Y * Y).sum(1)[None, :]
              - 2.0 * X @ Y.T)
        return np.exp(-self.gamma * np.maximum(d2, 0.0))

    def _feat(self, Z):
        return self._k(Z, self.B) @ self.M

    def fit(self, Z, y):
        rg = np.random.default_rng(self.seed)
        m = int(min(self.n_comp, len(Z)))
        self.B = Z[rg.choice(len(Z), m, replace=False)].astype(np.float64)
        ev, U = np.linalg.eigh(self._k(self.B, self.B))
        ok = ev > max(ev.max(), 1e-12) * 1e-10
        self.M = U[:, ok] / np.sqrt(ev[ok])
        self.clf = RidgeClassifierCV(alphas=RIDGE_ALPHAS).fit(self._feat(Z), y)
        return self

    def decision_function(self, Z):
        return self.clf.decision_function(self._feat(Z))


def fit_auc(X, y, g, n_fold=2, seed=0, kind="ridge", pca=0, log=None):
    """예제 단위 분할 AUC. 폴드 평균. kind='ridge', pca=0 이면 예전과 동일."""
    folds = _group_split(g, n_fold, seed)
    aucs = []
    for f in folds:
        te = np.isin(g, f)
        tr = ~te
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        # [2026-11-17] 수치 오류 한 번에 몇 시간짜리 실행이 죽지 않도록,
        # 실패한 폴드는 경고만 찍고 건너뛴다(귀무는 NaN 을 걸러서 쓴다)
        try:
            m = Readout(kind, pca, seed).fit(X[tr], y[tr], g[tr])
            aucs.append(roc_auc_score(y[te], m.score(X[te])))
            if log is not None and m.chosen is not None:
                log.append(m.chosen)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError) as err:
            print(f"\n  ** {kind} seed={seed} 폴드 실패, 건너뜀: {err}", flush=True)
    return float(np.mean(aucs)) if aucs else np.nan


def hp_summary(kind, real_log, null_log):
    """[2026-11-18] 고른 하이퍼파라미터를 한 줄로. 진짜는 폴드별, 귀무는 빈도."""
    if kind not in Readout.HP_NAME or not real_log:
        return None
    from collections import Counter
    nm = Readout.HP_NAME[kind]
    r = ", ".join(f"{v:g}" for v in real_log)
    cnt = Counter(null_log)
    n = sum(cnt.values())
    nl = ", ".join(f"{v:g}×{c}" for v, c in sorted(cnt.items()))
    line = f"선택 {nm}: 진짜 [{r}]" + (f"  | 귀무 {n}회: {nl}" if n else "")
    end = Readout.STRONG_END[kind]
    if end is not None:
        k = sum(1 for v in list(real_log) + list(null_log) if v == end)
        tot = len(real_log) + n
        if k:
            line += (f"  ** 가장 강한 규제 끝값 {end:g} 이 {k}/{tot}회 선택 "
                     f"→ 후보 범위를 더 강한 쪽으로 넓힐 것")
    wend = Readout.WEAK_END[kind]
    if wend is not None:
        k = sum(1 for v in list(real_log) + list(null_log) if v == wend)
        tot = len(real_log) + n
        if k:
            line += (f"  ** 가장 약한 규제 끝값 {wend:g} 이 {k}/{tot}회 선택"
                     + (" (행 단위 LOO 의 '이웃 행 외우기' 증상일 수 있음)"
                        if kind == "ridge" else ""))
    return line


# ──────────────────────── [2026-11-17] E2: 궤적 ────────────────────────
def traj_scores(A, li, spans, a, keep, feat, kind):
    """2폴드: 학습 예제의 ±K 창으로 판독기를 학습하고, 홀드아웃 예제의
    commit ±W 행 전체를 점수화해 commit 에 정렬한다. 반환 [n, 2W] (NaN 패딩).
    오프셋 0 = commit 이후 첫 행(j). 센터링은 학습과 같게 ±K 창 평균을 뺀다
    (선형 판독기에서는 입력 센터링과 정확히 같다)."""
    W, K = a.traj_w, a.K
    X, y, g, _, _, _ = build(A, li, spans, K, keep, a.center_ex, feat)
    out = []
    for f in _group_split(g, a.folds, 1):
        te = np.isin(g, f)
        if len(np.unique(y[te])) < 2 or len(np.unique(y[~te])) < 2:
            continue
        m = Readout(kind, a.pca, 1).fit(X[~te], y[~te], g[~te])
        for gi in f:
            rs, j = spans[int(gi)]
            pos = np.arange(max(j - W, 0), min(j + W, len(rs)))
            if feat is None or feat[0] == ["snap"]:
                Xe = A[rs[pos], li, :].astype(np.float32)
            else:
                Xe = feat_rows(A, li, rs, pos, keep, *feat)
            s = m.score(Xe).astype(np.float64)
            if keep is not None:
                s[~keep[rs[pos]]] = np.nan
            if a.center_ex:
                win = (pos >= j - K) & (pos < j + K)
                s = s - np.nanmean(s[win])
            row = np.full(2 * W, np.nan)
            row[pos - (j - W)] = s
            out.append(row)
    return np.array(out)


def step_pct(S, W, h=3):
    """예제마다 commit 의 h행 대 h행 점프가 같은 예제의 다른 위치 점프들
    사이에서 몇 분위인가. 계단이면 1 에 가깝고, 꾸준한 경사면 0.5 근처."""
    pcts, jumps = [], []
    for r in S:
        def jmp(c):
            a_, b_ = r[c - h:c], r[c:c + h]
            if np.isnan(a_).any() or np.isnan(b_).any():
                return np.nan
            return b_.mean() - a_.mean()
        J = jmp(W)
        if not np.isfinite(J):
            continue
        others = [jmp(c) for c in range(h, 2 * W - h + 1)
                  if abs(c - W) > h - 1]
        others = np.array([o for o in others if np.isfinite(o)])
        if len(others) < 5:
            continue
        pcts.append(float((others < J).mean()))
        jumps.append(J)
    return np.array(pcts), np.array(jumps)


def report_traj(L, S_real, S_fake, W, out_prefix):
    print(f"\n  [E2 · L{L}] w 위의 궤적 (홀드아웃, commit 정렬, 오프셋 0 = commit 직후 첫 행)")
    offs = [-24, -12, -6, -3, -2, -1, 0, 1, 2, 5, 11, 23]
    offs = [o for o in offs if -W <= o < W]
    hdr = "  ".join(f"{o:>+5d}" for o in offs)
    print(f"    오프셋      {hdr}")
    for name, S in (("진짜", S_real), ("가짜", S_fake)):
        if S is None or len(S) == 0:
            continue
        mu = np.nanmean(S, 0)
        print(f"    {name} 평균  " + "  ".join(f"{mu[o + W]:>+5.2f}" for o in offs)
              + f"   (n={len(S)})")
    pr, jr = step_pct(S_real, W)
    line = (f"    계단 지수 pct: 진짜 중앙 {np.median(pr):.3f}, "
            f"pct≥0.95 비율 {np.mean(pr >= 0.95)*100:.1f}% (n={len(pr)})")
    if S_fake is not None and len(S_fake):
        pf, jf = step_pct(S_fake, W)
        line += (f"  |  가짜 중앙 {np.median(pf):.3f}, "
                 f"≥0.95 {np.mean(pf >= 0.95)*100:.1f}% (n={len(pf)})")
        try:
            from scipy.stats import mannwhitneyu
            p = mannwhitneyu(pr, pf, alternative="greater").pvalue
            line += f"  |  진짜>가짜 MW p={p:.2g}"
        except ImportError:
            pass
    print(line)
    print("    → 진짜 pct 가 1 쪽에 몰리고 가짜는 0.5 근처면 '계단'(정착 = 사건). "
          "진짜도 0.5 근처면 w 는 '경사'(위치/진행도)를 읽는다")
    if out_prefix:
        fn = f"{out_prefix}_L{L}.npz"
        np.savez(fn, real=S_real, fake=S_fake if S_fake is not None else np.empty(0),
                 offsets=np.arange(-W, W))
        print(f"    저장: {fn}  (그림용: real/fake [예제, 2W], offsets)")


# ──────────────────────── 본체 ────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="sig_n500c.jsonl")
    ap.add_argument("--layers", type=int, nargs="+", default=[18, 27, 36])
    ap.add_argument("--K", type=int, default=16, help="정착 전/후 체크포인트 수")
    ap.add_argument("--nnull", type=int, default=20, help="가짜 정착 반복 수")
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--max-ex", type=int, default=0, help="0=전부")
    ap.add_argument("--drop-extreme", type=float, default=0.0,
                    help="노름이 중앙값의 이 배수를 넘는 행을 제외한다. "
                         "0=끔. massive activation 행의 지렛점 효과를 확인할 때 10 권장")
    ap.add_argument("--center-ex", action="store_true",
                    help="예제 안에서 창 평균을 빼고 학습한다. 예제 사이 차이를 "
                         "전부 없애고 같은 체인 안의 전/후 대비만 남긴다 — "
                         "'체인 후반인가' AUC 가 정착 AUC 와 비슷할 때 필수")
    ap.add_argument("--min-commit", type=int, default=0,
                    help="commit 이 이 토큰보다 이른 예제를 제외한다. "
                         "256 격자 데이터에서 첫 probe 점에 검열된 집단을 빼고 "
                         "문단 격자 데이터와 같은 부분집합을 비교할 때 쓴다. 0=끔")
    ap.add_argument("--null", choices=("matched", "uniform", "both"),
                    default="both",
                    help="matched=다른 예제의 정착 상대위치를 빌려 쓴다. "
                         "uniform=체인 안 아무 데나 — 위치 교란에 뚫린다. "
                         "both=둘 다 한 번에(기본, 35GB 재읽기를 아낀다)")
    # [2026-11-17] E1~E3
    ap.add_argument("--model", nargs="+", default=["ridge"],
                    choices=("ridge", "dom", "logreg", "mlp", "rbf"),
                    help="E1. 여러 개를 주면 같은 귀무 추첨 위에서 차례로 돈다")
    ap.add_argument("--pca", type=int, default=0,
                    help="E1. 모든 판독기에 같은 PCA 차원. mlp/rbf 는 0 이면 256 으로")
    ap.add_argument("--feat", default="snap",
                    help="E3. snap,delta,local,global 을 쉼표로 (기본 snap = 예전과 같음)")
    ap.add_argument("--lag", type=int, default=4, help="E3 delta 의 간격(덤프 행)")
    ap.add_argument("--win", type=int, default=8, help="E3 local 창(덤프 행)")
    ap.add_argument("--traj", action="store_true",
                    help="E2. 첫 번째 선형 --model 로 홀드아웃 궤적을 commit 정렬")
    ap.add_argument("--traj-w", type=int, default=36, help="E2 정렬 반폭(덤프 행)")
    ap.add_argument("--traj-out", default="", help="E2 궤적 저장 접두어(.npz)")
    ap.add_argument("--save-w", default="",
                    help="E2. 전체 데이터로 학습한 ridge/dom w 를 저장할 접두어. "
                         "cos(dom, ridge) 와 층간 cos 를 찍는다")
    a = ap.parse_args()

    kinds = parse_feat(a.feat)
    feat = (kinds, a.lag, a.win)
    pca_of = {m: (a.pca if a.pca else (256 if m in ("mlp", "rbf") else 0))
              for m in a.model}

    A, ex, tt = load_layers(a.out, a.layers)
    C = commit_of(a.out)

    idx = {int(e): np.where(ex == e)[0] for e in np.unique(ex)}
    TL = {r["ex_id"]: r.get("think_len") for r in map(json.loads, open(a.out))}
    spans, span_e, dropped, nfloor = [], [], [], 0
    for e, rs in idx.items():
        cs = C.get(e)
        if cs is None:
            continue
        # 격자 하한에 걸린(censored) 예제를 뺀다. 256 격자에서는 commit 이
        # 256 보다 이를 수 없으므로, 첫 probe 점에 찍힌 예제는 "256 에서
        # 굳었다"가 아니라 "그 전에 굳었는데 못 봤다"다. 문단 격자와 집단을
        # 맞출 때 쓴다 — 두 데이터셋의 **같은 부분집합**을 비교하기 위한 것
        if a.min_commit and cs < a.min_commit:
            nfloor += 1
            continue
        j = int(np.searchsorted(tt[rs], cs))
        if j >= a.K and j + a.K <= len(rs):
            spans.append((rs, j)); span_e.append(e)
        else:
            dropped.append(e)
    if a.max_ex:
        spans = spans[:a.max_ex]; span_e = span_e[:a.max_ex]
    print(f"\n쓸 수 있는 예제 {len(spans)}건 "
          f"(정착 있고 전후 {a.K}개씩 확보),  표본 {2*a.K*len(spans):,}개")
    if nfloor:
        print(f"  --min-commit {a.min_commit}: commit < {a.min_commit} 인 "
              f"{nfloor}건 사전 제외 (격자 하한 검열 집단)")

    # ── [2026-11-15 추가] 탈락 예제의 성격. 창에 안 들어가는 예제는
    #    **정착이 가장 이른** 예제들이고, 그건 절감 여지가 가장 큰 집단이다.
    #    무작위 탈락이 아니라 체계적 탈락이므로 어느 쪽인지 찍어둔다
    if dropped:
        kf = np.array([C[e] / max(TL.get(e) or 1, 1) for e in span_e], float)
        df = np.array([C[e] / max(TL.get(e) or 1, 1) for e in dropped], float)
        ks = np.array([C[e] for e in span_e], float)
        ds = np.array([C[e] for e in dropped], float)
        print(f"  탈락 {len(dropped)}건 / 정착 있는 {len(spans)+len(dropped)}건 "
              f"({len(dropped)/(len(spans)+len(dropped))*100:.1f}%)")
        print(f"    commit_step  유지 중앙 {np.median(ks):>6.0f} / "
              f"탈락 중앙 {np.median(ds):>6.0f}")
        print(f"    commit_frac  유지 중앙 {np.median(kf):>6.3f} / "
              f"탈락 중앙 {np.median(df):>6.3f}")
        print("    → 탈락이 더 이른 쪽이면 체계적 탈락이다. 아래 사분위별 AUC 로")
        print("      '이른 정착에서 신호가 약한가'를 확인할 것")
    print(f"위치 통제: 각 예제에서 정착 직전 {a.K}개(y=0) / 직후 {a.K}개(y=1)만 사용")
    if a.drop_extreme > 0:
        print(f"극단값 제외: 노름 > {a.drop_extreme:g}× 중앙값 인 행을 뺀다")
    # [2026-11-17]
    print(f"판독기: {', '.join(f'{m}(pca={pca_of[m]})' for m in a.model)}   "
          f"특징: {'+'.join(kinds)}"
          + (f" (lag={a.lag}행, win={a.win}행)" if kinds != ['snap'] else ""))
    print()

    # 경험적 p 의 하한은 1/(nnull+1) 이다. 이게 유의수준보다 크면 그 검정은
    # 애초에 기각할 수 없다 — 2026-10-25 에 순열 300회로 같은 실수를 했다.
    floor = 1.0 / (a.nnull + 1)
    print(f"가짜 정착 {a.nnull}회 → 경험적 p 의 하한 {floor:.3f}", end="")
    bonf = 0.05 / max(1, len(a.layers))
    if floor > bonf:
        print(f"  ** {len(a.layers)}층 Bonferroni({bonf:.4f}) 보다 크다 — "
              f"어떤 층도 p 로는 통과 불가. --nnull >= {int(np.ceil(1/bonf))} 필요 **")
    else:
        print()
    print("z 는 하한이 없으므로 둘을 같이 본다 (|z| >= 3 이면 분명한 차이)\n")

    rg = np.random.default_rng(0)
    print("귀무 matched = 다른 예제의 정착 상대위치를 빌려 쓴다 "
          "(위치 분포 보존, 예제-정착 짝만 깸)")
    print("     uniform = 체인 안 균등 — 실제 정착이 앞쪽에 쏠려 있으면 "
          "위치 차이만으로 거짓양성이 난다\n")
    multi = len(a.model) > 1
    print(f"{'레이어':>6} " + (f"{'모델':<7}" if multi else "")
          + f"{'귀무':<8} {'진짜':>7} {'가짜평균':>9} {'가짜sd':>8} "
          f"{'차':>8} {'z':>7} {'p':>8}")
    W_raw = {}
    for li, L in enumerate(a.layers):
        # [2026-11-13 추가] 레이어마다 노름 분포가 다르므로 마스크도 레이어별
        keep = None
        if a.drop_extreme > 0:
            keep, nrm, med = norm_mask(A, li, a.drop_extreme)
            print(f"  [L{L}] 전체 노름 중앙 {med:.1f}, "
                  f"임계 {a.drop_extreme*med:.1f} 초과 {int((~keep).sum()):,}행 "
                  f"({(~keep).mean()*100:.3f}%)  최대 {nrm.max():.0f}")

        X, y, g, rw, ndrop, nskip = build(A, li, spans, a.K, keep, a.center_ex, feat)
        if a.drop_extreme > 0 or nskip:
            print(f"  [L{L}] 학습 표본에서 제외 {ndrop}행, "
                  f"예제 제외 {nskip}건 → 최종 {len(y):,}행 / "
                  f"{len(np.unique(g))}예제")
        rlog = {m: [] for m in a.model}
        real = {m: fit_auc(X, y, g, a.folds, seed=1, kind=m, pca=pca_of[m],
                           log=rlog[m])
                for m in a.model}

        # 귀무의 컷 위치. uniform 은 체인 안 균등이라 **위치 분포가 다르다** —
        # 실제 정착은 체인 앞쪽(상대위치 평균 0.29)에 몰려 있고 균등은 0.52 다.
        # 체인 앞뒤로 내용이 달라지면(문제 읽기 → 풀기) 그 차이만으로 AUC 가
        # 벌어져 거짓양성이 난다. matched 는 **다른 예제의 정착 상대위치**를
        # 빌려 써서 위치 분포를 보존하고 예제-정착 짝만 깬다.
        # [2026-11-17] 추첨은 모드마다 한 번만 하고 모든 판독기가 같은 추첨을
        # 쓴다(판독기 사이 비교가 공정해지도록). 난수 순서는 예전과 같다.
        fracs = np.array([j / len(rs) for rs, j in spans])
        modes = ("matched", "uniform") if a.null == "both" else (a.null,)
        first_fake = None
        for mi, mode in enumerate(modes):
            null = {m: [] for m in a.model}
            nlog = {m: [] for m in a.model}
            for b in range(a.nnull):
                ns = []
                if mode == "matched":
                    perm = rg.permutation(len(spans))
                    for i, (rs, _) in enumerate(spans):
                        m_ = len(rs)
                        jj = int(round(fracs[perm[i]] * m_))
                        ns.append((rs, int(np.clip(jj, a.K, m_ - a.K))))
                else:
                    for rs, _ in spans:
                        m_ = len(rs)
                        ns.append((rs, int(rg.integers(a.K, m_ - a.K))))
                if mode == "matched" and first_fake is None:
                    first_fake = ns
                Xn, yn, gn, _, _, _ = build(A, li, ns, a.K, keep, a.center_ex, feat)
                for m in a.model:
                    null[m].append(fit_auc(Xn, yn, gn, a.folds, seed=100 + b,
                                           kind=m, pca=pca_of[m], log=nlog[m]))
                if multi and (b + 1) % 10 == 0:
                    print(f"    [{mode}] 귀무 {b+1}/{a.nnull}", end="\r", flush=True)
            for k, m in enumerate(a.model):
                nl = np.array([v for v in null[m] if np.isfinite(v)])
                if len(nl) < len(null[m]):
                    print(f"  ** {m}: 귀무 {len(null[m]) - len(nl)}회 실패 → {len(nl)}회로 계산")
                d = real[m] - nl.mean()
                z = d / (nl.std(ddof=1) + 1e-9)
                p = (np.sum(nl >= real[m]) + 1) / (len(nl) + 1)
                tag = f"{L}" if (mi == 0 and k == 0) else ""
                print(f"{tag:>6} " + (f"{m:<7}" if multi else "")
                      + f"{mode:<8} {real[m]:>7.3f} {nl.mean():>9.3f} "
                      f"{nl.std(ddof=1):>8.3f} {d:>+8.3f} {z:>+7.2f} {p:>8.3f}"
                      + ("  **" if p < 0.05 and d > 0 else ""))
                hs = hp_summary(m, rlog[m], nlog[m])
                if hs:
                    print(f"{'':>6}   ↳ {hs}")

        m0 = a.model[0]
        # 참고: 위치 자체는 얼마나 읽히나 (체인 전/후반 분류)
        if True:
            # [2026-11-13 수정] build 가 실제로 쓴 행(rw)으로 라벨을 만든다.
            # 예전에는 spans 에서 다시 만들어서, 행이 빠지면 길이가 어긋났다
            ex_of = ex[rw]
            yp = np.array([float(tt[r] > np.median(tt[idx[int(e)]]))
                           for r, e in zip(rw, ex_of)])
            if len(np.unique(yp)) == 2:
                print(f"{'(참고)':>6} 같은 표본으로 '체인 후반인가'를 맞히면 "
                      f"AUC {fit_auc(X, yp, g, a.folds, seed=2, kind=m0, pca=pca_of[m0]):.3f}"
                      + (f"  [{m0}]" if multi else ""))

        # ── [2026-11-15 추가] 정착 상대위치 삼분위별 AUC.
        #    체계적 탈락(이른 정착이 빠진다)이 결론을 바꾸는지 보는 장치다.
        #    AUC 가 삼분위에 걸쳐 평평하면 탈락이 결론을 흔들지 않는다.
        #    이른 쪽에서 낮으면 → 빠진 집단이 신호가 약한 쪽이고 real 은 낙관적
        if len(spans) >= 60:
            fr3 = np.quantile(fracs, [1/3, 2/3])
            lab = np.digitize(fracs, fr3, right=True)
            parts = []
            for t in range(3):
                sel = [s for s, l in zip(spans, lab) if l == t]
                if len(sel) < 20:
                    parts.append("  n부족")
                    continue
                Xt, yt, gt, _, _, _ = build(A, li, sel, a.K, keep, a.center_ex, feat)
                parts.append(f"{fit_auc(Xt, yt, gt, a.folds, seed=3, kind=m0, pca=pca_of[m0]):.3f}"
                             f"(n={len(sel)})")
            print(f"{'(삼분위)':>6} 정착 상대위치별 AUC  이른 {parts[0]}  "
                  f"중간 {parts[1]}  늦은 {parts[2]}"
                  + (f"  [{m0}]" if multi else ""))

        # ── [2026-11-17] E2 궤적
        if a.traj:
            lin = [m for m in a.model if m in LINEAR and pca_of[m] == 0]
            if not lin:
                print("  [E2] --traj 는 PCA 없는 선형 판독기(ridge/dom/logreg)에서만 돈다")
            else:
                if first_fake is None:   # --null uniform 만 준 경우 matched 1회 추첨
                    perm = np.random.default_rng(12345).permutation(len(spans))
                    first_fake = []
                    for i, (rs, _) in enumerate(spans):
                        m_ = len(rs)
                        jj = int(round(fracs[perm[i]] * m_))
                        first_fake.append((rs, int(np.clip(jj, a.K, m_ - a.K))))
                Sr = traj_scores(A, li, spans, a, keep, feat, lin[0])
                Sf = traj_scores(A, li, first_fake, a, keep, feat, lin[0])
                report_traj(L, Sr, Sf, a.traj_w, a.traj_out)

        # ── [2026-11-17] E2 w 저장, cos(dom, ridge)
        if a.save_w:
            if kinds != ["snap"]:
                print("  [E2] --save-w 는 --feat snap 에서만 (특징 공간이 다르면 w 의 뜻이 바뀜)")
            else:
                mr = Readout("ridge", 0, 0).fit(X, y, g)
                md = Readout("dom", 0, 0).fit(X, y, g)
                ws_r, ws_d = mr.w_std(), md.w_std()
                cs_ = float(ws_r @ ws_d / (np.linalg.norm(ws_r) * np.linalg.norm(ws_d) + 1e-12))
                W_raw[L] = mr.w_raw()
                fn = f"{a.save_w}_L{L}.npz"
                np.savez(fn, w_ridge_raw=mr.w_raw(), w_dom_raw=md.w_raw(),
                         w_ridge_std=ws_r, w_dom_std=ws_d,
                         mean=mr.sc.mean_, scale=mr.sc.scale_,
                         center_ex=a.center_ex, drop_extreme=a.drop_extreme, K=a.K)
                print(f"  [E2 · L{L}] cos(dom, ridge) 표준화 공간 = {cs_:+.3f}"
                      f"   저장: {fn}")
                print("      → 1 에 가까우면 릿지 w 는 '두 집단 평균 차이' 방향과 같다 "
                      "(가장 단순한 선형으로 충분하다는 신호)")

    if a.save_w and len(W_raw) > 1:
        Ls = sorted(W_raw)
        print("\n  [E2] 층간 cos(w_ridge) — 원시 좌표계 (residual 기저 공유 가정)")
        print("        " + "  ".join(f"L{L:>3}" for L in Ls))
        for L1 in Ls:
            r = []
            for L2 in Ls:
                u, v = W_raw[L1], W_raw[L2]
                r.append(float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)))
            print(f"    L{L1:>3} " + "  ".join(f"{x:+.2f}" for x in r))

    print("\n판정: '차' 가 0 에 붙고 p 가 크면 → 정착은 원시 표현에 "
          "선형으로 읽히는 흔적을 남기지 않는다 (H4 확정).")
    print("      '차' 가 뚜렷이 양수면 → 지금까지의 음성 결론이 뒤집힌다. "
          "어느 레이어인지까지 나온다.")
    if multi:
        print("E1: 판독기끼리는 '차' 와 z 로 비교한다. 비선형의 차가 ridge 의 차보다 "
              "ridge 귀무 sd 이상 크지 않으면 '선형으로 충분'.")


if __name__ == "__main__":
    main()