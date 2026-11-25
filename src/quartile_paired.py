#!/usr/bin/env python3
"""
길이 사분위 안에서 learned readout 을 두 귀무와 짝지어 비교한다. (v5)

v5 (2026-11-16) — 채점기 통일
  evaluate() 가 멈춘 예제는 grade2.eq_math2 로, 안 멈춘 완주 예제는 jsonl 에
  저장된 correct(수집 시 내장 eq_math, antlr 없어 열화)로 매기고 있었다.
  그래서 '많이 멈추는 팔'일수록 채점기 덕을 봤다. 기준선은 전부 안 멈추므로
  3%p 가량 깎였고, 오라클이 기준선보다 +3.0%p 높게 나온 것(정의상 0 이어야
  함)과 commit 집단 +12건 '고침'이 이 비대칭이었다(내장=오답·grade2=정답 13건).
  이제 완주 예제도 pred 를 grade2 로 다시 채점한다.

v3 결과에서 걸린 것
  사분위로 묶어도 구간 안에 길이가 1.4~2.8배 남아 있고, **규칙의 우세가
  그 잔여 분산과 순위가 일치**했다(Q1 2.79배 +13.6%p / Q3 1.89배 +4.4%p /
  Q2 1.67배 −2.0%p / Q4 1.37배 −2.4%p). 사분위가 길이를 충분히 통제하지
  못했을 가능성이 있다.

v4 에서 추가한 팔 — 오라클 길이 × ρ
  각 예제를 **자기 최종 길이의 ρ 배** 지점에서 자른다. 최종 길이는 배포 시
  알 수 없으므로 이 팔은 반칙이다. 바로 그래서 좋은 귀무다.
    · 규칙이 이 팔을 이기면  → 잔여 길이가 아니다. Q1 결과를 그대로 쓸 수 있다
    · 비기면                → 규칙은 길이 예측기다
  (per-example 오라클은 이 귀무를 같은 절감에서 답변경 0% vs 40.1% 로 이겼다.)

진단 하나 더
  정지 지점과 think_len 의 Spearman. fixed_T 는 0(상수), 오라클ρ 는 1(비례).
  규칙이 1 에 가까우면 길이를 읽는 것이다.

사용:
  python quartile_paired.py sig_n500c.jsonl --layer 18 --K 12 \
      --folds 5 --targets 0.3,0.5,0.7 --save-w w_L18.npz
"""
import argparse, json, os, sys
from math import comb

import numpy as np

try:
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.preprocessing import StandardScaler
except ImportError:
    sys.exit("sklearn 이 필요합니다")

try:
    from grade2 import eq_math2 as _eq
    GRADER = "grade2.eq_math2"
except Exception:
    import re
    _WS = re.compile(r"\s+")
    def _eq(a, b):
        f = lambda s: _WS.sub("", (s or "").replace("\\left", "").replace("\\right", "")
                             .replace("$", "").replace("\\!", "").strip().rstrip("."))
        return f(a) == f(b)
    GRADER = "폴백(정규화 문자열) — 숫자를 보수적으로 읽을 것"

# ── [2026-11-15] 채점 결과 캐시. 이게 이 스크립트의 병목이다.
#    pick_best 가 임계 후보 120개 × 5폴드 × 12셀 을 훑고, 후보마다
#    evaluate 가 예제별로 eq_math2(sympy)를 부른다. (답, gold) 쌍은
#    계속 같은 것이 반복되므로 캐시 한 줄로 수십 배 빨라진다.
_eq_raw = _eq
_eq_cache = {}


def _eq(a, b):
    k = (a, b)
    v = _eq_cache.get(k, None)
    if v is None:
        v = bool(_eq_raw(a, b))
        _eq_cache[k] = v
    return v


def load_layer(out, layer, chunk=2000):
    meta = [json.loads(l) for l in open(out + ".meta.jsonl") if l.strip()]
    n = len(meta)
    ex = np.array([m["ex_id"] for m in meta])
    tt = np.array([m["t"] for m in meta])
    H = np.memmap(out + ".h.f16", np.float16, "r")
    nl = H.size // (n * 4096)
    H = H.reshape(n, nl, 4096)

    # 2026-11 재수집부터 덤프가 6층만 담는다. 기존 35GB 덤프(37층, dumpinfo
    # 없음)는 항등사상. 두 데이터셋을 같은 --layer 번호로 돌릴 수 있게 흡수
    p = out + ".dumpinfo.json"
    if os.path.exists(p):
        dl = json.load(open(p))["dump_layers"]
        if layer not in dl:
            raise SystemExit(f"덤프에 레이어 {layer} 가 없다. 담긴 층: {dl}")
        print(f"  덤프 {nl}층 = 모델 층 {dl} → "
              f"레이어 {layer} 를 인덱스 {dl.index(layer)} 로 읽는다")
        layer = dl.index(layer)
    assert layer < nl, f"레이어 {layer} 가 범위({nl})를 넘는다"
    A = np.empty((n, 4096), np.float16)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        A[s:e] = H[s:e, layer, :]
        if (s // chunk) % 20 == 0:
            print(f"  읽는 중 {e:,}/{n:,}", end="\r", flush=True)
    print(f"  읽기 완료 {n:,} 행, 레이어 {layer}            ")
    return A, ex, tt


def commit_of(rows):
    out = {}
    for r in rows:
        ps = r.get("probes", [])
        if not ps:
            out[r["ex_id"]] = None; continue
        reps, cls = [], []
        for s in [p["ans"] for p in ps] + [r["pred"]]:
            for i, q in enumerate(reps):
                if _eq(s, q):
                    cls.append(i); break
            else:
                reps.append(s); cls.append(len(reps) - 1)
        pc, fc = cls[:-1], cls[-1]
        j = len(pc)
        while j > 0 and pc[j - 1] == fc:
            j -= 1
        out[r["ex_id"]] = ps[j]["t"] if j < len(pc) else None
    return out


def ans_at(probes, t):
    best = None
    for p in probes:
        if p["t"] <= t:
            best = p["ans"]
        else:
            break
    return best


def snap_up(probes, t):
    for p in probes:
        if p["t"] >= t:
            return p["t"]
    return None


def evaluate(EX, stops, extra=None):
    """extra: 예제별 추가 토큰 비용(검증기의 탈락 probe 등)."""
    tot = used = 0
    ok = []
    if extra is not None:
        used += float(np.sum(extra))
    for e, s in zip(EX, stops):
        L = e["think_len"]
        tot += L
        if s is None or s >= L:
            used += L
            if not e["fin"]:
                a = ans_at(e["probes"], L)
                ok.append(bool(a is not None and _eq(a, e["gold"])))
            else:
                # [v5] 저장된 correct 는 내장 eq_math 결과라 멈춘 예제(grade2)와
                # 채점기가 달랐다. 완주 예제도 최종 답을 grade2 로 다시 채점한다
                ok.append(bool(e["pred"] is not None and _eq(e["pred"], e["gold"])))
        else:
            used += s
            a = ans_at(e["probes"], s)
            ok.append(bool(a is not None and _eq(a, e["gold"])))
    return 1 - used / max(1, tot), np.array(ok, bool)


def sign_p(n01, n10):
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def rule_stops(EX, sc, tau):
    out = []
    for e, s in zip(EX, sc):
        hit = np.where(s > tau)[0]
        out.append(snap_up(e["probes"], int(e["ts"][hit[0]])) if len(hit) else None)
    return out


def rho_stops(EX, rho):
    """오라클 길이 × ρ — 최종 길이를 안다고 치고 그 비율 지점에서 자른다."""
    return [snap_up(e["probes"], int(rho * e["think_len"])) for e in EX]


def pred_stops(EX, rho, key="pred_len"):
    """**예측** 길이 × ρ. 오라클ρ 와 달리 배포 가능하다.

    v4 결과: 규칙이 fixed_T 는 이기는데 오라클ρ 와는 무승부였다. 그러면
    "규칙은 길이 예측기인가"가 남는다. 이 팔이 그걸 직접 가른다.
      규칙 ≈ 이 팔  → 규칙은 길이 예측기다
      규칙 >  이 팔  → 길이를 넘는 정보가 있다
    """
    return [snap_up(e["probes"], int(rho * max(e[key], 1))) for e in EX]


def verifier_stops(EX, sc, tau):
    """readout 발화 지점을 순서대로 보며 '멈춰도 되는' 첫 지점에서 종료.

    가역 정지의 **오라클 검증기**다. gold 로 판정하므로 어떤 실제 검증기도
    이걸 못 넘는다. 탈락한 발화는 probe 1회 비용(15.8토큰)만 내고 재개한다 —
    거짓양성 비용이 '예제 1개'가 아니라 '상수'가 되는 그 구조.

    이 팔과 규칙-단독의 차이 = 검증 기준을 찾는 일의 이득 상한.
    차이가 작으면 가역 정지 전체가 닫힌다(KL·섭동을 재볼 이유가 없다).
    """
    PROBE_COST = 15.8
    stops, extra = [], []
    for e, s in zip(EX, sc):
        seen, chosen, nrej = set(), None, 0
        for h in np.where(s > tau)[0]:
            t = snap_up(e["probes"], int(e["ts"][h]))
            if t is None or t in seen:
                continue                      # 같은 probe 점 중복 발화는 1회
            seen.add(t)
            a = ans_at(e["probes"], t)
            if a is not None and _eq(a, e["gold"]):
                chosen = t
                break
            nrej += 1
        stops.append(chosen)
        extra.append(nrej * PROBE_COST)
    return stops, extra


def pick_best(EX, target, cands, mk):
    """목표 절감률에 가장 가까운 임계값을 **학습 폴드에서만** 고른다.

    mk 는 stops 또는 (stops, extra) 를 돌려줄 수 있다. 검증기 팔은
    탈락 probe 비용이 있어 그걸 빼고 고르면 절감률이 과대평가된다.
    """
    best, bd = cands[0], 9e9
    for c in cands:
        r = mk(EX, c)
        s, _ = evaluate(EX, *r) if isinstance(r, tuple) else evaluate(EX, r)
        d = abs(s - target)
        if d < bd:
            best, bd = c, d
    return best


def nocommit_diag(EX, SC, fold_of, nfolds, targets):
    """commit 없는 예제에서 규칙이 어떻게 행동하나.

    학습은 commit 있는 예제로만 한다. 그래서 commit 없는 예제에서의 행동은
    **외삽**이고, 학습으로 보장되는 게 없다. 여기서 직접 잰다.

    주의: 'commit 없음' 은 '멈추면 해롭다' 가 아니다. 멈춰서 읽은 답이
    모델의 최종답과 다르다는 뜻일 뿐이고, gold 기준으로는 더 좋을 수도 있다
    (중간에 맞혔다가 뒤에서 틀려지는 경우). 그래서 발화율만 보면 안 되고
    **그 집단의 정확도가 베이스보다 떨어지는지**를 봐야 한다.
    """
    N = len(EX)

    def grp_of(e):
        if e["commit"] is not None:
            return "commit 있음"
        return "commit 없음(완주)" if e["fin"] else "commit 없음(절단)"

    GK = ["commit 있음", "commit 없음(완주)", "commit 없음(절단)"]
    print("=" * 104)
    print("commit 없는 예제에서의 행동 (학습 대상 밖 — 외삽 구간)")
    print("=" * 104)
    cnt = {k: sum(1 for e in EX if grp_of(e) == k) for k in GK}
    print("  " + "   ".join(f"{k} {cnt[k]}개" for k in GK))
    print()
    print("목표   집단                발화율   발화위치(진행률)  "
          "규칙 정확도  베이스   차이    절감률")
    print("-" * 104)
    for tgt in targets:
        got = {k: [] for k in GK}
        for fi in range(nfolds):
            tr = [i for i in range(N) if fold_of[i] != fi]
            te = [i for i in range(N) if fold_of[i] == fi]
            if not tr or not te:
                continue
            Str = [SC[fi][i] for i in tr]
            g = np.quantile(np.concatenate(Str), np.linspace(0.01, 0.999, 120))
            tau = pick_best([EX[i] for i in tr], tgt, list(g),
                            lambda E, c, S=Str: rule_stops(E, S, c))
            st = rule_stops([EX[i] for i in te], [SC[fi][i] for i in te], tau)
            for i, s in zip(te, st):
                got[grp_of(EX[i])].append((i, s))
        for k in GK:
            if not got[k]:
                continue
            sub = [EX[i] for i, _ in got[k]]
            sto = [s for _, s in got[k]]
            fired = [(e, s) for e, s in zip(sub, sto)
                     if s is not None and s < e["think_len"]]
            fr = len(fired) / len(sub)
            pos = (np.median([s / e["think_len"] for e, s in fired])
                   if fired else float("nan"))
            sv, ok = evaluate(sub, sto)
            _, bk = evaluate(sub, [None] * len(sub))
            print(f"{tgt*100:>4.0f}%  {k:<18} {fr*100:>6.1f}%   "
                  f"{pos*100:>10.1f}%      {ok.mean()*100:>7.1f}%  "
                  f"{bk.mean()*100:>6.1f}%  {(ok.mean()-bk.mean())*100:>+6.1f}%p  "
                  f"{sv*100:>6.1f}%")
        print("-" * 104)
    print("판정")
    print("  발화율이 세 집단에서 비슷하고 'commit 없음(완주)' 의 차이가 크게")
    print("  음수면 → 걱정대로다. 그 집단을 음성 예제로 넣어 다시 학습할 것")
    print("  (--neg-nocommit). 차이가 0 근처면 외삽이 해롭지 않다는 뜻이고,")
    print("  이미 사분위 표의 분모에 그 예제들이 들어가 있으므로 그 숫자는 그대로 쓴다")
    print()



# ── [2026-11-25] KL 검증기 팔 ─────────────────────────────────────────────
#   detector 가 발화한 probe 지점에서, 미리 재 둔 V-noise KL 이 '수치상 0' 이하면
#   닫고, 아니면 통과(계속 생성)해서 다음 발화를 기다린다. gold 검증기
#   (verifier_stops)와 구조가 같고 판정 기준만 gold 대신 KL 이다 — 배포 가능.
#   --kl-cell clean 이면 노이즈 없는 1 − p_top 으로 같은 판정을 한다(기준선).
def load_kl(path, cell):
    KL = {}
    for l in open(path):
        if not l.strip():
            continue
        r = json.loads(l)
        if cell == "clean":
            KL[r["ex_id"]] = {p["t"]: 1.0 - p["p_top"] for p in r["points"]}
        else:
            KL[r["ex_id"]] = {p["t"]: max(p["cells"][cell]["kl"], 0.0) for p in r["points"]}
    return KL


def kl_verifier_stops(EX, sc, tau, KL, thr, cost=15.8):
    """탈락 비용은 gold 검증기와 같은 상수(probe 1회)로 둬서 두 팔을 같은 조건에
    둔다. KL 이 없는 지점은 판정 불가 → 통과로 처리하고 개수를 센다."""
    stops, extra, miss = [], [], 0
    for e, s in zip(EX, sc):
        kmap = KL.get(e["ex_id"], {})
        seen, chosen, nrej = set(), None, 0
        for h in np.where(s > tau)[0]:
            t = snap_up(e["probes"], int(e["ts"][h]))
            if t is None or t in seen:
                continue
            seen.add(t)
            k = kmap.get(t)
            if k is None:
                miss += 1
            elif k <= thr:
                chosen = t
                break
            nrej += 1
        stops.append(chosen)
        extra.append(nrej * cost)
    kl_verifier_stops.miss = miss
    return stops, extra


def kl_diag(EX, SC, fold_of, nfolds, targets, KL, thr, label):
    """전 문제(사분위 없이) 기준: 규칙 단독 vs 규칙+KL 검증기 vs 규칙+gold 검증기.
    τ 는 팔마다 학습 폴드에서 따로 고른다(목표 절감률을 각자 맞추도록)."""
    N = len(EX)
    grp = lambda e: "A" if e["commit"] is not None else ("C" if e["fin"] else "B")
    print("=" * 104)
    print(f"[KL 검증기] {label}  thr={thr:g}  — 전 문제, 학습 폴드에서 팔마다 τ 따로")
    print("=" * 104)
    for tgt in targets:
        stR, stK, stV = [None] * N, [None] * N, [None] * N
        exK, exV = np.zeros(N), np.zeros(N)
        miss = 0
        for fi in range(nfolds):
            tr = [i for i in range(N) if fold_of[i] != fi]
            te = [i for i in range(N) if fold_of[i] == fi]
            if not tr or not te:
                continue
            Etr, Str = [EX[i] for i in tr], [SC[fi][i] for i in tr]
            Ete, Ste = [EX[i] for i in te], [SC[fi][i] for i in te]
            g = list(np.quantile(np.concatenate(Str), np.linspace(0.01, 0.999, 120)))
            tau = pick_best(Etr, tgt, g, lambda E, c, S=Str: rule_stops(E, S, c))
            tauk = pick_best(Etr, tgt, g, lambda E, c, S=Str: kl_verifier_stops(E, S, c, KL, thr))
            tauv = pick_best(Etr, tgt, g, lambda E, c, S=Str: verifier_stops(E, S, c))
            r = rule_stops(Ete, Ste, tau)
            k, ek = kl_verifier_stops(Ete, Ste, tauk, KL, thr); miss += kl_verifier_stops.miss
            v, ev = verifier_stops(Ete, Ste, tauv)
            for j, i in enumerate(te):
                stR[i], stK[i], stV[i] = r[j], k[j], v[j]
                exK[i], exV[i] = ek[j], ev[j]
        _, ob = evaluate(EX, [None] * N)
        sr, orr = evaluate(EX, stR)
        sk, ok = evaluate(EX, stK, exK)
        sv, ov = evaluate(EX, stV, exV)
        n01, n10 = int(np.sum(ok & ~orr)), int(np.sum(~ok & orr))
        print(f"\n목표 {tgt*100:.0f}%   (KL 이 없어 통과 처리한 발화 {miss}건)")
        print(f"  전체 n={N}  베이스 {ob.mean()*100:5.1f}%  │ 규칙 {orr.mean()*100:5.1f}% / 절감 {sr*100:5.1f}%"
              f"  │ +KL {ok.mean()*100:5.1f}% / {sk*100:5.1f}%  │ +gold {ov.mean()*100:5.1f}% / {sv*100:5.1f}%")
        print(f"  KL vs 규칙 (같은 문제 짝): KL만 맞힘 {n01} / 규칙만 맞힘 {n10}  부호검정 p={sign_p(n01, n10):.3g}")
        print(f"  {'집단':<4}{'n':>5}  {'베이스':>7}  {'규칙':>15}  {'+KL':>15}  {'+gold':>15}   망침(규칙→KL)  고침(규칙→KL)")
        for gname in ("A", "B", "C"):
            ix = np.array([i for i in range(N) if grp(EX[i]) == gname])
            if len(ix) == 0:
                continue
            sub = [EX[i] for i in ix]
            s1, o1 = evaluate(sub, [stR[i] for i in ix])
            s2, o2 = evaluate(sub, [stK[i] for i in ix], exK[ix])
            s3, o3 = evaluate(sub, [stV[i] for i in ix], exV[ix])
            b = ob[ix]
            print(f"  {gname:<4}{len(ix):>5}  {b.mean()*100:6.1f}%  {o1.mean()*100:6.1f}%/{s1*100:5.1f}%"
                  f"  {o2.mean()*100:6.1f}%/{s2*100:5.1f}%  {o3.mean()*100:6.1f}%/{s3*100:5.1f}%"
                  f"   {int(np.sum(~o1 & b)):>3} → {int(np.sum(~o2 & b)):<3}       {int(np.sum(o1 & ~b)):>3} → {int(np.sum(o2 & ~b)):<3}")
    print("\n읽는 법: 각 칸은 정확도/토큰 절감. 망침 = 베이스는 맞혔는데 그 팔이 틀린 문제 수.")
    print("  C 의 망침이 줄고 A·B 의 망침이 늘지 않으면, 같은 절감에서 KL 검증기가 손실을 막은 것.")
    print("  절감은 탈락 1건당 15.8토큰을 비용으로 넣은 값(gold 팔과 같은 조건).")
    print()

def spearman(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return np.nan
    a = np.argsort(np.argsort(x[m].astype(float)))
    b = np.argsort(np.argsort(y[m].astype(float)))
    return np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="sig_n500c.jsonl")
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--targets", default="0.3,0.5,0.7")
    ap.add_argument("--drop-extreme", type=float, default=10.0)
    ap.add_argument("--save-w", default=None)
    ap.add_argument("--early-t", type=int, default=256,
                    help="길이 예측에 쓸 조기 체크포인트의 토큰 위치")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="팔의 실제 절감률이 목표에서 이만큼 넘게 벗어나면 "
                         "요약 집계에서 제외한다 (짝비교가 불공정해지므로)")
    ap.add_argument("--neg-nocommit", action="store_true",
                    help="commit 없는 완주 예제를 전구간 y=0 음성으로 학습에 넣는다")
    ap.add_argument("--no-diag", action="store_true",
                    help="commit 없는 집단 진단표를 건너뛴다")
    ap.add_argument("--seed", type=int, default=0)
    # [v6 2026-11-18] probe_commit E1 에서 ridge+PCA256 이 Δ 0.098 → 0.139.
    # 원인은 PCA 자체가 아니라 행 단위 LOO 가 규제를 약하게 고르는 것으로 봄.
    # 배포 detector 에도 같은 처리를 걸어 손실이 줄어드는지 본다. 0 = v5 와 동일
    ap.add_argument("--pca", type=int, default=0,
                    help="detector 를 StandardScaler → PCA(N) → StandardScaler → "
                         "RidgeClassifierCV 로. 0=끔(v5 와 동일)")
    ap.add_argument("--kl", default=None,
                    help="vnoise_curve.jsonl — 주면 KL 검증기 팔 진단을 추가로 돈다")
    ap.add_argument("--kl-cell", default="18|2",
                    help="'18|2', 'all|0.5', 또는 'clean'(노이즈 없는 1 − p_top 기준선)")
    ap.add_argument("--kl-thr", type=float, default=None,
                    help="'수치상 0' 기준. 기본: KL 칸은 음수 KL |값|의 99%% 분위, clean 은 0")
    a = ap.parse_args()
    TARGETS = [float(x) for x in a.targets.split(",")]
    a_early = a.early_t

    print(f"[채점] {GRADER}")
    A, ex_all, tt_all = load_layer(a.out, a.layer)
    rows_j = [json.loads(l) for l in open(a.out)]
    C = commit_of(rows_j)
    byid = {r["ex_id"]: r for r in rows_j}

    keep = None
    if a.drop_extreme > 0:
        nrm = np.empty(len(A), np.float32)
        for s in range(0, len(A), 20000):
            e = min(s + 20000, len(A))
            b = A[s:e].astype(np.float32)
            nrm[s:e] = np.sqrt((b * b).sum(1))
        med = float(np.median(nrm))
        keep = nrm <= a.drop_extreme * med
        print(f"[극단값] 노름 > {a.drop_extreme:g}× 중앙({med:.1f}) 인 행 "
              f"{int((~keep).sum()):,} 제외")

    idx = {int(e): np.where(ex_all == e)[0] for e in np.unique(ex_all)}
    EX = []
    for eid, rs in idx.items():
        r = byid.get(eid)
        if r is None or not r.get("probes"):
            continue
        # 길이 예측용 조기 체크포인트. t≈256 — 256토큰 생성한 시점에서
        # "이 체인이 총 몇 토큰이 될까"를 묻는 배포 가능한 설정
        er = int(rs[int(np.argmin(np.abs(tt_all[rs] - a_early)))])
        EX.append(dict(ex_id=eid, rows=rs, ts=tt_all[rs], think_len=r["think_len"],
                       fin=bool(r.get("fin", True)), correct=bool(r["correct"]),
                       pred=r.get("pred"),
                       gold=r["gold"], probes=sorted(r["probes"], key=lambda p: p["t"]),
                       commit=C.get(eid), early_row=er))
    N = len(EX)
    print(f"[데이터] 예제 {N}개 "
          f"(commit 있는 {sum(1 for e in EX if e['commit'] is not None)}개로 학습), "
          f"절단 {sum(1 for e in EX if not e['fin'])}개")

    rg = np.random.default_rng(a.seed)
    rng2 = np.random.default_rng(a.seed + 1000)
    folds = np.array_split(rg.permutation(N), a.folds)
    fold_of = np.empty(N, int)
    for fi, te in enumerate(folds):
        fold_of[te] = fi
    SC = [[None] * N for _ in range(a.folds)]
    PL = [None] * a.folds          # PL[f][i] = w_{-f} 폴드의 길이 예측
    ws = []
    for fi, te in enumerate(folds):
        te_set = set(te.tolist())
        Xs, ys = [], []
        for i, e in enumerate(EX):
            if i in te_set or e["commit"] is None:
                continue
            j = int(np.searchsorted(e["ts"], e["commit"]))
            if j < a.K or j + a.K > len(e["rows"]):
                continue
            lo, hi = e["rows"][j - a.K], e["rows"][j + a.K - 1] + 1
            if hi - lo != 2 * a.K:
                continue
            blk = A[lo:hi].astype(np.float32)
            yy = np.r_[np.zeros(a.K), np.ones(a.K)]
            if keep is not None:
                m = keep[lo:hi]
                if len(np.unique(yy[m])) < 2:
                    continue
                blk, yy = blk[m], yy[m]
            Xs.append(blk); ys.append(yy)

        nneg = 0
        if a.neg_nocommit:
            # commit 없는 **완주** 예제는 어디서도 안 굳었다는 뜻이므로
            # 전 구간 y=0 이 올바른 라벨이다. 절단 예제는 최종답 자체가
            # 쓰레기여서 commit 이 '없음'이 아니라 '정의 안 됨'이라 뺀다.
            # 창의 중심은 양성 쪽 정착 상대위치를 빌려 잡는다 — 그래야
            # 음성이 체인 앞쪽에만 쏠려서 '늦은가' 를 배우는 걸 막는다
            rel = [np.searchsorted(e["ts"], e["commit"]) / max(len(e["rows"]), 1)
                   for i, e in enumerate(EX)
                   if i not in te_set and e["commit"] is not None]
            for i, e in enumerate(EX):
                if (i in te_set or e["commit"] is not None or not e["fin"]
                        or not rel):
                    continue
                j = int(rel[int(rng2.integers(len(rel)))] * len(e["rows"]))
                if j < a.K or j + a.K > len(e["rows"]):
                    continue
                lo, hi = e["rows"][j - a.K], e["rows"][j + a.K - 1] + 1
                if hi - lo != 2 * a.K:
                    continue
                blk = A[lo:hi].astype(np.float32)
                yy = np.zeros(2 * a.K)
                if keep is not None:
                    m = keep[lo:hi]
                    if m.sum() < 2:
                        continue
                    blk, yy = blk[m], yy[m]
                Xs.append(blk); ys.append(yy); nneg += len(yy)

        X, y = np.concatenate(Xs), np.concatenate(ys)
        sc = StandardScaler().fit(X)
        cw = "balanced" if nneg else None
        Z = sc.transform(X)
        pc = sc2 = None
        if a.pca:
            # [v6] 학습 폴드 행으로만 PCA 를 맞춘다(평가 예제 누수 없음)
            from sklearn.decomposition import PCA
            nc = int(min(a.pca, Z.shape[0] - 1, Z.shape[1]))
            pc = PCA(nc, random_state=a.seed).fit(Z)
            Z = pc.transform(Z)
            sc2 = StandardScaler().fit(Z)
            Z = sc2.transform(Z)

        def _pre(Xr):
            Zr = sc.transform(Xr)
            if pc is not None:
                Zr = sc2.transform(pc.transform(Zr))
            return Zr

        clf = RidgeClassifierCV(alphas=(1e1, 1e2, 1e3, 1e4, 1e5),
                                class_weight=cw).fit(Z, y)
        al = clf.alpha_
        edge = (" ** 가장 약한 끝값" if al == 1e1 else
                " ** 가장 강한 끝값" if al == 1e5 else "")
        print(f"  fold {fi}: 학습 {len(y):,}행"
              + (f" (commit 없는 완주 음성 {nneg:,}행 포함, "
                 f"y=1 비율 {y.mean()*100:.1f}%, class_weight=balanced)"
                 if nneg else "")
              + (f", PCA {pc.n_components_} (설명분산 "
                 f"{pc.explained_variance_ratio_.sum()*100:.1f}%)" if pc is not None else "")
              + f", alpha={al:g}{edge}")
        # w 를 표준화 공간(4096)으로 되돌려 저장 — PCA 를 써도 v5 와 같은 형식
        coef = clf.coef_.ravel()
        if pc is not None:
            w_std = (coef / sc2.scale_) @ pc.components_
        else:
            w_std = coef
        ws.append(dict(w_std=w_std.copy(), mean=sc.mean_.copy(),
                       scale=sc.scale_.copy(), w_h=(w_std/sc.scale_).copy()))
        for i, e in enumerate(EX):
            SC[fi][i] = clf.decision_function(
                _pre(A[e["rows"]].astype(np.float32)))

        # ── 길이 예측기도 같은 폴드에서 맞춘다 (조기 h → log think_len).
        #    로그를 쓰는 이유: think_len 이 668~8192 로 치우쳐 있어 선형으로
        #    맞추면 긴 쪽에 끌린다
        from sklearn.linear_model import RidgeCV
        tr_i = [i for i in range(N) if i not in te_set]
        Xl = A[[EX[i]["early_row"] for i in tr_i]].astype(np.float32)
        yl = np.log(np.array([EX[i]["think_len"] for i in tr_i], float))
        scl = StandardScaler().fit(Xl)
        rg_ = RidgeCV(alphas=(1e1, 1e2, 1e3, 1e4, 1e5)).fit(scl.transform(Xl), yl)
        Xa = A[[e["early_row"] for e in EX]].astype(np.float32)
        PL[fi] = np.exp(rg_.predict(scl.transform(Xa)))
        if fi == 0:
            rho_l = np.corrcoef(
                np.log([EX[i]["think_len"] for i in te]),
                np.log(PL[fi][te]))[0, 1]
            print(f"    길이 예측기: alpha={rg_.alpha_:g}, "
                  f"홀드아웃 corr(log 실제, log 예측) = {rho_l:+.3f}")

    if a.save_w:
        np.savez(a.save_w, layer=a.layer,
                 **{f"{k}_{fi}": v[k] for fi, v in enumerate(ws)
                    for k in ("w_std", "w_h", "mean", "scale")})
        print(f"  w 저장 → {a.save_w}")

    if not a.no_diag:
        nocommit_diag(EX, SC, fold_of, a.folds,
                      [float(x) for x in a.targets.split(",")])

    if a.kl:
        KLM = load_kl(a.kl, a.kl_cell)
        thr = a.kl_thr
        if thr is None:
            if a.kl_cell == "clean":
                thr = 0.0
            else:
                neg = [-p["cells"][a.kl_cell]["kl"] for l in open(a.kl) if l.strip()
                       for p in json.loads(l)["points"] if p["cells"][a.kl_cell]["kl"] < 0]
                thr = float(np.quantile(neg, 0.99)) if neg else 0.0
        have = sum(1 for e in EX if e["ex_id"] in KLM)
        print(f"[KL] {a.kl} 칸 {a.kl_cell}: 예제 {have}/{len(EX)} 에 KL 있음, thr={thr:.3g}")
        kl_diag(EX, SC, fold_of, a.folds, [float(x) for x in a.targets.split(",")],
                KLM, thr, a.kl_cell)

    _, base_ok = evaluate(EX, [None] * N)
    so, oo = evaluate(EX, [e["commit"] for e in EX])
    print(f"\n[전체] 베이스라인(캡 닫기) {base_ok.mean()*100:.1f}%   "
          f"오라클 절감 {so*100:.1f}% / {oo.mean()*100:.1f}%")
    # [v5] commit 답 = 최종 답이므로 정의상 같아야 한다. 어긋나면 채점 경로가 또 갈린 것
    print(f"       ↳ 오라클 정확도는 정의상 베이스와 같아야 한다 "
          f"(commit 답 = 최종 답). 차이 {(oo.mean()-base_ok.mean())*100:+.1f}%p")
    old_ok = np.array([e["correct"] if e["fin"] else ok_
                       for e, ok_ in zip(EX, base_ok)], bool)
    print(f"       ↳ 참고: 완주 예제를 저장된 correct(내장 eq_math)로 매기면 "
          f"베이스 {old_ok.mean()*100:.1f}%  (v4 까지의 값)")

    L = np.array([e["think_len"] for e in EX])
    qs = np.quantile(L, [0.25, 0.50, 0.75])
    qof = np.digitize(L, qs, right=True)
    TGRID = list(range(128, 8193, 128))
    RGRID = list(np.linspace(0.05, 0.99, 95))
    m_tests = 4 * len(TARGETS)
    print(f"\n셀 {m_tests}개 × 검정 4종(vs T, vs 오라클ρ, vs 예측ρ, 검증기). "
          f"계열별 Bonferroni {0.05/m_tests:.4f}\n")

    summary = []
    for q in range(4):
        sub_i = [i for i in range(N) if qof[i] == q]
        sub = [EX[i] for i in sub_i]
        sl = np.array([e["think_len"] for e in sub])
        _, bok = evaluate(sub, [None] * len(sub))
        sq, oq = evaluate(sub, [e["commit"] for e in sub])
        print("=" * 112)
        print(f"Q{q+1}  think_len {sl.min()}~{sl.max()} (중앙 {int(np.median(sl))}, "
              f"폭 {sl.max()/sl.min():.2f}배), {len(sub)}개 │ "
              f"베이스 {bok.mean()*100:.1f}% │ 오라클 {sq*100:.1f}%/{oq.mean()*100:.1f}%"
              + ("   ← 절단 지배" if q == 3 else ""))
        print("=" * 112)
        for tgt in TARGETS:
            okR, okT, okP, okQ, okV = [], [], [], [], []
            uR = uT = uP = uQ = uV = tot = 0
            taus, Ts, rhos, rhoqs = [], [], [], []
            stop_t, stop_L, nrej = [], [], []
            for fi in range(a.folds):
                tr = [i for i in sub_i if fold_of[i] != fi]
                te = [i for i in sub_i if fold_of[i] == fi]
                if len(tr) < 10 or not te:
                    continue
                EXtr = [EX[i] for i in tr]; SCtr = [SC[fi][i] for i in tr]
                EXte = [EX[i] for i in te]; SCte = [SC[fi][i] for i in te]
                for i in tr + te:                      # 폴드별 길이 예측 주입
                    EX[i]["pred_len"] = float(PL[fi][i])
                g = np.quantile(np.concatenate(SCtr), np.linspace(0.01, 0.999, 120))
                tau = pick_best(EXtr, tgt, list(g),
                                lambda E, c, S=SCtr: rule_stops(E, S, c))
                T = pick_best(EXtr, tgt,
                              [t for t in TGRID if t <= max(e["think_len"] for e in EXtr)],
                              lambda E, c: [c] * len(E))
                rho = pick_best(EXtr, tgt, RGRID, lambda E, c: rho_stops(E, c))
                rhoq = pick_best(EXtr, tgt, RGRID, lambda E, c: pred_stops(E, c))
                # 검증기는 자기 τ 를 따로 고른다. 같은 τ 를 쓰면 탈락분 때문에
                # 더 늦게 멈춰서 절감률이 안 맞고, '늦게 멈췄으니 정확' 이
                # 되어 비교가 성립하지 않는다
                tauv = pick_best(EXtr, tgt, list(g),
                                 lambda E, c, S=SCtr: verifier_stops(E, S, c))
                taus.append(tau); Ts.append(T); rhos.append(rho); rhoqs.append(rhoq)

                stR = rule_stops(EXte, SCte, tau)
                stV, exV = verifier_stops(EXte, SCte, tauv)
                s1, o1 = evaluate(EXte, stR)
                s2, o2 = evaluate(EXte, [T] * len(EXte))
                s3, o3 = evaluate(EXte, rho_stops(EXte, rho))
                s4, o4 = evaluate(EXte, pred_stops(EXte, rhoq))
                s5, o5 = evaluate(EXte, stV, exV)
                tl = sum(e["think_len"] for e in EXte)
                uR += (1-s1)*tl; uT += (1-s2)*tl; uP += (1-s3)*tl
                uQ += (1-s4)*tl; uV += (1-s5)*tl; tot += tl
                okR.append(o1); okT.append(o2); okP.append(o3)
                okQ.append(o4); okV.append(o5)
                nrej += exV
                for e, st in zip(EXte, stR):
                    if st is not None:
                        stop_t.append(st); stop_L.append(e["think_len"])
            if not okR:
                continue
            o1, o2 = np.concatenate(okR), np.concatenate(okT)
            o3, o4 = np.concatenate(okP), np.concatenate(okQ)
            o5 = np.concatenate(okV)
            sR, sT = 1 - uR/tot, 1 - uT/tot
            sP, sQ, sV = 1 - uP/tot, 1 - uQ/tot, 1 - uV/tot
            pair = lambda x, y: (int(np.sum(x & ~y)), int(np.sum(~x & y)))
            aT, aP, aQ, aV = (pair(o1, o2), pair(o1, o3),
                              pair(o1, o4), pair(o5, o1))
            pT, pP, pQ, pV = (sign_p(*aT), sign_p(*aP),
                              sign_p(*aQ), sign_p(*aV))
            bf = 0.05 / m_tests
            mk = lambda n, p: ("승" if n[0] > n[1] and p < bf else
                               "패" if n[1] > n[0] and p < bf else
                               "·승" if n[0] > n[1] and p < 0.05 else
                               "·패" if n[1] > n[0] and p < 0.05 else " ")
            rho_corr = spearman(np.array(stop_t, float), np.array(stop_L, float))
            print(f"{tgt*100:>4.0f}%  규칙 {sR*100:>5.1f}%/{o1.mean()*100:>5.1f}%"
                  f" │ T({int(np.mean(Ts)):>4}) {sT*100:>5.1f}%/{o2.mean()*100:>5.1f}%"
                  f"  오라클ρ {sP*100:>5.1f}%/{o3.mean()*100:>5.1f}%"
                  f"  예측ρ {sQ*100:>5.1f}%/{o4.mean()*100:>5.1f}%"
                  f"  +검증기 {sV*100:>5.1f}%/{o5.mean()*100:>5.1f}%")
            print(f"       vsT {aT[0]:>3}:{aT[1]:<3}p={pT:<6.4f}{mk(aT,pT)}"
                  f" │ vs오라클ρ {aP[0]:>3}:{aP[1]:<3}p={pP:<6.4f}{mk(aP,pP)}"
                  f" │ vs예측ρ {aQ[0]:>3}:{aQ[1]:<3}p={pQ:<6.4f}{mk(aQ,pQ)}"
                  f" │ 검증기이득 {aV[0]:>3}:{aV[1]:<3}p={pV:<6.4f}{mk(aV,pV)}"
                  f" │ ρ_len={rho_corr:+.2f}")
            # 팔이 목표 절감률을 못 맞추면 짝비교가 불공정해진다. 절감률이
            # 다른 두 팔을 비교하면 '덜 잘라서 정확' 또는 '더 잘라서 부정확'
            # 이 승패로 둔갑한다. 요약에서 그런 셀을 빼기 위해 어긋남을 기록
            off = dict(T=abs(sT - tgt), P=abs(sP - tgt),
                       Q=abs(sQ - tgt), V=abs(sV - tgt), R=abs(sR - tgt))
            summary.append((q + 1, tgt, aT, pT, aP, pP, rho_corr,
                            o1.mean() - o2.mean(), o1.mean() - o3.mean(),
                            aQ, pQ, o1.mean() - o4.mean(),
                            aV, pV, o5.mean() - o1.mean(), off))
        print()

    print("=" * 112)
    print("요약")
    print("=" * 112)
    bf = 0.05 / m_tests
    TOL = a.tol
    print(f"  (절감률 어긋남 > {TOL*100:.0f}%p 인 셀은 괄호로 따로 센다 — "
          f"절감률이 다른 두 팔의 승패는 해석 불가)\n")
    for name, gi, pi, di, ky in (("vs fixed_T", 2, 3, 7, "T"),
                                 ("vs 오라클길이×ρ", 4, 5, 8, "P"),
                                 ("vs 예측길이×ρ", 9, 10, 11, "Q"),
                                 ("검증기 추가이득", 12, 13, 14, "V")):
        fair = [s for s in summary
                if s[15][ky] <= TOL and s[15]["R"] <= TOL]
        unfair = len(summary) - len(fair)
        w = sum(1 for s in fair if s[gi][0] > s[gi][1] and s[pi] < bf)
        l = sum(1 for s in fair if s[gi][1] > s[gi][0] and s[pi] < bf)
        sg = sum(1 for s in fair if s[gi][0] > s[gi][1])
        s3 = [s for s in fair if s[0] <= 3]
        d3 = (f"{np.mean([x[di] for x in s3])*100:+.1f}%p" if s3 else "n/a")
        print(f"  {name:<16} Bonferroni 통과 승 {w} / 패 {l} / 무 {len(fair)-w-l}"
              f"   부호 우세 {sg}/{len(fair)}"
              f"   Q1~Q3 정확도차 평균 {d3}"
              + (f"   (절감률 어긋남으로 제외 {unfair}셀)" if unfair else ""))
    cc = [s[6] for s in summary if np.isfinite(s[6])]
    print(f"\n  정지지점–길이 Spearman: 중앙 {np.median(cc):+.2f}  "
          f"(fixed_T 는 0, 오라클ρ 는 +1)")
    print("\n판정")
    print("  [1] vs 예측ρ 가 결정적이다. 여기서 이기면 readout 은 길이 예측기가 아니다.")
    print("      무승부면 '길이를 h 에서 읽고 비례로 자른 것'이 전부라는 뜻이다.")
    print("  [2] vs 오라클ρ 는 배포 불가 팔이다. 이기라고 두는 게 아니라")
    print("      '길이만으로 설명되는가'의 상한을 재는 장치다.")
    print("  [3] 검증기 팔은 가역 정지의 상한이다. 추가이득이 0 이면")
    print("      detector 가 이미 충분해 섭동/검증 축의 Amdahl 상한이 없다.")


if __name__ == "__main__":
    main()