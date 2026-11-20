#!/usr/bin/env python3
"""attn_pt_heads.py — commit 지점 키를 더 보는 '헤드'가 따로 있는가 (절반 선택 → 절반 검정).

배경
    헤드 평균에서는 commit 키의 이웃 대비 분위(pct)가 0.50 (A, n=287). 특정 receiver
    head 만 보고 있다면 평균에서 묻힌다(Thought Anchors). 헤드 36×32 = 1,152개 × 거리 6
    을 한 데이터에서 훑고 최고를 고르면 우연을 고르게 되므로 고르는 데이터와 재는 데이터를 나눈다.

절차
    1. A 예제를 무작위로 반반(sel / test).
    2. sel 에서 (층, 헤드, 거리)마다 평균 pct 의 z (귀무: 예제마다 키 수에 맞는 이산 균등분포).
       단측(>0.5) p 에 Benjamini–Hochberg FDR q 를 걸어 통과한 칸 = 선택 집합 S.
       선택 기준이 통계 절차이지 '상위 k개' 같은 임의 숫자가 아니다.
    3. test 에서 S 의 평균 pct (예제마다 S 평균 → 예제 평균). 귀무 = 예제마다 commit 키 대신
       아무 키나(S 전체에 같은 키) → 순열 p. 헤드 간 상관은 보존된다.
    4. 같은 S 를 B(가짜 commit)에 적용 — 위치 성질이면 B 도 같이 올라간다. 0.5 여야 한다.
    5. sel/test 를 바꿔서 한 번 더(두 방향 다 통과해야 재현).
    S 가 비면 그 자체가 결과다: 'FDR q 에서 commit 키를 더 보는 헤드가 없다'.

사용
    python -u src/attn_pt_heads.py results/attn/attn_pt_heads.npz [--q 0.05] [--seed 0]
"""
import argparse
import json
import math

import numpy as np


def pct_matrix(VH):
    """VH [L, H, nk, nd] → 키마다 이웃 대비 대조 s 의 체인 내 분위 [L, H, nk, nd] (끝 키·nan 은 nan)
    와 키마다 유효 키 수 m [L, H, nd]."""
    lv = np.log(VH.astype(np.float64))
    s = np.full(lv.shape, np.nan)
    s[:, :, 1:-1] = lv[:, :, 1:-1] - 0.5 * (lv[:, :, :-2] + lv[:, :, 2:])
    ok = np.isfinite(s)
    m = ok.sum(2)                                                   # [L, H, nd]
    sv = np.where(ok, s, np.inf)
    less = (sv[:, :, None, :, :] < sv[:, :, :, None, :]).sum(3)     # [L,H,nk,nd] 나보다 작은 수
    eq = (sv[:, :, None, :, :] == sv[:, :, :, None, :]).sum(3) - 1
    pct = (less + 0.5 * eq) / np.maximum(m[:, :, None, :] - 1, 1)
    pct[~ok] = np.nan
    pct[np.broadcast_to((m < 3)[:, :, None, :], pct.shape)] = np.nan
    return pct, m


def load(path):
    z = np.load(path, allow_pickle=False)
    summ = json.loads(str(z["summary"]))
    if f"ex{summ[0]['ex_id']}_VH" not in z.files:
        raise SystemExit("VH 가 없다 — attn_commitpt.py 를 --save-heads 로 다시 돌릴 것")
    ex = []
    for x in summ:
        VH = z[f"ex{x['ex_id']}_VH"]
        ci = x["n_pre"]
        if not (0 < ci < VH.shape[2] - 1):
            continue
        pct, m = pct_matrix(VH)
        ex.append(dict(g=x["group"], id=x["ex_id"], ci=ci, pct=pct, m=m))
    return ex, z["dists"].tolist()


def sel_z(E):
    """예제들의 commit pct 평균 → (층, 헤드, 거리)별 z. 귀무 분산 = 이산 균등 (m+1)/(12(m-1))."""
    P = np.stack([e["pct"][:, :, e["ci"], :] for e in E])          # [n, L, H, nd]
    M = np.stack([e["m"] for e in E]).astype(np.float64)
    var = (M + 1) / (12 * np.maximum(M - 1, 1))
    ok = np.isfinite(P)
    n = ok.sum(0)
    mean = np.nansum(P, 0) / np.maximum(n, 1)
    se = np.sqrt(np.where(ok, var, 0).sum(0)) / np.maximum(n, 1)
    zz = (mean - 0.5) / np.where(se > 0, se, np.nan)
    return mean, zz, n


def bh(p, q):
    """Benjamini–Hochberg. 통과 마스크."""
    flat = p.ravel()
    o = np.argsort(flat)
    k = len(flat)
    thr = q * np.arange(1, k + 1) / k
    passed = flat[o] <= thr
    keep = np.zeros(k, bool)
    if passed.any():
        keep[o[:np.max(np.flatnonzero(passed)) + 1]] = True
    return keep.reshape(p.shape)


def test_set(E, S, nperm, rng):
    """S(마스크 [L,H,nd]) 의 평균 pct 를 예제별로 → 평균, 순열 p(단측 >0.5)."""
    obs, pools = [], []
    for e in E:
        v = e["pct"].transpose(0, 1, 3, 2)[S]                                           # [|S|, nk]  (S 칸 × 키)
        perkey = np.nanmean(np.where(np.isfinite(v), v, np.nan), 0) if v.size else np.array([])
        if not np.isfinite(perkey[e["ci"]]):
            continue
        obs.append(perkey[e["ci"]])
        pools.append(perkey[np.isfinite(perkey)])
    if not obs:
        return np.nan, np.nan, 0
    o = float(np.mean(obs))
    null = np.mean([pool[rng.integers(len(pool), size=nperm)] for pool in pools], 0)
    p = (1 + np.sum(null >= o)) / (nperm + 1)
    return o, p, len(obs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--q", type=float, default=0.05, help="BH FDR 수준")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nperm", type=int, default=10000)
    a = ap.parse_args()
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    ex, D = load(a.npz)
    A = [e for e in ex if e["g"] == "A"]
    B = [e for e in ex if e["g"] == "B"]
    L, H = A[0]["pct"].shape[:2]
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(A))
    halves = [[A[i] for i in perm[: len(A) // 2]], [A[i] for i in perm[len(A) // 2:]]]
    print(f"A {len(A)}개 → 반반 {len(halves[0])}/{len(halves[1])}, B {len(B)}개 | "
          f"후보 {L}층 × {H}헤드 × 거리 {len(D)} = {L*H*len(D):,}칸, BH q={a.q}")

    # 헤드 평균에서 했던 것과 같은 크기의 전체 그림: A 전체 z 분포
    mean_all, z_all, _ = sel_z(A)
    print(f"\n[참고] A 전체 칸별 z: 중앙 {np.nanmedian(z_all):+.2f}, "
          f"|z|>3.0 인 칸 {int(np.nansum(np.abs(z_all) > 3))}개 "
          f"(귀무에서 기대 ~{L*H*len(D)*0.0027:.0f}개), 최대 {np.nanmax(z_all):+.2f}")

    for r, (sel, tst) in enumerate([(halves[0], halves[1]), (halves[1], halves[0])]):
        mean, zz, n = sel_z(sel)
        # 단측 p (commit 키를 더 본다 = z > 0)
        p1 = np.vectorize(lambda v: 0.5 * math.erfc(v / math.sqrt(2))
                          if np.isfinite(v) else 1.0)(zz)
        S = bh(p1, a.q)
        Sneg = bh(1 - p1, a.q)
        print(f"\n── 방향 {r+1}: 선택 {len(sel)}개 → 검정 {len(tst)}개")
        print(f"  선택 단계 BH 통과(commit 을 더 봄) {int(S.sum())}칸 / "
              f"(반대로 덜 봄) {int(Sneg.sum())}칸")
        if S.any():
            li, hi, di = np.nonzero(S)
            top = np.argsort(-zz[S])[:10]
            print("  선택된 칸(상위 10): " + ", ".join(
                f"L{li[k]}H{hi[k]}d{D[di[k]]}(z{zz[S][k]:+.1f})" for k in top))
            print(f"  거리별 칸 수: " + " ".join(f"d{d}:{int(S[:, :, i].sum())}"
                                              for i, d in enumerate(D)))
            o, p, k = test_set(tst, S, a.nperm, rng)
            ob, pb, kb = test_set(B, S, a.nperm, rng)
            print(f"  검정(A 나머지 반): 평균 pct {o:.3f} (n={k}), 순열 p {p:.4f}")
            print(f"  대조(B 가짜 commit): 평균 pct {ob:.3f} (n={kb}), 순열 p {pb:.4f}")
        else:
            print("  → 선택 집합이 비었다: 이 절반에서 FDR 을 통과하는 '더 보는 헤드' 없음")
        # 탐색용: 선택 절반 최고 칸 1개를 그대로 검정 (임의 선택 보정 없음 — 참고만)
        k = np.nanargmax(zz)
        best = np.zeros_like(S); best.flat[k] = True
        li, hi, di = np.unravel_index(k, zz.shape)
        o, p, kk = test_set(tst, best, a.nperm, rng)
        print(f"  [참고] 선택 절반 최고 칸 L{li}H{hi}d{D[di]} (z {zz.flat[k]:+.2f}) → "
              f"검정 절반 pct {o:.3f}, p {p:.4f}")

    print("\n판정: 두 방향 모두 '검정 p < 0.05 이고 B 는 0.5 근처'면 commit 키를 보는 헤드가 재현된다.")
    print("      선택이 비거나 검정에서 0.5 로 돌아가면 헤드 단위에서도 없다(이 좌표계에서).")


if __name__ == "__main__":
    main()