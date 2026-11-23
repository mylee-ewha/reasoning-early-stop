#!/usr/bin/env python3
"""vnoise_sep.py — V 노이즈 반응이 **예제 하나 단위로** commit 을 가를 수 있나 (CPU, 몇 초).

왜:
  지금까지의 결과(짝지은 차이차, 부호검정)는 **집단 수준**이다 — "142개 중 53:14 로
  commit 을 건너면 KL 이 더 떨어진다". 배포 검증기는 예제 하나를 보고 수락/탈락을
  정해야 하므로, 예제 단위 구별력(AUC)이 0.5 근처면 배포 실험은 의미가 없다.
  프로토콜 1번(상한 먼저)에 해당한다.

입력: vnoise_pretest.py 의 저장 json (예: results/vnoise/vnoise_v3.json)
  예제마다 pre(commit 직전 probe), post(commit probe), c0·c1(commit 이전 인접 쌍)의
  (섭동군, c) 별 kl / qa(=clean top 토큰의 noisy 확률) / flip.

지표 (섭동군·c 마다):
  [비] 검증기가 '인접 두 지점의 KL 변화'를 본다고 할 때.
       d_cross = log KL(post) − log KL(pre)    (commit 을 건너는 쌍)
       d_ctrl  = log KL(c1)   − log KL(c0)     (둘 다 commit 이전인 쌍)
       AUC_비 = P(d_cross < d_ctrl)  — 건너는 쌍이 더 떨어지면 1
       짝 = 같은 예제 안에서 d_cross < d_ctrl 인 비율 (예전 부호검정과 같은 것)
  [수준] 검증기가 '한 지점의 KL 크기'만 본다고 할 때.
       AUC_수준 = P(KL(post) < KL(pre)) — 예제를 섞어서, 굳은 지점이 더 낮으면 1
  감도 선별: pre·c0 의 qa < 0.99 인 예제만 (반응이 바닥인 줄은 버린다)
  95% CI: 예제 부트스트랩 2,000회

판정 (사전 고정):
  AUC_비 의 95% CI 하한이 0.5 를 넘는 칸이 있어야 배포 실험(1문단 늦게 닫기)을 할
  이유가 생긴다. 전부 0.5 를 걸치면 v-noise 는 집단 수준 메커니즘 증거로만 쓴다.

사용:
  python -u src/vnoise_sep.py results/vnoise/vnoise_v3.json | tee results/vnoise/vnoise_sep.log
"""
import json
import sys

import numpy as np

EPS = 1e-9


def auc(neg, pos):
    """P(pos > neg) + 0.5·P(같음). Mann-Whitney 로."""
    neg, pos = np.asarray(neg), np.asarray(pos)
    if len(neg) == 0 or len(pos) == 0:
        return np.nan
    allv = np.concatenate([neg, pos])
    r = allv.argsort().argsort().astype(float)
    # 같은 값 처리: 평균 순위
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt) - cnt
    r = (csum + (cnt - 1) / 2.0)[inv]
    rp = r[len(neg):].sum()
    return (rp - len(pos) * (len(pos) - 1) / 2.0) / (len(pos) * len(neg))


def main():
    path = sys.argv[1]
    rec = json.load(open(path))
    keys = sorted({k for r in rec for k in r["kl"]})
    cells = sorted({(k.split("|")[1], float(k.split("|")[2])) for k in keys},
                   key=lambda x: (x[0] != "18", x[0], x[1]))
    rng = np.random.default_rng(0)
    print(f"예제 {len(rec)}개, 칸 {len(cells)}개 (섭동군 × c)\n")
    print("감도 선별: pre·c0 의 q(top) < 0.99 인 예제만.  ↑ 는 '건너는 쪽이 더 낮다' = 1 이 이상적\n")
    hdr = (f"{'섭동군':>6} {'c':>5} │ {'n(비)':>5} {'짝':>6} {'AUC_비':>7} {'[95% CI]':>15} │ "
           f"{'n(수준)':>6} {'AUC_수준':>8} {'[95% CI]':>15}")
    print(hdr)
    print("─" * len(hdr))
    for g, c in cells:
        kp, kq, k0, k1 = (f"{t}|{g}|{c:g}" for t in ("pre", "post", "c0", "c1"))
        # 키 포맷이 '0.25' / '1.0' 등 그대로라서 원 문자열로 다시 찾는다
        def key(tag):
            for k in keys:
                a, b, cc = k.split("|")
                if a == tag and b == g and float(cc) == c:
                    return k
            return None
        kp, kq, k0, k1 = key("pre"), key("post"), key("c0"), key("c1")
        dc, dn, lv_pre, lv_post = [], [], [], []
        for r in rec:
            K = r["kl"]
            if kp in K and kq in K and K[kp]["qa"] < 0.99:
                lv_pre.append(np.log(max(K[kp]["kl"], 0.0) + EPS))
                lv_post.append(np.log(max(K[kq]["kl"], 0.0) + EPS))
                if k0 in K and k1 in K and K[k0]["qa"] < 0.99:
                    dc.append(np.log(max(K[kq]["kl"], 0.0) + EPS) - np.log(max(K[kp]["kl"], 0.0) + EPS))
                    dn.append(np.log(max(K[k1]["kl"], 0.0) + EPS) - np.log(max(K[k0]["kl"], 0.0) + EPS))
        dc, dn = np.array(dc), np.array(dn)
        lv_pre, lv_post = np.array(lv_pre), np.array(lv_post)

        def ci(f, *arrs):
            n = len(arrs[0])
            bs = []
            for _ in range(2000):
                ix = rng.integers(n, size=n)
                bs.append(f(*[a_[ix] for a_ in arrs]))
            return np.nanpercentile(bs, [2.5, 97.5])

        if len(dc) >= 10:
            a_r = auc(-dn, -dc)                       # 건너는 쪽이 더 떨어지면 1
            lo, hi = ci(lambda x, y: auc(-y, -x), dc, dn)
            pair = float((dc < dn).mean())
            part1 = f"{len(dc):>5} {pair:>6.2f} {a_r:>7.3f} [{lo:.3f}, {hi:.3f}]"
            mark = " ★" if lo > 0.5 else ""
        else:
            part1, mark = f"{len(dc):>5} {'-':>6} {'-':>7} {'':>15}", ""
        if len(lv_pre) >= 10:
            a_l = auc(-lv_pre, -lv_post)              # post 가 더 낮으면 1
            lo2, hi2 = ci(lambda x, y: auc(-x, -y), lv_pre, lv_post)
            part2 = f"{len(lv_pre):>6} {a_l:>8.3f} [{lo2:.3f}, {hi2:.3f}]"
        else:
            part2 = f"{len(lv_pre):>6} {'-':>8}"
        print(f"{g:>6} {c:>5g} │ {part1} │ {part2}{mark}")

    print("\n읽는 법")
    print("  AUC_비   : 인접 두 지점의 KL 변화로 '이 쌍이 commit 을 건넜나'를 가르는 힘."
          " 1문단 늦게 닫기·직전 지점 비교 검증기가 쓰는 정보")
    print("  AUC_수준 : 한 지점의 KL 크기만으로 '굳었나'를 가르는 힘 (예제 사이 KL 크기 차이가 섞임)")
    print("  짝       : 같은 예제 안에서 건너는 쪽이 더 떨어진 비율 — 예전 부호검정과 같은 양")
    print("  ★        : AUC_비 95% CI 하한 > 0.5. 하나도 없으면 배포 실험은 접는다")
    print("  주의     : 통제쌍(c0,c1) 간격과 (pre,post) 간격이 같다는 보장이 없다"
          " (v3 레코드에 토큰 위치가 없음) — vnoise_curve.py 로 해소")


if __name__ == "__main__":
    main()