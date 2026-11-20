#!/usr/bin/env python3
"""attn_pt_analyze.py — attn_commitpt.py 결과(npz)를 '위치 추세를 지운' 지표로 다시 판정.

왜
    파일럿에서 commit vs pre 는 양수, commit vs post 는 음수가 A·B·C 전부에 나왔다.
    가짜 commit 인 B·C 에서도 같으므로 commit 효과가 아니라 '뒤에 있는 probe 키일수록
    값이 크다'는 위치 추세다. pre(앞)·post(뒤) 한쪽만 대조로 쓰면 그 추세가 그대로 남는다.

지표 — 이웃 대비 대조(local contrast)
    s_j = log V_j − ½(log V_{j−1} + log V_{j+1})       (j 는 체인 안 probe 키 순서)
    양옆 키와 비교하므로 키 위치에 대해 선형인 추세는 지워진다.
    commit 키의 s 가 그 체인의 모든 키 s 중 몇 분위인가 = pct (0~1, 귀무에서 평균 0.5).

귀무
    같은 예제 안에서 commit 키 대신 아무 키나 하나 고른다(예제마다 독립) → 평균 pct 의
    순열 분포. 위치·체인 구조는 그대로 두고 'commit 이라는 표지'만 깬다.
    순열 횟수는 비교 수(층 묶음 × 거리)에 대해 Bonferroni 가 성립하도록 자동으로 맞춘다.

사용
    python -u src/attn_pt_analyze.py results/attn/attn_pt_pilot.npz
"""
import json
import sys

import numpy as np


def main():
    path = sys.argv[1]
    alpha = 0.05
    z = np.load(path, allow_pickle=False)
    summ = json.loads(str(z["summary"]))
    D = z["dists"].tolist()
    NL = z[f"ex{summ[0]['ex_id']}_V"].shape[0]
    bands = [(0, NL // 3), (NL // 3, 2 * NL // 3), (2 * NL // 3, NL)]
    m = len(bands) * len(D)
    nperm = int(np.ceil(m / alpha)) * 10            # p 하한 1/(N+1) 이 alpha/m 보다 충분히 작게
    rng = np.random.default_rng(0)
    print(f"비교 {m}개 (층 묶음 {len(bands)} × 거리 {len(D)}), Bonferroni {alpha/m:.1e}, "
          f"순열 {nperm:,}회 (p 하한 {1/(nperm+1):.1e})")

    groups = sorted({x["group"] for x in summ})
    for g in groups:
        G = [x for x in summ if x["group"] == g]
        print(f"\n[{g}] 예제 {len(G)}개 — 칸 = 평균 pct (예제 수, 순열 p)   귀무 0.50")
        print("  " + f"{'':<8}" + "".join(f"{'d='+str(d):>20}" for d in D))
        for bi, (lo, hi) in enumerate(bands):
            cells = []
            for di in range(len(D)):
                pcts, pools = [], []
                for x in G:
                    V = z[f"ex{x['ex_id']}_V"][lo:hi, :, di].mean(0)     # [nk]
                    lv = np.log(V)
                    ci = x["n_pre"]
                    s = np.full(len(lv), np.nan)
                    s[1:-1] = lv[1:-1] - 0.5 * (lv[:-2] + lv[2:])
                    ok = np.isfinite(s)
                    if not (0 < ci < len(lv) - 1) or not ok[ci] or ok.sum() < 3:
                        continue
                    sv = s[ok]
                    rank = (sv < s[ci]).sum() + 0.5 * ((sv == s[ci]).sum() - 1)
                    pcts.append(rank / (len(sv) - 1))
                    # 귀무용: 이 예제의 모든 키의 pct
                    order = sv.argsort().argsort()
                    pools.append(order / (len(sv) - 1))
                if not pcts:
                    cells.append(f"{'-':>20}"); continue
                obs = float(np.mean(pcts))
                null = np.mean([pool[rng.integers(len(pool), size=nperm)] for pool in pools], 0)
                p = (1 + np.sum(np.abs(null - 0.5) >= abs(obs - 0.5))) / (nperm + 1)
                mark = "*" if p < alpha / m else " "
                cells.append(f"{obs:.2f} ({len(pcts)}, {p:.4f}){mark}".rjust(20))
            print(f"  {f'L{lo}-{hi-1}':<8}" + "".join(cells))
    print("\n* = Bonferroni 통과. B·C 는 가짜 commit 이라 0.50 근처여야 한다.")
    print("A 만 0.5 위로 일관되면 commit 지점 키가 양옆 probe 지점보다 더 attend 된다는 뜻.")


if __name__ == "__main__":
    main()