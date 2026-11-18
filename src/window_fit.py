#!/usr/bin/env python3
"""window_fit.py — ±K 창이 몇 예제에 들어가나. 덤프(.h.f16) 안 읽는다.

commit 이 평균 98토큰 앞당겨졌으므로 체인 앞쪽에서 ±K 가 안 들어가는 예제가
늘어난다. probe_commit.py 를 5GB 읽고 나서 "쓸 수 있는 예제 0건"을 보는 대신
여기서 먼저 센다. .meta.jsonl 만 읽으므로 초 단위.

K 를 결과 보고 고르면 선택편향이다. 그래서 **데이터와 무관한 기준**으로
미리 정한다: 사용 가능 예제가 commit 있는 예제의 90% 이상인 가장 큰 K.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from grade2 import eq_math2 as _eq
    GRADER = "grade2.eq_math2"
except Exception:
    def _eq(a, b):
        return str(a).strip() == str(b).strip()
    GRADER = "문자열 비교 (grade2 없음 — 참고용)"


def commit_of(r, arm="A"):
    pa = r.get("probes_arms")
    ps = (pa or {}).get(arm) or (r.get("probes") if arm == "A" else None)
    if not ps or r.get("pred") is None:
        return None
    ps = sorted(ps, key=lambda p: p["t"])
    ok = [p.get("ans") is not None and _eq(p["ans"], r["pred"]) for p in ps]
    c = None
    for i in range(len(ps) - 1, -1, -1):
        if not ok[i]:
            break
        c = ps[i]["t"]
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="sig_para500.jsonl")
    ap.add_argument("--arm", default="A")
    ap.add_argument("--ks", default="4,6,8,10,12,16,20")
    ap.add_argument("--frac", type=float, default=0.90)
    a = ap.parse_args()
    print(f"[채점] {GRADER}")

    meta = [json.loads(l) for l in open(a.out + ".meta.jsonl") if l.strip()]
    ex = np.array([m["ex_id"] for m in meta])
    tt = np.array([m["t"] for m in meta])
    rows = {r["ex_id"]: r for r in map(json.loads, open(a.out))}
    idx = {int(e): np.where(ex == e)[0] for e in np.unique(ex)}

    C = {e: commit_of(r, a.arm) for e, r in rows.items()}
    have = [e for e, c in C.items() if c is not None and e in idx]
    gaps = []
    for e in have:
        d = np.diff(tt[idx[e]])
        if len(d):
            gaps.append(np.median(d))
    print(f"[덤프] 행 {len(meta):,}  예제 {len(idx)}  "
          f"예제당 중앙 {np.median([len(v) for v in idx.values()]):.0f}행")
    print(f"       행 간격 중앙 {np.median(gaps):.1f}토큰  "
          f"→ ±K 창의 폭 ≈ 2K × {np.median(gaps):.0f}토큰")
    print(f"[라벨] commit 있는 예제 {len(have)} / {len(rows)}\n")

    print("  K   들어감   비율    창 폭(토큰)   걸린 이유: 앞부족 / 뒤부족")
    print("-" * 68)
    best = None
    for K in [int(x) for x in a.ks.split(",")]:
        okn = lo_bad = hi_bad = 0
        for e in have:
            rs = idx[e]
            ts = tt[rs]
            j = int(np.searchsorted(ts, C[e]))
            if j < K:
                lo_bad += 1
            elif j + K > len(rs):
                hi_bad += 1
            else:
                lo, hi = rs[j - K], rs[j + K - 1] + 1
                if hi - lo == 2 * K:          # 연속성(덤프가 비연속이면 탈락)
                    okn += 1
                else:
                    hi_bad += 1
        fr = okn / max(len(have), 1)
        mark = ""
        if fr >= a.frac:
            best = K
            mark = "  ←"
        print(f"{K:>3}  {okn:>6}  {fr*100:>5.1f}%   {2*K*np.median(gaps):>9.0f}"
              f"      {lo_bad:>4} / {hi_bad:<4}{mark}")
    print("-" * 68)
    if best:
        print(f"\n사전등록 기준({a.frac*100:.0f}% 이상)을 만족하는 가장 큰 K = {best}")
        print(f"  python probe_commit.py {a.out} --layers 18 27 36 "
              f"--K {best} --nnull 100 --null both --drop-extreme 10")
    else:
        print(f"\n어떤 K 도 {a.frac*100:.0f}% 를 못 넘는다. 창이 아니라 덤프 격자가")
        print("  문제일 수 있다 — 행 간격 중앙과 commit 중앙을 같이 볼 것")
    print("\n주의: K 를 키우면 창이 commit 에서 멀어져 '정착 직전/직후'가 아니라")
    print("'체인 전반/후반'을 묻게 된다. 큰 K 가 좋은 게 아니다 — 들어가는")
    print("한도 안에서 선행연구와 같은 값(K=12)을 쓰는 게 비교에 낫다")


if __name__ == "__main__":
    main()