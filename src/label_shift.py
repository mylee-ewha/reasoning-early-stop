#!/usr/bin/env python3
"""label_shift.py — 구 라벨(256토큰 격자) vs 신 라벨(문단 정렬) 비교.

재수집에서 바뀐 것은 **probe 를 찍는 지점**뿐이다. 생성 토큰열은 같다
(같은 seed·greedy, 로그로 확인됨). 그래서 commit_step 이 움직였다면
그건 모델이 달라진 게 아니고 "격자가 달라져서 측정값이 달라진 것"이다.

이 스크립트가 답하는 것:
  (1) 라벨이 얼마나, 어느 쪽으로 움직였나  — probe_commit 의 z 를
      새 데이터에서 재현해도 되는지의 전제
  (2) 움직임이 think_len / commit_frac 과 상관있나 — 상관이 있으면
      라벨 잡음이 길이와 얽혀서 길이 교란이 되살아난다
  (3) 팔(A/B/C/D) 사이 일치율 — 라벨 자체의 측정 오차 하한

덤프(.h.f16) 불필요. 초 단위로 끝난다.

사용:
  python label_shift.py --old out/sig_n500c.jsonl --new out/sig_para500.jsonl
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


def commit_of(probes, final):
    """probes 중 '이후 전부 final 과 같아지는' 가장 이른 t.

    final 은 gold 가 아니라 **모델 자신의 최종 답**이다. gold 로 잡으면
    틀린 예제가 전부 None 이 되고 샘플이 쉬운 문제로 쏠린다.
    """
    if final is None or not probes:
        return None
    ps = sorted(probes, key=lambda p: p["t"])
    ans = [p.get("ans") for p in ps]
    ok = [a is not None and _eq(a, final) for a in ans]
    c = None
    for i in range(len(ps) - 1, -1, -1):
        if not ok[i]:
            break
        c = ps[i]["t"]
    return c


def arm_probes(r, arm):
    """arm='A' 는 구 포맷의 r['probes'] 와 같다."""
    pa = r.get("probes_arms")
    if pa and arm in pa:
        return pa[arm]
    return r.get("probes") if arm == "A" else None


def summarize(name, rows, arm="A"):
    out = {}
    for r in rows:
        ps = arm_probes(r, arm)
        if not ps:
            continue
        c = commit_of(ps, r.get("pred"))
        c_in = (r.get("feat", {}).get("commit_arms", {}) or {}).get(arm)
        if arm == "A" and c_in is None:
            c_in = r.get("feat", {}).get("commit_step")
        out[r["ex_id"]] = dict(
            commit=c, commit_in=c_in, think_len=r["think_len"],
            fin=bool(r.get("fin", True)),
            correct=bool(r["correct"]), nprobe=len(ps),
            frac=(c / r["think_len"]) if (c and r["think_len"]) else None)
    have = [v for v in out.values() if v["commit"] is not None]
    print(f"[{name}:{arm}] 예제 {len(out)}  commit 있음 {len(have)} "
          f"({len(have)/max(1,len(out))*100:.1f}%)  "
          f"probe 중앙 {np.median([v['nprobe'] for v in out.values()]):.0f}개")
    if have:
        print(f"         commit_frac 중앙 {np.median([v['frac'] for v in have]):.3f}  "
              f"commit_step 중앙 {np.median([v['commit'] for v in have]):.0f}")
    # 스크립트 내장 eq_math(antlr 없음)로 찍힌 라벨과의 차이.
    # 생성은 같고 채점만 다르다 — 채점기가 라벨을 얼마나 움직이는지의 척도
    bothc = [(v["commit_in"], v["commit"]) for v in out.values()
             if v["commit_in"] is not None and v["commit"] is not None]
    if bothc:
        dd = np.array([y - x for x, y in bothc], float)
        ex = sum(1 for x, y in bothc if x == y)
        print(f"         내장채점 대비: 일치 {ex}/{len(bothc)} "
              f"({ex/len(bothc)*100:.1f}%)  |Δ| 중앙 {np.median(np.abs(dd)):.0f}토큰")
    return out


def quant(x, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    x = np.asarray(x, float)
    return "  ".join(f"p{int(q*100)}={np.quantile(x,q):+.0f}" for q in qs)


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 8:
        return float("nan")
    rx = np.argsort(np.argsort(x[m])).astype(float)
    ry = np.argsort(np.argsort(y[m])).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="out/sig_n500c.jsonl")
    ap.add_argument("--new", default="out/sig_para500.jsonl")
    ap.add_argument("--arms", default="A,B,C,D")
    a = ap.parse_args()

    print(f"[채점] {GRADER}\n")
    OLD = [json.loads(l) for l in open(a.old)]
    NEW = [json.loads(l) for l in open(a.new)]
    arms = [s.strip() for s in a.arms.split(",") if s.strip()]

    print("=" * 96)
    print("1. 각 데이터셋의 라벨 분포")
    print("=" * 96)
    o = summarize("구(격자)", OLD, "A")
    n = {}
    for arm in arms:
        n[arm] = summarize("신(문단)", NEW, arm)
    print()

    # ── 생성이 정말 같은지 먼저 확인한다. 다르면 아래 비교 전체가 무의미하다
    print("=" * 96)
    print("2. 생성 동일성 (이게 깨지면 아래 비교는 못 쓴다)")
    print("=" * 96)
    bo = {r["ex_id"]: r for r in OLD}
    bn = {r["ex_id"]: r for r in NEW}
    both = sorted(set(bo) & set(bn))
    dl = [(bo[i]["think_len"], bn[i]["think_len"]) for i in both]
    nsame = sum(1 for x, y in dl if x == y)
    print(f"  짝지은 예제 {len(both)}개 (구 {len(bo)} / 신 {len(bn)})")
    print(f"  think_len 일치 {nsame}/{len(both)} ({nsame/max(1,len(both))*100:.1f}%)")
    dif = [(i, bo[i]["think_len"], bn[i]["think_len"]) for i in both
           if bo[i]["think_len"] != bn[i]["think_len"]]
    for i, x, y in dif[:5]:
        print(f"    ex {i}: {x} → {y}")
    if len(dif) > 5:
        print(f"    … {len(dif)-5}개 더")
    ca = [(bo[i]["correct"], bn[i]["correct"]) for i in both]
    print(f"  correct 일치 {sum(1 for x,y in ca if x==y)}/{len(both)}")
    print()

    # ── 3. 핵심: commit_step 이 어디로 움직였나
    print("=" * 96)
    print("3. commit_step 이동 (구 A → 신 A)")
    print("=" * 96)
    na = n.get("A", {})
    pair = [(i, o[i]["commit"], na[i]["commit"]) for i in both
            if i in o and i in na]
    bb = [(i, x, y) for i, x, y in pair if x is not None and y is not None]
    print(f"  둘 다 commit 있음 {len(bb)}  "
          f"구만 {sum(1 for _,x,y in pair if x is not None and y is None)}  "
          f"신만 {sum(1 for _,x,y in pair if x is None and y is not None)}  "
          f"둘 다 없음 {sum(1 for _,x,y in pair if x is None and y is None)}")
    if bb:
        d = np.array([y - x for _, x, y in bb], float)
        rel = np.array([(y - x) / max(1, bo[i]["think_len"]) for i, x, y in bb])
        print(f"  Δ토큰: 중앙 {np.median(d):+.0f}  평균 {d.mean():+.0f}  "
              f"|Δ| 중앙 {np.median(np.abs(d)):.0f}")
        print(f"    {quant(d)}")
        print(f"  Δ진행률: 중앙 {np.median(rel)*100:+.1f}%p  "
              f"|Δ| 중앙 {np.median(np.abs(rel))*100:.1f}%p")
        print(f"  부호: 앞당겨짐 {int((d<0).sum())}  뒤로 {int((d>0).sum())}  "
              f"같음 {int((d==0).sum())}")
        # 격자 간격 차이만으로 설명되는 양: 신 격자가 더 촘촘하면
        # commit 이 **앞으로** 당겨지는 게 기대값이다(더 이른 지점을 볼 수 있으니)
        gap_o = np.median([np.median(np.diff(sorted(p["t"] for p in bo[i]["probes"])))
                           for i, _, _ in bb if len(bo[i]["probes"]) > 2])
        pn = [arm_probes(bn[i], "A") for i, _, _ in bb]
        gap_n = np.median([np.median(np.diff(sorted(p["t"] for p in q)))
                           for q in pn if q and len(q) > 2])
        print(f"  격자 간격 중앙: 구 {gap_o:.0f}토큰 → 신 {gap_n:.0f}토큰")
        print(f"    → 격자가 촘촘해진 것만으로 기대되는 이동은 음(앞당김) 방향, "
              f"크기 ~{(gap_o-gap_n)/2:.0f}토큰")
        print()
        print("  교란 점검 (이동이 길이·위치와 얽혔나)")
        L = np.array([bo[i]["think_len"] for i, _, _ in bb], float)
        F = np.array([x / max(1, bo[i]["think_len"]) for i, x, _ in bb], float)
        print(f"    Spearman(Δ, think_len)   = {spearman(d, L):+.3f}")
        print(f"    Spearman(|Δ|, think_len) = {spearman(np.abs(d), L):+.3f}")
        print(f"    Spearman(Δ, 구 commit_frac) = {spearman(d, F):+.3f}")
        print("    |ρ| < 0.2 면 이동이 길이와 무관 → 길이 교란이 되살아나지 않는다")
    print()

    # ── 4. 팔 사이 일치율 = 라벨 측정오차의 하한
    print("=" * 96)
    print("4. 팔 사이 일치 (신 데이터 안에서). 라벨 잡음의 하한")
    print("=" * 96)
    print("  팔쌍      commit 완전일치   |Δ| 중앙   frac 중앙차   둘다있음")
    base = n.get("A", {})
    for i0 in range(len(arms)):
        for i1 in range(i0 + 1, len(arms)):
            A1, A2 = arms[i0], arms[i1]
            d1, d2 = n.get(A1, {}), n.get(A2, {})
            ks = [k for k in d1 if k in d2]
            bo2 = [(d1[k]["commit"], d2[k]["commit"]) for k in ks
                   if d1[k]["commit"] is not None and d2[k]["commit"] is not None]
            if not bo2:
                continue
            dd = np.array([y - x for x, y in bo2], float)
            exact = sum(1 for x, y in bo2 if x == y)
            f1 = np.median([v["frac"] for v in d1.values() if v["frac"] is not None])
            f2 = np.median([v["frac"] for v in d2.values() if v["frac"] is not None])
            print(f"  {A1}–{A2}      {exact/len(bo2)*100:>5.1f}%        "
                  f"{np.median(np.abs(dd)):>5.0f}      "
                  f"{(f2-f1)*100:+5.1f}%p        {len(bo2)}")
    print()
    print("  해석: 완전일치율이 낮아도 frac 중앙차가 작으면")
    print("  → 예제별 라벨은 잡음이 많지만 집계 통계는 안전하다.")
    print("  라벨 잡음은 ±K 블록의 y 를 뒤섞으므로 **영가설 쪽으로** 편향된다.")
    print("  즉 지금 z 는 과장이 아니라 하한이다.")
    print()

    # ── 5. probe_commit 재실행 전 체크리스트
    print("=" * 96)
    print("5. 다음 단계")
    print("=" * 96)
    print("  python probe_commit.py out/sig_para500.jsonl \\")
    print("      --layers 18 27 36 --K 12 --nnull 100 --null both")
    print("  비교 대상: 구 데이터 L18 +0.073 (z=7.89) / L27 +0.064 (z=6.18) /")
    print("            L36 +0.040 (z=4.54)")
    print("  신 데이터에서 z 가 유지되면 → 문단 격자로 바꿔도 신호가 산다")
    print("  z 가 떨어지면 → 256 격자의 신호 일부가 '격자 자체'였다는 뜻이고,")
    print("                  그건 그것대로 보고할 결과다")


if __name__ == "__main__":
    main()