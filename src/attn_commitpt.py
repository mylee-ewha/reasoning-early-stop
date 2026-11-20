#!/usr/bin/env python3
"""attn_commitpt.py — commit 지점 토큰을 이후 토큰이 더 attend 하는가 (설계 (나), 파일럿).

가설 (교수님, 2026-11-18)
    답이 commit 된 뒤의 생각 토큰은 commit 이 일어난 지점의 토큰을 더 본다.
    probe 가 읽은 중간답은 롤백되어 본문에 없으므로, 이후 토큰이 볼 수 있는 대상은
    commit 이 찍힌 probe 지점의 본문 토큰(대개 문단 경계 \\n\\n)이다. 그 토큰의 h 에는
    이미 답이 들어 있다(detector 가 그걸 읽는다).

키(보이는 쪽)
    probe 지점 t 는 '생성 토큰 t 개 뒤'에 찍힌다(run_commit_math: t = s+1).
    그러므로 그 지점의 본문 토큰은 생각 좌표 t-1 이다.
    commit 키   = commit probe 의 토큰
    pre  키     = commit 이전 probe 들의 토큰   (대조, 같은 종류의 토큰)
    post 키     = commit 이후 probe 들의 토큰   (참고)

쿼리(보는 쪽) — 거리를 정확히 맞춘다
    각 키 u 마다 u+d (d ∈ --dists) 위치의 토큰을 쿼리로 쓴다. 그래서 commit 키와
    pre 키를 **같은 거리 d** 에서 비교한다. attention 은 가까울수록 크므로(최근성)
    거리를 안 맞추면 commit 키가 pre 키보다 가깝다는 것만으로 이긴다.

정규화
    쿼리 위치 q 의 attention 은 (프롬프트 + q+1) 개 키에 나뉘므로, 앞쪽 쿼리일수록
    키 하나가 받는 몫이 기계적으로 크다. 균등 분배 대비 배수 a·(npr+q+1) 로 바꾼다
    (1 = 균등과 같음). 이걸 안 하면 pre 키(앞쪽)가 유리해져 가설 반대쪽으로 치우친다.

지표 (예제마다, 층 묶음·거리마다)
    log( commit 키 / pre 키 평균 )   >0 이면 commit 지점을 더 본다
    예제 단위 부호검정. B·C 는 commit 이 없으므로 A 예제의 commit 상대위치를 빌려
    가짜 commit 을 둔다(probe_commit 의 matched 귀무와 같은 논리) → 기대값 0.

한계
    commit 이 첫 probe 인 예제는 pre 키가 없어 빠진다(= 가장 일찍 굳은 집단).
    commit 라벨은 probe 격자(중앙 ~140토큰) 해상도라, 실제로 굳은 토큰이 commit 키보다
    앞일 수 있다 — 그 경우 효과는 pre 쪽으로 새어 0 쪽으로 깎인다(보수적).

정확성 자체점검
    첫 예제·첫 층에서 다시 계산한 attention 으로 출력(p·V)을 만들어 sdpa 출력과 비교한다.

사용 (저장소 최상위에서)
    python -u src/attn_commitpt.py --model Qwen/Qwen3-8B --out data/sig_para500.jsonl --dry-run
    python -u src/attn_commitpt.py --model Qwen/Qwen3-8B --out data/sig_para500.jsonl \\
        --per-group 10 --c-correct --save results/attn/attn_pt_pilot.npz \\
        | tee results/attn/attn_pt_pilot.log
"""
import argparse
import json
import math
import os
import sys

import numpy as np


def sign_p(k, n):
    """양측 부호검정 정확 p."""
    if n == 0:
        return float("nan")
    m = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(m + 1)) / 2 ** n)


# ──────────────────────── attention 측정 훅 ────────────────────────
CTX = {}


def install_probe():
    """'sdpa' 를 감싸서 출력은 그대로 두고, 고른 쿼리 행의 attention 을 다시 계산해
    고른 키 열만 남긴다. 전체 행렬은 만들지 않는다([H, nq, T] 한 층분만 잠깐)."""
    import torch
    from transformers import AttentionInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    def probe_sdpa(module, query, key, value, attention_mask, **kw):
        out = sdpa_attention_forward(module, query, key, value, attention_mask, **kw)
        if CTX.get("active"):
            L = module.layer_idx
            qi, ki = CTX["q_idx"], CTX["k_idx"]
            H = query.shape[1]
            k, v = key, value
            if k.shape[1] != H:
                k = k.repeat_interleave(H // k.shape[1], dim=1)
                v = v.repeat_interleave(H // v.shape[1], dim=1)
            scale = kw.get("scaling") or module.scaling
            qs = query[0, :, qi, :].float()                          # [H, nq, D]
            sc = torch.einsum("hqd,hkd->hqk", qs, k[0].float()) * scale
            T = k.shape[2]
            causal = torch.arange(T, device=sc.device)[None, :] > qi[:, None]
            sc.masked_fill_(causal[None], float("-inf"))
            p = torch.softmax(sc, dim=-1)                            # [H, nq, T]
            CTX["out"][L] = p[:, :, ki].cpu().numpy().astype(np.float32)
            if CTX.get("check") and L == 0:
                o_ref = out[0][0, qi].float().transpose(0, 1)        # [H, nq, D]
                o_me = torch.einsum("hqk,hkd->hqd", p, v[0].float())
                CTX["check_err"] = float((o_me - o_ref).norm() / o_ref.norm())
        return out

    AttentionInterface.register("sdpa", probe_sdpa)


# ──────────────────────── 본체 ────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="data/sig_para500.jsonl")
    ap.add_argument("--script", default="src/run_commit_math.py")
    ap.add_argument("--data-file", default="data/math500.jsonl",
                    help="없으면 HF HuggingFaceH4/MATH-500 으로 대체된다")
    ap.add_argument("--data-n", type=int, default=500)
    ap.add_argument("--per-group", type=int, default=10)
    ap.add_argument("--groups", default="A,B,C")
    ap.add_argument("--c-correct", action="store_true",
                    help="C 를 최종 답이 정답인 예제로 제한 (grade2)")
    ap.add_argument("--dists", default="16,32,64,128,256,512",
                    help="키에서 쿼리까지 거리(토큰). 측정 격자이지 판정 임계가 아님")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-heads", action="store_true",
                    help="헤드별 VH[층, 헤드, 키, 거리] 도 저장 (예제당 ~1MB, attn_pt_heads.py 용)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--save", default="results/attn/attn_pt_pilot.npz")
    a = ap.parse_args()
    D = [int(x) for x in a.dists.split(",")]

    sys.path.insert(0, os.path.dirname(os.path.abspath(a.script)))
    from run_commit_math import build_prompt, load_math500
    from grade2 import eq_math2 as eq
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download
    md = a.model if os.path.isdir(a.model) else snapshot_download(a.model, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(md)

    ds = load_math500(a.data_file, None, a.data_n)
    by_uid = {r.get("unique_id"): r for r in ds}
    rows = [json.loads(l) for l in open(a.out) if l.strip()]
    rows = [r for r in rows if r.get("text") and r.get("uid") in by_uid]

    # commit — quartile_paired.commit_of 와 같은 정의(grade2)
    _ec = {}
    def eqc(x, y):
        if (x, y) not in _ec:
            _ec[(x, y)] = bool(eq(x, y))
        return _ec[(x, y)]

    def commit_g(r):
        ps = r.get("probes") or []
        if not ps:
            return None
        reps, cls = [], []
        for s_ in [p["ans"] for p in ps] + [r.get("pred")]:
            for i, q in enumerate(reps):
                if eqc(s_, q):
                    cls.append(i); break
            else:
                reps.append(s_); cls.append(len(reps) - 1)
        pc, fc = cls[:-1], cls[-1]
        j = len(pc)
        while j > 0 and pc[j - 1] == fc:
            j -= 1
        return ps[j]["t"] if j < len(pc) else None

    def group_of(r):
        if r["_commit"] is not None:
            return "A"
        if not r.get("fin", True):
            return "B"
        if a.c_correct and not eqc(r.get("pred"), r.get("gold")):
            return None
        return "C"

    for r in rows:
        r["_commit"] = commit_g(r)

    # ── 1단계: 토크나이저만으로 후보 정리 ──
    rng = np.random.default_rng(a.seed)
    cand = {g: [] for g in "ABC"}
    stat = {g: dict(n=0, badtok=0, first=0, ok=0, nl=0, nk=0) for g in "ABC"}
    a_fracs = []
    pending = []
    for r in rows:
        g = group_of(r)
        if g is None:
            continue
        st = stat[g]
        st["n"] += 1
        gids = tok(r["text"], add_special_tokens=False).input_ids
        deficit = r["gen_len"] - len(gids)
        if deficit < 0 or deficit > max(8, int(0.01 * r["gen_len"])):
            st["badtok"] += 1
            continue
        tl = min(r["think_len"], len(gids))
        ts = sorted({p["t"] for p in (r.get("probes") or []) if 1 <= p["t"] <= tl})
        if len(ts) < 2:
            st["first"] += 1
            continue
        if g == "A":
            c = r["_commit"]
            if c not in ts:
                st["badtok"] += 1
                continue
            a_fracs.append(c / tl)
            ci = ts.index(c)
        else:
            ci = None                      # 가짜 commit 은 A 상대위치가 다 모인 뒤에
        pending.append((g, r, gids, tl, ts, ci))

    for g, r, gids, tl, ts, ci in pending:
        st = stat[g]
        if ci is None:                     # B·C: A 예제 하나의 commit 상대위치를 빌린다
            f = a_fracs[rng.integers(len(a_fracs))]
            ci = int(np.argmin([abs(t - f * tl) for t in ts]))
        if ci == 0:
            st["first"] += 1               # pre 키가 없음
            continue
        st["ok"] += 1
        u = [t - 1 for t in ts]            # 키의 생각 좌표
        st["nl"] += sum("\n" in tok.decode([gids[x]]) for x in u)
        st["nk"] += len(u)
        cand[g].append(dict(r=r, gids=gids, tl=tl, u=u, ci=ci))

    print("[후보] commit 키·pre 키를 둘 다 가진 예제")
    print(f"  {'그룹':<4} {'예제':>5} {'사용':>5} {'첫 probe 가 commit':>18} "
          f"{'토큰 불일치':>10}  키 토큰에 개행 포함")
    for g in "ABC":
        st = stat[g]
        nlr = st["nl"] / st["nk"] if st["nk"] else float("nan")
        print(f"  {g:<4} {st['n']:>5} {st['ok']:>5} {st['first']:>18} "
              f"{st['badtok']:>10}  {nlr:.1%}")
    print("  (B·C 의 commit 은 A 예제의 commit 상대위치를 빌린 가짜 — 기대값 0 인 대조)")
    print("  ('첫 probe 가 commit' 은 pre 키가 없어 빠진 수 — 가장 일찍 굳은 집단)")
    if a.dry_run:
        return

    picks = []
    for g in a.groups.split(","):
        pool = cand[g]
        idx = rng.choice(len(pool), size=min(a.per_group, len(pool)), replace=False)
        picks += [(g, pool[i]) for i in sorted(idx)]
    print(f"\n[추출] 시드 {a.seed}: " + ", ".join(
        f"{g} {sum(1 for x, _ in picks if x == g)}개" for g in a.groups.split(",")))

    import torch
    from transformers import AutoModelForCausalLM
    install_probe()
    model = AutoModelForCausalLM.from_pretrained(
        md, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
    NL = model.config.num_hidden_layers
    bands = [(0, NL // 3), (NL // 3, 2 * NL // 3), (2 * NL // 3, NL)]
    bname = [f"L{lo}-{hi-1}" for lo, hi in bands]

    saved, summ = {}, []
    for k, (g, c) in enumerate(picks):
        r, tl, u, ci = c["r"], c["tl"], c["u"], c["ci"]
        pids = list(build_prompt(tok, by_uid[r["uid"]]))
        npr = len(pids)
        ids = torch.tensor([pids + list(c["gids"][:tl])], device="cuda")

        # 쿼리 = 각 키 u + d (생각 범위 안). 중복 제거 후 인덱스 표
        qset = sorted({x + d for x in u for d in D if x + d < tl})
        qpos = {q: i for i, q in enumerate(qset)}
        q_abs = torch.tensor([npr + q for q in qset], device="cuda")
        k_abs = torch.tensor([npr + x for x in u], device="cuda")
        CTX.update(active=True, q_idx=q_abs, k_idx=k_abs, out={}, check=(k == 0))
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
        CTX["active"] = False
        if k == 0:
            print(f"  [자체점검] 층0 p·V vs sdpa 출력 상대오차 {CTX['check_err']:.2e} "
                  "(bf16 이라 1e-2 안팎이면 정상, 0.1 이상이면 훅이 틀림)")

        P = np.stack([CTX["out"][L] for L in range(NL)])           # [L, H, nq, nk]
        unif = (npr + np.array(qset) + 1).astype(np.float64)       # 균등 분배 대비 배수
        P = P * unif[None, None, :, None]

        # VH[L, H, key, d] — 키 u 에서 거리 d 의 쿼리가 준 attention (없으면 nan)
        VH = np.full((NL, P.shape[1], len(u), len(D)), np.nan, dtype=np.float32)
        for j, x in enumerate(u):
            for di, d in enumerate(D):
                if x + d in qpos:
                    VH[:, :, j, di] = P[:, :, qpos[x + d], j]
        V = VH.mean(1)                                             # [L, nk, nd] 헤드 평균

        rec = dict(group=g, ex_id=r["ex_id"], think_len=tl, commit_t=u[ci] + 1,
                   n_pre=ci, n_post=len(u) - ci - 1)
        line = f"  [{g}] ex {r['ex_id']:>3}  키 {len(u)}개 (pre {ci}, post {len(u)-ci-1})"
        for bi, (lo, hi) in enumerate(bands):
            Vb = V[lo:hi].mean(0)                                  # [nk, nd]
            cm = Vb[ci]
            pre = np.nanmean(Vb[:ci], 0) if ci else np.full(len(D), np.nan)
            post = (np.nanmean(Vb[ci + 1:], 0) if ci + 1 < len(u)
                    else np.full(len(D), np.nan))
            rec[f"lr_pre_b{bi}"] = np.log(cm / pre).tolist()
            rec[f"lr_post_b{bi}"] = np.log(cm / post).tolist()
            rec[f"commit_b{bi}"] = cm.tolist()
            rec[f"pre_b{bi}"] = pre.tolist()
        mid = rec["lr_pre_b1"]
        line += "  중간층 log(commit/pre) " + " ".join(
            f"{d}:{v:+.2f}" if np.isfinite(v) else f"{d}:  - " for d, v in zip(D, mid))
        print(line, flush=True)
        summ.append(rec)
        saved[f"ex{r['ex_id']}_u"] = np.array(u)
        saved[f"ex{r['ex_id']}_V"] = V.astype(np.float32)          # [L, nk, nd]
        if a.save_heads:
            saved[f"ex{r['ex_id']}_VH"] = VH.astype(np.float16)    # [L, H, nk, nd]
        torch.cuda.empty_cache()

    # ── 요약: 그룹·층 묶음·거리마다 예제 단위 부호검정 ──
    print("\n[요약] log(commit 키 / pre 키 평균), 같은 거리 d 에서. >0 = commit 지점을 더 본다")
    print("       B·C 는 가짜 commit — 0 근처여야 한다. 칸 = 중앙값 (양수/유효 예제, 부호검정 p)")
    for key, title in [("lr_pre", "commit vs pre"), ("lr_post", "commit vs post (참고)")]:
        print(f"\n  ── {title}")
        print("  " + f"{'':<10}" + "".join(f"{'d='+str(d):>22}" for d in D))
        for g in a.groups.split(","):
            G = [x for x in summ if x["group"] == g]
            if not G:
                continue
            for bi in range(len(bands)):
                M = np.array([x[f"{key}_b{bi}"] for x in G])       # [n, nd]
                cells = []
                for di in range(len(D)):
                    col = M[:, di][np.isfinite(M[:, di])]
                    if len(col) == 0:
                        cells.append(f"{'-':>22}"); continue
                    kpos = int((col > 0).sum())
                    cells.append(f"{np.median(col):+.2f} ({kpos}/{len(col)}, "
                                 f"{sign_p(kpos, len(col)):.2f})".rjust(22))
                print(f"  {g+' '+bname[bi]:<10}" + "".join(cells))
    print("\n판정(파일럿): A 의 commit vs pre 가 여러 d 에서 양수 쪽이고 B·C 는 0 근처면 가설 쪽.")
    print("              A 도 B·C 도 같이 양수면 '최근 probe 지점을 더 본다' 같은 위치 성질이지")
    print("              commit 고유의 효과가 아니다. 그룹당 10개는 경향만 — 판정은 확대 후.")

    os.makedirs(os.path.dirname(a.save) or ".", exist_ok=True)
    np.savez_compressed(a.save, summary=json.dumps(summ, ensure_ascii=False),
                        dists=np.array(D), **saved)
    print(f"\n저장: {a.save}  (예제별 키 위치 u, V[층, 키, 거리] 헤드 평균"
          + (", VH[층, 헤드, 키, 거리]" if a.save_heads else "") + ")")


if __name__ == "__main__":
    main()