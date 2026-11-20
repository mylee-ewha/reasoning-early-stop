#!/usr/bin/env python3
"""attn_commit.py — commit 이후 토큰이 '앞서 적힌 답'을 더 attend 하는가 (파일럿).

가설 (교수님, 2026-11-18)
    답이 commit 된 뒤의 생각 토큰은, 체인 안에서 그 답이 처음 적힌 구간에
    attention 을 더 많이 준다. 이 증가분은 같은 거리의 이웃 구간 대비로도 남는다.

그룹 (예제 단위, 고정 시드로 무작위 추출)
    A  commit 있음                        기준 답 = 모델 최종 답(pred)
    B  commit 없음 + 절단                 기준 답 = 마지막 probe 답(최종 답이 없으므로)
    C  commit 없음 + 완주(답이 계속 바뀜)   기준 답 = 최종 답. --c-correct 면 정답만

답 구간
    생각 텍스트에서 기준 답 문자열이 **처음** 나타난 곳의 토큰들.
    숫자 답은 앞뒤가 숫자·소수점이 아닌 경우만 매칭(‘12’ 가 ‘312’ 에 걸리지 않게).
    본문에 답이 안 나오는 예제는 이 가설로 볼 수 없으므로 표에서 빠지고, 비율을 찍는다.

지표 (거리 맞춘 대조)
    쿼리 = 답 구간 뒤의 생각 토큰을 --qstride 마다.
    키당 attention:   span = (답 구간 attention 합) / 답 구간 길이
                      nb   = (답 구간 앞뒤 ±W 이웃의 attention 합) / 이웃 길이
    R = span / nb  (헤드 평균 후 비). 1 보다 크면 '같은 거리의 이웃보다 답을 더 본다'.
    A 는 쿼리를 commit 전/후로 나눠 R_post vs R_pre 를 예제 안에서 짝지어 비교.

계산 (전체 attention 행렬을 저장하지 않는다)
    sdpa 는 그대로 쓰고(출력 동일), 각 층에서 **고른 쿼리 행만** q·kᵀ 로 다시 계산해
    softmax 한 뒤 구간 합만 남기고 버린다. 8k 토큰 전체 행렬(층당 수 GB)은 만들지 않는다.
    층당 추가 메모리 ≈ 헤드 32 × 쿼리 수 × 시퀀스 길이 × 4B (쿼리 512 개면 약 0.5GB).

사용
    python -u src/attn_commit.py --model Qwen/Qwen3-8B --out data/sig_para500.jsonl \\
        --data-file <MATH-500 파일> --dry-run          # GPU 없이: 본문에 답이 나오는 비율
    python -u src/attn_commit.py --model Qwen/Qwen3-8B --out data/sig_para500.jsonl \\
        --data-file <MATH-500 파일> --per-group 10 --save results/attn/attn_pilot.npz \\
        | tee results/attn/attn_pilot.log

이 스크립트는 **파일럿**이다. 그룹당 10개로는 패턴을 보고 가설을 다듬는 데까지만
쓴다. 판정은 전체 확대 + 검정에서.
"""
import argparse
import json
import os
import re
import sys

import numpy as np


# ──────────────────────── 답 구간 찾기 ────────────────────────
NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")


def variants(ans):
    """기준 답의 표기 변형. 너무 넓히면 엉뚱한 곳에 걸리므로 최소한만."""
    a = (ans or "").strip()
    if not a:
        return []
    vs = {a, a.replace(" ", ""), a.replace("\\dfrac", "\\frac"),
          a.replace("\\tfrac", "\\frac")}
    return [v for v in vs if v]


def find_first(think, ans):
    """생각 텍스트에서 기준 답이 처음 나오는 (시작, 끝) 문자 위치와 총 등장 횟수."""
    best, count = None, 0
    for v in variants(ans):
        if NUM_RE.match(v):
            pat = re.compile(r"(?<![\d.])" + re.escape(v) + r"(?![\d]|\.\d)")
        else:
            pat = re.compile(re.escape(v))
        ms = list(pat.finditer(think))
        count = max(count, len(ms))
        if ms and (best is None or ms[0].start() < best[0]):
            best = (ms[0].start(), ms[0].end())
    return best, count


def find_all(think, ans):
    """기준 답의 모든 등장 (시작, 끝) 문자 구간 — 변형끼리 겹치면 합친다."""
    iv = []
    for v in variants(ans):
        if NUM_RE.match(v):
            pat = re.compile(r"(?<![\d.])" + re.escape(v) + r"(?![\d]|\.\d)")
        else:
            pat = re.compile(re.escape(v))
        iv += [(m.start(), m.end()) for m in pat.finditer(think)]
    iv.sort()
    out = []
    for st, en in iv:
        if out and st < out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], en))
        else:
            out.append((st, en))
    return out


# ──────────────────────── attention 측정 훅 ────────────────────────
CTX = {}   # forward 직전에 채운다: q_idx, spans, out


def install_probe():
    """transformers 의 'sdpa' 함수를 감싸서, 출력은 그대로 두고
    고른 쿼리 행의 attention 만 따로 계산해 구간 합을 CTX['out'] 에 쌓는다."""
    import torch
    from transformers import AttentionInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    def probe_sdpa(module, query, key, value, attention_mask, **kw):
        out = sdpa_attention_forward(module, query, key, value, attention_mask, **kw)
        if CTX.get("active"):
            L = module.layer_idx
            qi = CTX["q_idx_t"]                          # [nq] 절대 위치
            H, D = query.shape[1], query.shape[-1]
            k = key
            if k.shape[1] != H:                          # GQA: kv 헤드를 늘린다
                k = k.repeat_interleave(H // k.shape[1], dim=1)
            scale = kw.get("scaling") or module.scaling
            qs = query[0, :, qi, :].float()              # [H, nq, D]
            sc = torch.einsum("hqd,hkd->hqk", qs, k[0].float()) * scale
            T = k.shape[2]
            causal = torch.arange(T, device=sc.device)[None, :] > qi[:, None]
            sc.masked_fill_(causal[None], float("-inf"))
            p = torch.softmax(sc, dim=-1)                # [H, nq, T]
            res = {}
            for name, m in CTX["masks"].items():         # m: [T] bool
                res[name] = (p * m[None, None, :]).sum(-1)   # [H, nq]
            CTX["out"][L] = {n: v.cpu().numpy().astype(np.float32)
                             for n, v in res.items()}
        return out

    AttentionInterface.register("sdpa", probe_sdpa)


# ──────────────────────── 본체 ────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="data/sig_para500.jsonl")
    ap.add_argument("--script", default="src/run_commit_math.py")
    ap.add_argument("--data-file", default="test.jsonl")
    ap.add_argument("--data-n", type=int, default=500,
                    help="수집 때 쓴 --n 과 같아야 uid 가 맞는다")
    ap.add_argument("--per-group", type=int, default=10)
    ap.add_argument("--groups", default="A,B,C")
    ap.add_argument("--c-correct", action="store_true",
                    help="C 그룹을 최종 답이 정답인 예제로 제한 (grade2)")
    ap.add_argument("--span", choices=["all", "first"], default="all",
                    help="all: 답의 모든 등장 위치(쿼리 시점까지 적힌 것) / first: 첫 등장만")
    ap.add_argument("--qstride", type=int, default=16, help="쿼리 간격(토큰)")
    ap.add_argument("--nb", type=int, default=64, help="이웃 대조 구간 반폭(토큰)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="모델 없이: 그룹별 예제 수와 본문에 답이 나오는 비율만")
    ap.add_argument("--save", default="results/attn/attn_pilot.npz")
    a = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(a.script)))
    from run_commit_math import build_prompt, load_math500
    try:
        from grade2 import eq_math2 as eq
    except ImportError:
        eq = None
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download
    md = a.model if os.path.isdir(a.model) else snapshot_download(a.model, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(md)

    ds = load_math500(a.data_file, None, a.data_n)
    by_uid = {r.get("unique_id"): r for r in ds}
    rows = [json.loads(l) for l in open(a.out) if l.strip()]
    rows = [r for r in rows if r.get("text") and r.get("uid") in by_uid]

    _ec = {}
    def eqc(x, y):
        k = (x, y)
        if k not in _ec:
            _ec[k] = bool(eq(x, y))
        return _ec[k]

    def commit_g(r):
        """quartile_paired.commit_of 와 같은 정의(grade2). 내장 eq_math 기반
        feat.commit_step 은 표기 차이를 '답 바뀜'으로 세어 commit 을 놓친다."""
        if eq is None:
            return r.get("feat", {}).get("commit_step")
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

    for r in rows:
        r["_commit"] = commit_g(r)
    n_moved = sum(1 for r in rows if (r["_commit"] is None) !=
                  (r.get("feat", {}).get("commit_step") is None))
    print(f"[commit] grade2 로 재도출 — 내장 채점 기준과 유무가 다른 예제 {n_moved}개")

    def group_of(r):
        cs = r["_commit"]
        if cs is not None:
            return "A"
        if not r.get("fin", True):
            return "B"
        if a.c_correct and eq is not None and not eq(r.get("pred"), r.get("gold")):
            return None
        return "C"

    def ref_answer(r, g):
        if g == "B":
            ps = sorted(r.get("probes") or [], key=lambda p: p["t"])
            return ps[-1]["ans"] if ps else None
        return r.get("pred")

    # ── 1단계: 모든 예제에서 답 구간을 찾는다 (토크나이저만) ──
    cand = {g: [] for g in "ABC"}
    stat = {g: dict(n=0, found=0, nofind=0, badtok=0, occ=[]) for g in "ABC"}
    for r in rows:
        g = group_of(r)
        if g is None:
            continue
        st = stat[g]
        st["n"] += 1
        enc = tok(r["text"], add_special_tokens=False, return_offsets_mapping=True)
        gids, offs = enc.input_ids, enc.offset_mapping
        deficit = r["gen_len"] - len(gids)
        if deficit < 0 or deficit > max(8, int(0.01 * r["gen_len"])):
            st["badtok"] += 1
            continue
        tl = min(r["think_len"], len(gids))
        think_chars = offs[tl - 1][1] if tl > 0 else 0
        ivs = find_all(r["text"][:think_chars], ref_answer(r, g))
        if not ivs:
            st["nofind"] += 1
            continue
        if a.span == "first":
            ivs = ivs[:1]
        st["found"] += 1
        st["occ"].append(len(find_all(r["text"][:think_chars], ref_answer(r, g))))
        smask = np.zeros(tl, dtype=bool)
        for c0, c1 in ivs:
            s_tok = next(i for i, (s, e) in enumerate(offs) if e > c0)
            e_tok = next((i for i, (s, e) in enumerate(offs) if s >= c1), tl)
            smask[s_tok:min(e_tok, tl)] = True
        # 이웃 = 각 등장의 앞뒤 ±nb 토큰에서 답 토큰을 뺀 것
        nmask = np.zeros(tl, dtype=bool)
        idx = np.flatnonzero(smask)
        for b0, b1 in zip(np.r_[idx[0], idx[1:][np.diff(idx) > 1]],
                          np.r_[idx[:-1][np.diff(idx) > 1], idx[-1]] + 1):
            nmask[max(0, b0 - a.nb):b0] = True
            nmask[b1:min(tl, b1 + a.nb)] = True
        nmask &= ~smask
        e_first = int(idx[idx < tl][0]) + 1
        while e_first < tl and smask[e_first]:
            e_first += 1
        cand[g].append(dict(r=r, gids=gids, tl=tl, smask=smask, nmask=nmask,
                            s=int(idx[0]), e=e_first, cnt=len(ivs),
                            ans=ref_answer(r, g)))

    print("[답 구간] 생각 텍스트에 기준 답이 문자열로 나오는가")
    print(f"  {'그룹':<4} {'예제':>5} {'나옴':>5} {'안 나옴':>7} {'토큰 불일치':>10}  등장 횟수 중앙")
    for g in "ABC":
        st = stat[g]
        occ = np.median(st["occ"]) if st["occ"] else float("nan")
        print(f"  {g:<4} {st['n']:>5} {st['found']:>5} {st['nofind']:>7} "
              f"{st['badtok']:>10}  {occ:.0f}")
    print(f"  (--span {a.span}: " + ("쿼리 시점까지 적힌 모든 등장을 답 구간으로 본다"
          if a.span == "all" else "첫 등장만 답 구간으로 본다") + ")")
    if a.dry_run:
        return

    rng = np.random.default_rng(a.seed)
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

    saved = {}
    summ = []
    for k, (g, c) in enumerate(picks):
        r = c["r"]
        pids = list(build_prompt(tok, by_uid[r["uid"]]))
        npr = len(pids)
        tl, s, e = c["tl"], c["s"], c["e"]
        ids = torch.tensor([pids + list(c["gids"][:tl])], device="cuda")
        T = ids.shape[1]
        qs = np.arange(e + a.qstride, tl, a.qstride)        # 생각 토큰 좌표
        if len(qs) == 0:
            print(f"  [{g}] ex {r['ex_id']}: 답 구간 뒤 쿼리 없음 — 건너뜀")
            continue
        qa = qs + npr                                        # 절대 위치
        span_m = torch.zeros(T, dtype=torch.bool, device="cuda")
        span_m[npr:npr + tl] = torch.from_numpy(c["smask"]).cuda()
        nb_m = torch.zeros(T, dtype=torch.bool, device="cuda")
        nb_m[npr:npr + tl] = torch.from_numpy(c["nmask"]).cuda()
        CTX.update(active=True, q_idx_t=torch.tensor(qa, device="cuda"),
                   masks={"span": span_m, "nb": nb_m}, out={})
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
        CTX["active"] = False

        # 쿼리보다 뒤의 키는 causal 로 0 이므로 쿼리마다 '그때까지 적힌' 키 수로 나눈다
        span_len = np.cumsum(c["smask"])[qs].astype(np.float64)
        nb_len = np.maximum(np.cumsum(c["nmask"])[qs], 1).astype(np.float64)
        span_pk = np.stack([CTX["out"][L]["span"] for L in range(NL)]) / span_len[None, None, :]
        nb_pk = np.stack([CTX["out"][L]["nb"] for L in range(NL)]) / nb_len[None, None, :]
        R = span_pk.mean(1) / np.maximum(nb_pk.mean(1), 1e-12)                       # [L,nq]

        cs = r["_commit"]
        rec = dict(group=g, ex_id=r["ex_id"], think_len=tl, commit=cs,
                   ans=c["ans"], span=(s, e), occ=c["cnt"])
        rec["span_len_q"] = [int(x) for x in span_len]
        bands = [(0, NL // 3), (NL // 3, 2 * NL // 3), (2 * NL // 3, NL)]
        line = f"  [{g}] ex {r['ex_id']:>3}  답 '{str(c['ans'])[:14]}' @{s}-{e} (등장 {c['cnt']}, 첫 등장 기준)"
        if g == "A" and cs is not None:
            pre, post = qs < cs, qs >= cs
            rec.update(n_pre=int(pre.sum()), n_post=int(post.sum()))
            line += f"  commit {cs}  쿼리 전/후 {pre.sum()}/{post.sum()}"
            for bi, (lo, hi) in enumerate(bands):
                Rb = R[lo:hi].mean(0)
                rp = float(np.mean(Rb[pre])) if pre.any() else np.nan
                rq = float(np.mean(Rb[post])) if post.any() else np.nan
                rec[f"R_pre_b{bi}"], rec[f"R_post_b{bi}"] = rp, rq
            line += "  R 전→후 " + "  ".join(
                f"L{lo}-{hi-1} {rec[f'R_pre_b{bi}']:.2f}→{rec[f'R_post_b{bi}']:.2f}"
                for bi, (lo, hi) in enumerate(bands))
        else:
            for bi, (lo, hi) in enumerate(bands):
                rec[f"R_all_b{bi}"] = float(R[lo:hi].mean())
            line += "  R " + "  ".join(
                f"L{lo}-{hi-1} {rec[f'R_all_b{bi}']:.2f}" for bi, (lo, hi) in enumerate(bands))
        print(line, flush=True)
        summ.append(rec)
        saved[f"ex{r['ex_id']}_q"] = qs
        saved[f"ex{r['ex_id']}_span_pk"] = span_pk.astype(np.float16)   # [L,H,nq]
        saved[f"ex{r['ex_id']}_nb_pk"] = nb_pk.astype(np.float16)
        torch.cuda.empty_cache()

    # ── 요약 ──
    print("\n[요약] R = (답 구간 키당 attention) ÷ (같은 거리 이웃 키당 attention). 1 = 이웃과 같음")
    A = [x for x in summ if x["group"] == "A" and x.get("n_pre", 0) and x.get("n_post", 0)]
    for bi, nm in enumerate(["앞쪽 층", "중간 층", "뒤쪽 층"]):
        if A:
            d = np.array([x[f"R_post_b{bi}"] - x[f"R_pre_b{bi}"] for x in A])
            print(f"  A {nm}: R 전 중앙 {np.median([x[f'R_pre_b{bi}'] for x in A]):.2f} → "
                  f"후 {np.median([x[f'R_post_b{bi}'] for x in A]):.2f}   "
                  f"후>전 {int((d > 0).sum())}/{len(d)}")
        for g in "BC":
            G = [x[f"R_all_b{bi}"] for x in summ if x["group"] == g]
            if G:
                print(f"  {g} {nm}: R 중앙 {np.median(G):.2f}  (n={len(G)})")
    nA_nopre = sum(1 for x in summ if x["group"] == "A" and not x.get("n_pre"))
    if nA_nopre:
        print(f"  A 중 {nA_nopre}개는 답이 commit 이후에야 본문에 처음 적혀서 '전' 쿼리가 없음")
    print("\n판정(파일럿): A 에서 후 > 전 이 대부분이고 B·C 의 R 이 1 근처면 가설 쪽.")
    print("              A 의 전/후가 비슷하면 '답을 다시 본다'는 commit 과 무관한 일반 성질.")
    print("              그룹당 10개라 경향만 본다 — 판정은 전체 확대 + 부호검정에서.")

    os.makedirs(os.path.dirname(a.save) or ".", exist_ok=True)
    np.savez_compressed(a.save, summary=json.dumps(summ, ensure_ascii=False), **saved)
    print(f"\n저장: {a.save}  (예제별 q, span_pk/nb_pk [층, 헤드, 쿼리] — receiver head 분석용)")


if __name__ == "__main__":
    main()