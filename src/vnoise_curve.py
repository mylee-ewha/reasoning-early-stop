#!/usr/bin/env python3
"""vnoise_curve.py — **모든 probe 지점**에서 V 노이즈 반응(KL)을 잰다 (GPU).

vnoise_pretest.py 와 같은 측정(같은 forced-close 문구, 같은 헤드·토큰별 RMS 노이즈,
같은 지배 채널 제외)을 쓰되, 두 가지가 다르다.

  1. 지점 4개(pre/post/c0/c1)가 아니라 **예제의 모든 probe 지점**. 토큰 위치 t 를
     같이 저장한다. → v3 의 구멍 ①(위치 미저장)·②(commit 이후 통제쌍 없음)를 메우고,
     배포 시뮬레이션(detector 발화 → 1문단 뒤 KL 비교 → 수락/계속)을 오프라인으로
     돌릴 수 있는 곡선이 된다. commit 없는 예제(B·C)도 포함한다 — 배포에서는
     detector 가 거기서도 발화하기 때문이다.
  2. 지점마다 접두사를 다시 prefill 하지 않는다. 체인 전체를 **한 번** prefill 하고,
     지점 t 의 캐시는 그 K·V 의 앞 (프롬프트+t) 개를 잘라 쓴다. causal attention
     이므로 앞 위치의 K·V 는 뒤 토큰과 무관하다. 자른 텐서는 건드리지 않고
     (노이즈는 새 텐서로 더함), 닫기 문구는 매번 새 캐시에 붙는다.
     → 예제당 prefill 1회 + 지점마다 짧은 forward 몇 번.
     첫 예제에서 '잘라 쓴 캐시' 와 '그 길이만 새로 prefill 한 캐시' 의 답 분포를
     비교하는 자체점검을 찍는다 (bf16 이라 1e-2 안팎이면 정상).

칸(섭동군:c)은 실행 전에 고정한다. 기본값은 v3 재판정에서 깨끗했던 칸 중
예제 단위 구별력(vnoise_sep.py)이 가장 높았던 둘: 18:2.0, all:0.5.

출력: jsonl, 예제 한 줄
  {ex_id, uid, think_len, fin, commit_feat, n_prompt,
   points: [{t, p_top, cells: {"18|2.0": {kl, qa, flip}, ...}}, ...]}
  --resume 이면 이미 있는 ex_id 는 건너뛴다(긴 실행이 끊겨도 이어서).

사용 (저장소 최상위):
  python -u src/vnoise_curve.py --model Qwen/Qwen3-8B --out data/sig_para500.jsonl --dry-run
  python -u src/vnoise_curve.py --model Qwen/Qwen3-8B --out data/sig_para500.jsonl --n 5 \\
      --save results/vnoise/vnoise_curve_smoke.jsonl
  python -u src/vnoise_curve.py --model Qwen/Qwen3-8B --out data/sig_para500.jsonl \\
      --save results/vnoise/vnoise_curve.jsonl --resume | tee results/vnoise/vnoise_curve.log
"""
import argparse
import json
import os
import sys
import time

import numpy as np


def resolve_model_dir(m):
    if os.path.isdir(m):
        return m
    from huggingface_hub import snapshot_download
    return snapshot_download(m, local_files_only=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="data/sig_para500.jsonl")
    ap.add_argument("--script", default="src/run_commit_math.py")
    ap.add_argument("--data-file", default="data/math500.jsonl",
                    help="없으면 HF HuggingFaceH4/MATH-500 으로 대체된다")
    ap.add_argument("--data-n", type=int, default=500)
    ap.add_argument("--n", type=int, default=0, help="예제 수 (0 = 전부). 시드로 섞은 순서")
    ap.add_argument("--cells", default="18:2.0,all:0.5",
                    help="섭동군:c 를 쉼표로. 섭동군 = '18' 단일층 / '18-27' 구간 / 'all'")
    ap.add_argument("--reps", type=int, default=3, help="칸당 노이즈 샘플 수")
    ap.add_argument("--hit-sink", action="store_true",
                    help="헤드별 지배 V 채널도 흔든다(기본은 제외, vnoise_pretest 와 같음)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="results/vnoise/vnoise_curve.jsonl")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    sys.path.insert(0, os.path.dirname(os.path.abspath(a.script)))
    from run_commit_math import build_prompt, load_math500, PROBE_ARMS

    md = resolve_model_dir(a.model)
    tok = AutoTokenizer.from_pretrained(md)
    close_ids = tok(PROBE_ARMS["A"], add_special_tokens=False).input_ids

    ds = load_math500(a.data_file, None, a.data_n)
    by_uid = {r.get("unique_id"): r for r in ds}
    rows = [json.loads(l) for l in open(a.out) if l.strip()]
    rows = [r for r in rows if r.get("probes") and r.get("text") and r.get("uid") in by_uid]
    order = np.random.default_rng(a.seed).permutation(len(rows))
    rows = [rows[i] for i in order]
    if a.n:
        rows = rows[:a.n]

    done = set()
    if a.resume and os.path.exists(a.save):
        done = {json.loads(l)["ex_id"] for l in open(a.save) if l.strip()}

    def gen_ids(r):
        # vnoise_pretest.gen_ids 와 같음: text 는 </think> 등 특수토큰이 빠져 있어
        # 재토큰화 길이가 gen_len 보다 1 작다. 앞 t 개만 쓰므로 그대로 쓴다.
        gids = tok(r["text"], add_special_tokens=False).input_ids
        d = r["gen_len"] - len(gids)
        if d < 0 or d > max(8, int(0.01 * r["gen_len"])):
            return None
        return gids

    def points_of(r, gids):
        lim = min(r["think_len"], len(gids))
        return sorted({p["t"] for p in r["probes"] if 1 <= p["t"] <= lim})

    if a.dry_run:
        ok = bad = 0
        npts = []
        for r in rows:
            g = gen_ids(r)
            if g is None:
                bad += 1
                continue
            ok += 1
            npts.append(len(points_of(r, g)))
        nc = len(a.cells.split(","))
        print(f"[dry-run] 예제 {len(rows)}개 (이미 저장 {len(done)}개): 사용 {ok}, 토큰 결손 {bad}")
        if npts:
            print(f"  지점 수: 중앙 {np.median(npts):.0f}, 합 {sum(npts):,}  → forward "
                  f"{sum(npts) * (1 + nc * a.reps):,}회 (닫기 문구 {len(close_ids)}토큰짜리)"
                  f" + prefill {ok}회")
        return

    model = AutoModelForCausalLM.from_pretrained(
        md, dtype=torch.bfloat16, device_map="cuda").eval()
    NL = model.config.num_hidden_layers
    CELLS = []
    for spec in a.cells.split(","):
        gname, c = spec.split(":")
        if gname == "all":
            Ls = list(range(NL))
        elif "-" in gname:
            lo_, hi_ = gname.split("-")
            Ls = list(range(int(lo_), int(hi_) + 1))
        else:
            Ls = [int(gname)]
        CELLS.append((gname, float(c), Ls))
    print("[칸] " + ", ".join(f"{g}:{c:g}({len(Ls)}층)" for g, c, Ls in CELLS)
          + f"  × 노이즈 {a.reps}회")
    gen = torch.Generator(device="cuda").manual_seed(a.seed)

    def kv_of(cache, L):
        if hasattr(cache, "layers"):
            return cache.layers[L].keys, cache.layers[L].values
        return cache.key_cache[L], cache.value_cache[L]

    @torch.no_grad()
    def dist(Ks, Vs):
        """잘라 둔 K·V 로 새 캐시를 만들고 닫기 문구를 붙여 답 첫 토큰 log 분포를 낸다.
        update 가 cat 으로 새 텐서를 만들므로 Ks·Vs 원본은 바뀌지 않는다."""
        cache = DynamicCache()
        for L in range(NL):
            cache.update(Ks[L], Vs[L], L)
        o = model(input_ids=torch.tensor([close_ids], device="cuda"),
                  past_key_values=cache, use_cache=True, logits_to_keep=1)
        return torch.log_softmax(o.logits[0, -1].float(), -1)

    CTXCHK = {}

    @torch.no_grad()
    def run(r, check=False):
        gids = gen_ids(r)
        if gids is None:
            return None
        ts = points_of(r, gids)
        if not ts:
            return None
        pids = list(build_prompt(tok, by_uid[r["uid"]]))
        npr = len(pids)
        ids = torch.tensor([pids + list(gids[:max(ts)])], device="cuda")
        base = model(input_ids=ids, use_cache=True, logits_to_keep=1).past_key_values
        K, V = zip(*[kv_of(base, L) for L in range(NL)])

        pts = []
        for ti, t in enumerate(ts):
            n = npr + t
            Ks = [k[:, :, :n] for k in K]
            Vs = [v[:, :, :n] for v in V]
            lp0 = dist(Ks, Vs)
            p0 = lp0.exp()
            top = int(lp0.argmax())

            if check and ti == len(ts) // 2:
                fresh = model(input_ids=ids[:, :n], use_cache=True,
                              logits_to_keep=1).past_key_values
                Kf, Vf = zip(*[kv_of(fresh, L) for L in range(NL)])
                lpf = dist(list(Kf), list(Vf))
                idx = lpf.topk(20).indices
                err = float((lp0[idx] - lpf[idx]).abs().max())
                kl_sf = float((lpf.exp() * (lpf - lp0)).sum())
                print(f"  [자체점검] ex{r['ex_id']} t={t}: 잘라 쓴 캐시 vs 새 prefill — "
                      f"KL(새‖잘라쓴) {kl_sf:.2e}, top 확률 {float(lpf.exp()[top]):.4f} vs {float(p0[top]):.4f}, "
                      f"argmax {'같음' if int(lpf.argmax()) == top else '다름'}, 상위20 log p 최대차 {err:.3f}")
                print("      → KL 이 아래 노이즈 KL(이 지점 값)보다 한 자릿수 이상 작고 argmax 가 같으면 정상."
                      " 상위20 최대차는 확률 낮은 꼬리 토큰에서 크게 나올 수 있어 판정에 안 쓴다")
                CTXCHK["kl_sf"] = kl_sf
                del fresh, Kf, Vf

            cells = {}
            for gname, c, Ls in CELLS:
                prep = []
                for L in Ls:
                    v = Vs[L][0]                                  # (nkv, n, hd)
                    vf = v.float()
                    rms = vf.pow(2).mean(-1, keepdim=True).sqrt()  # 헤드·토큰별
                    msk = None
                    if not a.hit_sink:
                        kk = vf.pow(2).sum(1).argmax(-1)         # 헤드별 지배 채널
                        msk = torch.ones_like(v, dtype=torch.bool)
                        msk[torch.arange(v.shape[0], device=v.device), :, kk] = False
                    prep.append((L, rms, msk))
                kl, qa, fl = [], [], []
                for _ in range(a.reps):
                    Vn = list(Vs)
                    for L, rms, msk in prep:
                        eps = torch.randn(Vs[L][0].shape, generator=gen, device="cuda",
                                          dtype=torch.float32) * (c * rms)
                        if msk is not None:
                            eps = eps.masked_fill(~msk, 0.0)
                        Vn[L] = Vs[L] + eps.unsqueeze(0).to(Vs[L].dtype)
                    lp1 = dist(Ks, Vn)
                    kl.append(float((p0 * (lp0 - lp1)).sum()))
                    qa.append(float(lp1[top].exp()))
                    fl.append(float(int(lp1.argmax()) != top))
                cells[f"{gname}|{c:g}"] = dict(kl=float(np.mean(kl)),
                                               qa=float(np.mean(qa)),
                                               flip=float(np.mean(fl)))
            pts.append(dict(t=int(t), p_top=float(p0[top]), cells=cells))
            if check and ti == len(ts) // 2:
                print("      이 지점의 노이즈 KL: " + ", ".join(
                    f"{k} {v['kl']:.2e}" for k, v in cells.items()))
        del base, K, V
        torch.cuda.empty_cache()
        return dict(ex_id=r["ex_id"], uid=r["uid"], think_len=r["think_len"],
                    fin=bool(r.get("fin", True)),
                    commit_feat=r.get("feat", {}).get("commit_step"),
                    n_prompt=npr, points=pts)

    os.makedirs(os.path.dirname(a.save) or ".", exist_ok=True)
    fo = open(a.save, "a" if a.resume else "w")
    todo = [r for r in rows if r["ex_id"] not in done]
    print(f"[데이터] {len(rows)}개 중 이미 저장 {len(rows) - len(todo)}개 → 이번에 {len(todo)}개")
    t0 = time.time()
    nskip = 0
    for i, r in enumerate(todo):
        out = run(r, check=(i == 0))
        if out is None:
            nskip += 1
            continue
        fo.write(json.dumps(out) + "\n")
        fo.flush()
        el = time.time() - t0
        print(f"  {i+1}/{len(todo)} ex{r['ex_id']} 지점 {len(out['points'])}  "
              f"경과 {el/60:.1f}분, 남은 예상 {el/(i+1)*(len(todo)-i-1)/60:.0f}분",
              end="\r", flush=True)
    fo.close()
    print(f"\n저장 → {a.save}   건너뜀(토큰 결손·지점 없음) {nskip}개")


if __name__ == "__main__":
    main()