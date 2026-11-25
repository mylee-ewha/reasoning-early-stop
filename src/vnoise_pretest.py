#!/usr/bin/env python3
"""vnoise_pretest.py — V 캐시 노이즈 섭동의 **사전 민감도 검사**.

왜 이걸 먼저 하나:
  섭동 축을 열려면 "노이즈를 넣었을 때 반응이 commit 과 상관있나"가 먼저다.
  상관이 없으면(z≈0) GPU 를 더 쓸 이유가 없고, 그 자체로 축 하나가 닫힌다.
  네 프로토콜 "도구 민감도를 먼저 재라"의 그 단계다.

측정 대상 (KL 은 '무엇과 무엇' 사이인가):
  같은 접두사에서 forced-close 한 뒤 **답 첫 토큰 위치의 다음토큰 분포** 두 개.
    p = clean   V 캐시 그대로
    q = noisy   지정 구간의 V 에만 노이즈를 더한 캐시
  KL(p‖q) 가 작다 = 노이즈를 먹여도 답 분포가 안 흔들린다 = 그 지점에서
  답이 '굳었다'. 가설: commit 이후 KL 이 commit 이전보다 작다.

왜 V 인가:
  K 에는 RoPE 가 걸려 있어 위치를 건드리면 내용 섭동과 분리가 안 된다.
  V 는 RoPE 가 없어 내용만 흔들 수 있다. 그리고 v = W_V · RMSNorm(h) 이므로
  덤프의 h 에서 바로 만들 수 있다(재prefill 불필요).

노이즈 스케일 (여기가 '어떤 수치로' 의 답):
  v 는 1024차원 = KV헤드 8 × head_dim 128 이고 헤드별 스케일이 다르다.
  그래서 전역 ‖v‖ 비례 노이즈는 못 쓴다. 헤드별 RMS 로 정규화한다.
      ε_head ~ N(0, (c · rms_head)²),  rms_head = sqrt(mean(v_head²))
  c 격자: 0.05 / 0.1 / 0.2 / 0.4 / 0.8   (사전등록. 결과 보고 늘리지 않는다)
  d2276 제외: 그 채널이 제곱노름의 99.1% 라 섞어 넣으면 c 가 사실상
  그 한 채널의 눈금이 된다. massive activation 채널 섭동은 **다른 개입**이므로
  --hit-sink 로 따로 돌린다.

사용:
  python vnoise_pretest.py --model <경로> --out out/sig_para500.jsonl \\
      --n 50 --layers 18 27 --cs 0.05,0.1,0.2,0.4,0.8
"""
import argparse
import json
import os

import copy

import numpy as np

from quartile_paired import commit_of, GRADER


def resolve_model_dir(m):
    import os
    if os.path.isdir(m):
        return m
    from huggingface_hub import snapshot_download
    return snapshot_download(m, local_files_only=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="data/sig_para500.jsonl")
    ap.add_argument("--n", type=int, default=50, help="예제 수")
    ap.add_argument("--layers", nargs="+", default=["18", "27", "all"],
                    help="V 를 흔들 층. '18'=단일, '18-27'=구간 동시, "
                         "'all'=전층 동시. 한 층만 흔들면 attention 평균에서 "
                         "희석돼 감도가 안 나온다(2026-11-15 확인)")
    ap.add_argument("--cs", default="0.1,0.25,0.5,1.0,2.0,4.0")
    ap.add_argument("--reps", type=int, default=3, help="c 당 노이즈 샘플 수")
    ap.add_argument("--hit-sink", action="store_true",
                    help="massive activation 채널도 흔든다(기본은 제외)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--script", default="src/run_commit_math.py",
                    help="수집 스크립트 경로. build_prompt 를 그대로 가져온다")
    ap.add_argument("--data-file", default="test.jsonl")
    ap.add_argument("--data-n", type=int, default=500,
                    help="수집 때 쓴 --n 과 **같아야** uid 집합이 맞는다")
    ap.add_argument("--dry-run", action="store_true",
                    help="모델을 안 올리고 접두사 복원·전후쌍 가능 여부만 센다. "
                         "GPU 태우기 전에 2초로 확인하는 용도")
    ap.add_argument("--save", default="vnoise_pretest.json")
    a = ap.parse_args()
    CS = [float(x) for x in a.cs.split(",")]

    global torch
    import torch

    from transformers import AutoModelForCausalLM, AutoTokenizer
    md = resolve_model_dir(a.model)
    tok = AutoTokenizer.from_pretrained(md)
    model = None
    if not a.dry_run:
        model = AutoModelForCausalLM.from_pretrained(
            md, dtype=torch.bfloat16, device_map="cuda").eval()
        nkv = model.config.num_key_value_heads
        hd = model.config.hidden_size // model.config.num_attention_heads
        print(f"[모델] 층 {model.config.num_hidden_layers}, "
              f"KV헤드 {nkv} × head_dim {hd} = {nkv*hd}차원")

    NL = model.config.num_hidden_layers if model is not None else 36
    GROUPS = []
    for spec in a.layers:
        if spec == "all":
            GROUPS.append(("all", list(range(NL))))
        elif "-" in spec:
            lo_, hi_ = spec.split("-")
            GROUPS.append((spec, list(range(int(lo_), int(hi_) + 1))))
        else:
            GROUPS.append((spec, [int(spec)]))
    print("[섭동 그룹] " + ",  ".join(f"{n}({len(v)}층)" for n, v in GROUPS))

    # 프롬프트는 수집 때와 **같은 함수**로 만든다. 여기서 chat template 을
    # 손으로 다시 쓰면 접두사가 1토큰이라도 달라지고 t 인덱스가 전부 밀린다
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(a.script)))
    from run_commit_math import build_prompt, load_math500, PROBE_ARMS
    CLOSE = PROBE_ARMS["A"]
    close_ids = tok(CLOSE, add_special_tokens=False).input_ids

    ds = load_math500(a.data_file, None, a.data_n)
    by_uid = {r.get("unique_id"): r for r in ds}
    rows = [json.loads(l) for l in open(a.out)]
    assert GRADER.startswith("grade2"), f"grade2 필요 — 현재 {GRADER}"
    CM = commit_of(rows)
    for r in rows:
        r.setdefault("feat", {})["commit_step"] = CM.get(r["ex_id"])   # 아래 세 곳이 그대로 grade2 값을 읽게
    rows = [r for r in rows if r.get("probes")
            and r.get("feat", {}).get("commit_step")
            and r.get("text") and r.get("uid") in by_uid]
    rows = rows[:a.n]
    print(f"[데이터] commit 있고 text 있는 예제 {len(rows)}개")
    if not rows:
        raise SystemExit("--save-text 로 수집한 jsonl 이 필요하다. "
                         "text 가 없으면 접두사를 복원할 수 없다")

    # 노이즈를 뺄 채널: 덤프에서 '제곱노름 지배' 채널을 h 쪽에서 찾았었다.
    # V 쪽에서도 같은 짓을 한다 — 헤드별로 제곱합 1등 채널을 뽑는다
    if a.dry_run:
        ok = defi = nopair = 0
        dl = []
        for r in rows:
            gids = tok(r["text"], add_special_tokens=False).input_ids
            d = r["gen_len"] - len(gids)
            dl.append(d)
            if d < 0 or d > max(8, int(0.01 * r["gen_len"])):
                defi += 1
                continue
            cs = r["feat"]["commit_step"]
            ts = sorted(p["t"] for p in r["probes"])
            if not [t for t in ts if t < cs] or not [t for t in ts if t >= cs]:
                nopair += 1
                continue
            ok += 1
        dl = np.array(dl)
        print(f"\n[dry-run] 예제 {len(rows)}개")
        print(f"  gen_len - 재토큰화 길이: 중앙 {np.median(dl):+.0f}  "
              f"최소 {dl.min()}  최대 {dl.max()}  "
              f"(</think> 1개만 빠지면 +1 이 정상)")
        print(f"  쓸 수 있음 {ok}   토큰 결손 과다 {defi}   전/후 쌍 없음 {nopair}")
        if ok == 0:
            print("\n  0개다. 결손 분포를 보고 가정을 다시 세워야 한다 —")
            print("  결손이 크면 text 가 생성 전체가 아닐 수 있다")
        else:
            print(f"\n  OK. --dry-run 빼고 돌려라 "
                  f"(forward {ok*4}회 + {ok*4*len(GROUPS)*len(CS)*a.reps}회)")
        return

    g = torch.Generator(device="cuda").manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    rec = []

    def gridstep(ts):
        """체크포인트 간격 중앙값. 문단 격자는 간격이 고르지 않다."""
        d = np.diff(ts)
        return float(np.median(d)) if len(d) else 1.0


    @torch.no_grad()
    def dist_at(cache):
        """캐시를 받아 forced-close 를 이어 넣고 답 첫 토큰 분포를 돌려준다."""
        ids = torch.tensor([close_ids], device="cuda")
        o = model(input_ids=ids, past_key_values=cache, use_cache=True)
        return torch.log_softmax(o.logits[0, -1].float(), -1)

    nskip = {"deficit": 0, "nopair": 0}
    align_log = []

    def gen_ids(r):
        """생성 토큰열을 text 에서 복원한다. None 이면 쓸 수 없는 예제.

        왜 필요한가: run_commit_math 가 text 를
          text = tok.decode(gen, skip_special_tokens=True)
        로 만들어서 특수토큰이 빠져 있다. 그래서 재토큰화 길이가 gen_len 과
        **항상** 다르고, 전체 길이 일치를 요구하면 전건 탈락한다
        (2026-11-15 에 50/50 탈락한 원인).

        빠지는 건 사실상 </think> 하나뿐이고 그건 thinking 끝(think_len)에
        있다. 우리가 쓰는 t 는 전부 그 앞이므로, 앞쪽 토큰열은 그대로다.
        그래서 전체를 한 번 토큰화하고 앞 t 개를 쓰면 **토큰 경계가 깨지지
        않는다** (문자 오프셋 이분탐색은 토큰 중간을 자른다).

        결손이 몇 개를 넘으면 가정이 깨진 것이므로 버린다.
        """
        gids = tok(r["text"], add_special_tokens=False).input_ids
        deficit = r["gen_len"] - len(gids)
        if deficit < 0 or deficit > max(8, int(0.01 * r["gen_len"])):
            return None, deficit
        return gids, deficit

    @torch.no_grad()
    def run(r):
        # 접두사 = 프롬프트 + 생성텍스트의 앞 t 토큰
        pids = build_prompt(tok, by_uid[r["uid"]])
        gids, deficit = gen_ids(r)
        align_log.append(deficit)
        if gids is None:
            nskip["deficit"] += 1
            return None
        npr = len(pids)
        ids = torch.tensor([list(pids) + list(gids)], device="cuda")
        cstar = r["feat"]["commit_step"]
        ts = sorted(p["t"] for p in r["probes"])
        pre = max([t for t in ts if t < cstar], default=None)
        post = min([t for t in ts if t >= cstar], default=None)
        if pre is None or post is None:
            nskip["nopair"] += 1
            return None
        # ── 위치 통제. pre/post 는 commit 을 사이에 둔 **인접** 체크포인트라
        #    위치 차이가 격자 한 칸뿐이다. 그래도 '인접한 두 칸은 원래 KL 이
        #    다르다' 를 배제해야 하므로 통제쌍을 하나 더 잡는다.
        #
        #    [2026-11-15 수정] 처음엔 commit 에서 '멀리 떨어진' 인접 쌍을
        #    잡았는데, commit 상대위치 중앙이 0.30 이라 그 쌍이 거의 항상
        #    **commit 이후**에 걸렸다. 그러면 양쪽 다 KL 바닥이고 비는
        #    0 나누기 0 이 된다 — 실제로 통제 비가 0.04~0.44 로 흩어졌다.
        #    통제쌍은 **둘 다 commit 이전**이어야 한다. 그래야 "굳기 전의
        #    인접한 두 칸은 비가 1.0" 과 대조가 된다.
        pr = [j for j in range(len(ts) - 1) if ts[j + 1] < cstar]
        out = {}
        tags = [("pre", pre), ("post", post)]
        if pr:
            j = int(pr[-1] if len(pr) == 1 else rng.choice(pr))
            tags += [("c0", ts[j]), ("c1", ts[j + 1])]
        def vslot(cache, L):
            return (cache.layers[L].values if hasattr(cache, "layers")
                    else cache.value_cache[L])

        for tag, t in tags:
            cut = min(npr + t, ids.shape[1])
            base = model(input_ids=ids[:, :cut], use_cache=True).past_key_values
            lp0 = dist_at(copy.deepcopy(base))
            p0 = lp0.exp()
            top = int(lp0.argmax())
            for gname, Ls in GROUPS:
                # 층별 스케일·마스크를 미리 잡아둔다 (rep 마다 다시 안 재게)
                prep = []
                for L in Ls:
                    v = vslot(base, L)[0]               # (nkv, T, hd)
                    rms = v.float().pow(2).mean(-1, keepdim=True).sqrt()
                    msk = None
                    if not a.hit_sink:
                        k = v.float().pow(2).sum(1).argmax(-1)  # 헤드별 지배 채널
                        msk = torch.ones_like(v, dtype=torch.bool)
                        for h in range(v.shape[0]):
                            msk[h, :, k[h]] = False
                    prep.append((L, v.shape, rms, msk))
                for c in CS:
                    kl, qa, flip = [], [], []
                    for _ in range(a.reps):
                        cc = copy.deepcopy(base)
                        for L, shp, rms, msk in prep:
                            eps = torch.randn(shp, generator=g, device="cuda",
                                              dtype=torch.float32) * (c * rms)
                            if msk is not None:
                                eps = eps.masked_fill(~msk, 0.0)
                            vslot(cc, L)[0] += eps.to(vslot(cc, L).dtype)
                        lp1 = dist_at(cc)
                        kl.append(float((p0 * (lp0 - lp1)).sum()))
                        # KL 은 p0 가 거의 델타면 −log q(top) 로 수렴해 바닥에
                        # 깔린다. q(top) 와 argmax 변경 여부를 같이 남겨서
                        # '기구가 안 먹는 것'과 '굳어서 안 흔들리는 것'을 구별
                        qa.append(float(lp1[top].exp()))
                        flip.append(float(int(lp1.argmax()) != top))
                    out[(tag, gname, c)] = dict(
                        kl=float(np.mean(kl)), qa=float(np.mean(qa)),
                        flip=float(np.mean(flip)))
        return out

    for i, r in enumerate(rows):
        o = run(r)
        if o is None:
            continue
        rec.append(dict(ex_id=r["ex_id"], think_len=r["think_len"],
                        commit=r["feat"]["commit_step"],
                        kl={f"{k[0]}|{k[1]}|{k[2]}": v for k, v in o.items()}))
        print(f"  {i+1}/{len(rows)} ex{r['ex_id']}", end="\r", flush=True)
    print()

    json.dump(rec, open(a.save, "w"))
    print(f"저장 → {a.save}  ({len(rec)}개)")
    print(f"  건너뜀: 토큰 결손 과다 {nskip['deficit']}, "
          f"전/후 쌍 없음 {nskip['nopair']}")
    if align_log:
        al = np.array(align_log)
        print(f"  gen_len - 재토큰화 길이: 중앙 {np.median(al):+.0f}, "
              f"최소 {al.min()}, 최대 {al.max()}  "
              f"(</think> 1개만 빠지면 +1 이 정상)")
    if not rec:
        raise SystemExit("측정된 예제가 0개다 — 위 건너뜀 사유를 볼 것")
    print()

    # ── 판정. 예제내 쌍이므로 부호검정. 통제쌍과 같이 본다
    from math import comb

    def sgn(pairs):
        n01 = sum(1 for x, y in pairs if x > y)
        n10 = sum(1 for x, y in pairs if y > x)
        n = n01 + n10
        k = min(n01, n10)
        p = (min(1.0, 2*sum(comb(n, i) for i in range(k+1))/2**n) if n else 1.0)
        return n01, n10, p

    def grab(t0, t1, L, c, f="kl"):
        k0, k1 = f"{t0}|{L}|{c}", f"{t1}|{L}|{c}"
        return [(x["kl"][k0][f], x["kl"][k1][f]) for x in rec
                if k0 in x["kl"] and k1 in x["kl"]]

    m_tests = len(GROUPS) * len(CS)
    bf = 0.05 / m_tests
    print(f"검정 {m_tests}개 → Bonferroni {bf:.4f}")
    print("먼저 **감도**를 본다. pre 쪽 q(top) 가 1.000 에 붙어 있으면 "
          "노이즈가 답을 못 흔든 것이고,\nKL 이 작은 게 '굳어서'가 아니라 "
          "'기구가 안 먹어서'다 — 그 줄은 해석하지 않는다.\n")
    print("통제쌍은 **둘 다 commit 이전**인 인접 두 지점이다. commit 이 첫"
          " probe 지점에 있는\n예제는 그 쌍을 못 만들므로 통제 n 이 작다 —"
          " n 과 함께 볼 것.\n")
    print("=" * 128)
    print("섭동군  c      pre q(top)  pre flip   pre KL     post KL  "
          " 비 post/pre  부호(pre>post)      p   n │ 통제비  통제p   n")
    print("=" * 128)
    for gname, _ in GROUPS:
        for c in CS:
            pq = grab("pre", "post", gname, c)
            if not pq:
                continue
            qq = grab("pre", "post", gname, c, "qa")
            ff = grab("pre", "post", gname, c, "flip")
            n01, n10, p = sgn(pq)
            mp = np.median([x for x, _ in pq]); mq = np.median([y for _, y in pq])
            mqa = np.median([x for x, _ in qq])
            mfl = np.mean([x for x, _ in ff])
            cq = grab("c0", "c1", gname, c)
            if cq:
                _, _, cp = sgn(cq)
                cm0 = np.median([x for x, _ in cq])
                cm1 = np.median([y for _, y in cq])
                ctl = f"{cm1/max(cm0,1e-12):>6.2f}  {cp:.4f} {len(cq):>3}"
            else:
                ctl = "     -       -   0"
            dead = "  (무반응)" if mqa > 0.999 else ""
            print(f"{gname:>6} {c:<6.2f} {mqa:>10.4f} {mfl*100:>8.1f}% "
                  f"{mp:>9.4f} {mq:>10.4f} {mq/max(mp,1e-12):>11.3f}  "
                  f"{n01:>4}:{n10:<4} {p:>8.4f}"
                  f"{'**' if p < bf else '  '} {len(pq):>3} │ {ctl}{dead}")
    print()
    print("판정 기준 (사전등록)")
    print("  [0순위: 감도] pre q(top) 가 0.99 미만으로 내려간 c 가 있어야 한다.")
    print("    1.000 에 붙어 있으면 노이즈가 답에 닿지 않은 것 — 그 줄은 버린다.")
    print("    단일 층은 attention 평균에서 1/sqrt(T) 로 희석되므로 'all' 을 볼 것")
    print("  [열림] 감도 있는 c 에서 post/pre < 0.8, 부호검정 Bonferroni 통과,")
    print("         **그리고 통제쌍 비는 1.0 근처** → 효과가 위치가 아니라 commit 이다")
    print("  [교란] 통제쌍 비도 같이 내려가면 → 그냥 '뒤로 갈수록 KL 이 준다'다.")
    print("         commit 과 무관. 이 경우 잔여길이 교란과 같은 종류의 문제다")
    print("  [닫힘] 비가 1.0 근처거나 부호가 뒤집히면 → V 노이즈 반응은 commit 과")
    print("         무관. 축을 닫고 음성 결과로 보고한다")
    print("  [무반응] 어느 c 에서도 pre KL < 0.01 이면 도구가 안 먹는 것이다.")
    print("           c 를 키우기 전에 노이즈가 캐시에 실제로 들어갔는지 확인")
    print("  [포화]  가장 큰 c 에서 pre/post 가 둘 다 > 1.0 이면 과섭동이다.")
    print("          구분력이 아니라 파괴력을 재고 있는 것이므로 c 를 낮춰서 다시")


if __name__ == "__main__":
    main()