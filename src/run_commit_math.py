"""
조기 확신 탐지 — MATH-500 / 중간 답 읽기(probe) 방식 + 내부 신호 기록

왜 벤치마크를 바꾸나
    MuSiQue 에서 p_end 궤적을 실측해 보니 멈추기 전 구간이 1e-9 이하,
    최댓값이 스텝 0 에 있었다. </think> 로짓은 계단이고 사전 정보가 없다.
    "언제 멈출지" 를 재는 신호로는 1비트짜리다.

    재야 할 것은 "언제 답을 정했는지" 다. 그건 </think> 로짓이 아니라
    모델의 현재 답 자체에 있다. 그래서 think 블록 중간에서 캐시를 포크해
    답을 직접 꺼내 읽는다. 이 측정에는 증거 스팬도 hop 구조도 필요 없고,
    필요한 것은 긴 CoT 와 검증 가능한 답뿐이다 — MATH-500 이 그 조건이다.

핵심 측정
    think 블록을 생성하는 동안 --probe-every 토큰마다
        \\n</think>\\n\\nFinal Answer: \\boxed{
    를 붙여 답을 그리디로 읽고, 캐시를 되돌려 본 생성을 그대로 이어간다.
    probe 는 본 생성에 흔적을 남기지 않는다 (T4 가 이것을 검사한다).

    a_1..a_m  각 체크포인트의 중간 답
    commit_step   a_j 부터 끝까지 답이 안 바뀌고 최종 답과 같은 최초 지점
    commit_frac   commit_step / think 길이      ← 낮으면 조기 확신
    n_flips       중간 답이 바뀐 횟수

내부 신호 (--signals)
    본 생성 스텝에서 forward 산출물을 그대로 읽는다. probe 와 달리 포크가
    없으므로 배포 가능한 신호의 후보다.

    sdpa  : hidden / logit 계열 전부 (h_norm, h_delta, h_cos, dl_q*,
            tok_ent, top1, margin, top5_mass, layer_kl, layer_agree,
            layer_ent_mean, layer_ent_drop)
            output_hidden_states 는 이미 계산된 텐서를 돌려줄 뿐이라 공짜.
            비싼 것은 logit lens 뿐이므로 --sig-layers 로 제한한다.
    eager : 위 전부 + attention 계열 (head_jsd, attn_ent, attn_ent_norm,
            attn_sink). 다만 sdpa 와는 부동소수점 연산 순서가 달라
            **생성 토큰열 자체가 갈린다**(실측 111스텝에서 분기).
            기존 sdpa jsonl 과 교차 확인하려면 sdpa 로 돌릴 것.

이 한 번의 실행이 이득 상한을 준다 (2차 실행 불필요)
    조기 종료 상한 = P(어느 체크포인트의 답이 정답 | 최종 답이 오답)
    probe 가 이미 "그 시점에 멈췄다면 뭐라고 답했을까" 를 다 갖고 있으므로,
    어떤 조기 종료 탐지기도 이 값을 넘을 수 없다. 공짜 oracle 이다.
    이 값이 낮으면 탐지기를 완벽하게 만들어도 얻을 게 없으니 거기서 접는다.

두 실패 방향
    locked_wrong      오답에 일찍 고정 — 더 생각해야 하는 쪽
    right_abandoned   정답을 거쳐갔으나 최종 오답 — 덜 생각해야 하는 쪽
    두 방향의 처방이 반대라서, 섞어서 세면 둘 다 안 보인다.

2026-11 재수집 — 왜 다시 돌리나
    `commit_step` 이 모든 숫자의 라벨인데(AUC, z, 오라클, 사분위 결과 전부),
    probe 접미사를 하나만 써서 그 라벨이 **표현에 의존하는지 모른다**.
    Stable Answers §4.2 는 체인 앞쪽 1/10 에서 표현만 바꾸면 54%가 다른 답을
    낸다고 보고한다. 우리 commit_frac 중앙이 0.268 이라 정확히 그 사정권이다.
    같은 체크포인트에서 접미사 4종(--arms)을 포크해 읽어 이것을 검사한다.
    probe 지점도 `\\n\\n` 경계로 옮기고(--probe-mode para), 덤프 레이어를
    줄여(--dump-layers) 용량을 35GB → 1~2GB 로 낮춘다.

실행
    # 스모크 — T4/T7 PASS 와 팔별 답, 1건 소요 시간을 먼저 확인할 것.
    # 팔이 늘면 포크/롤백도 4배라 crop 이 틀리면 밤새 돌린 게 전부 거짓이 된다
    python run_commit_math.py --model Qwen/Qwen3-8B --data-file math500.jsonl \\
        --n 3 --max-new 8192 --probe-mode para --arms A,B,C,D \\
        --signals --sig-every 32 --log-every 16 \\
        --dump-hidden --dump-layers 9 18 27 31 35 36 --out smoke.jsonl

    # 본 실행 (200건 먼저, --resume 으로 이어붙이기)
    python run_commit_math.py --model Qwen/Qwen3-8B --data-file math500.jsonl \\
        --n 200 --max-new 8192 --probe-mode para --arms A,B,C,D \\
        --signals --sig-every 32 --log-every 16 \\
        --dump-hidden --dump-layers 9 18 27 31 35 36 \\
        --resume --out sig_para200.jsonl

    python run_commit_math.py --analyze sig_para200.jsonl
    python run_commit_math.py --rescore sig_para200.jsonl   # 채점기만 고쳐 재계산

    # 기존 실행을 그대로 재현하려면
    #   --arms A --probe-mode grid --probe-every 256 --dump-layers (빈 값)

하위 호환
    r["probes"] 와 r["feat"]["commit_step"] 은 **기준 팔 A** 기준으로 그대로
    둔다. probe_commit.py / quartile_paired.py 가 이 두 필드를 읽으므로
    모양이 바뀌면 전부 깨진다. 새 데이터는 r["probes_arms"] 와
    feat["commit_arms"] 로 따로 들어간다.
    덤프 메타(.meta.jsonl)는 이제 행마다 para/grid 플래그를 갖고,
    레이어 목록은 .dumpinfo.json 에 있다 — 다운스트림에서
    [행, len(dump_layers), hidden] 으로 reshape 할 때 필요하다.
"""
import argparse, json, math, os, re
import statistics as st
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

PROBE_PREFIX = "\n</think>\n\nFinal Answer: \\boxed{"
PROMPT_SUFFIX = ("\n\nPlease reason step by step, and put your final answer "
                 "within \\boxed{}.")

# ──────────────────────── probe 팔 (2026-11 추가) ────────────────────────
# 같은 체크포인트에서 캐시를 포크해 접미사만 바꿔 여러 번 읽는다.
#   A  현행. 기존 500건 분석 전부와 이어지는 다리이므로 절대 바꾸지 않는다
#   B  같은 의도, 다른 표현.  A vs B = 순수 wording 검사
#      (Stable Answers §4.2: 체인 앞쪽 1/10 에서 표현만 바꿔도 54%가 다른 답)
#   C  적대적. "틀렸다고 밀어넣어도 안 바뀌는 답" = 더 엄격한 정착 정의
#   D  C 와 길이를 맞춘 중립 대조. C−D 만이 '틀렸다는 암시'의 내용 효과다
#      (SABER 에 없는 팔. 길이 맞춘 대조군이 없으면 "긴 문구를 넣으니 달라졌다"
#       와 구분이 안 된다)
PROBE_ARMS = {
    "A": "\n</think>\n\nFinal Answer: \\boxed{",
    "B": "\n</think>\n\nThe answer is \\boxed{",
    "C": ("\n</think>\n\nWait, my reasoning above may be wrong. "
          "Let me reconsider. The answer is \\boxed{"),
    "D": ("\n</think>\n\nWait, my reasoning above is written out. "
          "Let me summarize it. The answer is \\boxed{"),
}
PRIMARY_ARM = "A"

# 토큰 id → 문자열 캐시. 문단 경계 검출과 문자 수 세기에 쓴다.
# 매 스텝 decode 를 부르면 느리므로 한 번 만든 걸 재사용한다
_TOKSTR = {}


def tokstr(tok, i):
    s = _TOKSTR.get(i)
    if s is None:
        try:
            s = tok.convert_tokens_to_string([tok.convert_ids_to_tokens(i)])
        except Exception:
            s = tok.decode([i], skip_special_tokens=True)
        _TOKSTR[i] = s
    return s


# ──────────────────────────── 데이터 ────────────────────────────
def load_math500(data_file=None, hf_name=None, n=500, seed=0, levels=None):
    rows = None
    if data_file and os.path.exists(data_file):
        rows = [json.loads(l) for l in open(data_file) if l.strip()]
    else:
        try:
            from datasets import load_dataset
            d = load_dataset(hf_name or "HuggingFaceH4/MATH-500", split="test")
            rows = [dict(r) for r in d]
        except Exception as e:
            raise FileNotFoundError(
                f"MATH-500 을 못 읽었다 ({type(e).__name__}: {e}).\n"
                "  파일로 받으려면:\n"
                "    huggingface-cli download HuggingFaceH4/MATH-500 "
                "test.jsonl --repo-type dataset --local-dir .\n"
                "  그 다음 --data-file test.jsonl")
    need = ("problem", "answer")
    bad = [k for k in need if k not in rows[0]]
    if bad:
        raise KeyError(f"필수 필드 없음: {bad}. 가진 필드: {list(rows[0])}")
    if levels:
        rows = [r for r in rows if r.get("level") in levels]
    if n and n < len(rows):
        import random
        random.Random(seed).shuffle(rows)
        rows = rows[:n]
    return rows


def build_prompt(tok, ex):
    msgs = [{"role": "user", "content": ex["problem"] + PROMPT_SUFFIX}]
    return tok.apply_chat_template(msgs, tokenize=True,
                                   add_generation_prompt=True,
                                   enable_thinking=True)


# ──────────────────────── 답 추출·정규화 ────────────────────────
def close_brace(s):
    """\\boxed{ 가 이미 열린 상태의 뒷부분 s 에서, 짝이 맞는 닫는 중괄호까지.

    첫 번째 '}' 에서 끊으면 \\frac{1}{2} 가 '\\frac{1' 로,
    \\begin{pmatrix}...\\end{pmatrix} 가 '\\begin{pmatrix' 로 잘린다.
    아직 안 닫혔으면 None.
    """
    d = 1
    for i, ch in enumerate(s):
        if ch == "{":
            d += 1
        elif ch == "}":
            d -= 1
            if d == 0:
                return s[:i]
    return None


def last_boxed(s):
    """마지막 \\boxed{...} 의 내용. 중괄호 깊이를 세서 중첩을 버티게 한다."""
    i = s.rfind("\\boxed")
    if i < 0:
        i = s.rfind("\\fbox")
        if i < 0:
            return None
    j = s.find("{", i)
    if j < 0:
        # \boxed 12 처럼 중괄호 없는 형태
        m = re.match(r"\\boxed\s*([^\s$,.]+)", s[i:])
        return m.group(1) if m else None
    d = 0
    for k in range(j, len(s)):
        if s[k] == "{":
            d += 1
        elif s[k] == "}":
            d -= 1
            if d == 0:
                return s[j + 1:k]
    return s[j + 1:]            # 닫히지 않음 (probe 가 잘린 경우)


# 순서가 중요하다. "$" 를 "\\$" 보다 먼저 지우면 \$18.90 이 \18.90 이 된다.
# 긴 패턴부터 지울 것.
_RM = ["\\left", "\\right", "\\quad", "\\qquad", "\\!", "\\,", "\\;", "\\:",
       "\\$", "\\%", "\\ ", "$", "%", "~"]


def norm_math(s):
    """표기 차이만 지운다. 수학적 동치 판정은 하지 않는다 (sympy 는 eq_math 에서)."""
    if s is None:
        return None
    s = str(s).strip()
    if "\\boxed" in s or "\\fbox" in s:
        s = last_boxed(s) or s
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mbox\s*\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\^\s*\{?\\circ\}?", "", s)
    s = re.sub(r"\\mathrm\s*\{([^}]*)\}", r"\1", s)
    for r in _RM:
        s = s.replace(r, "")
    # \frac ab → \frac{a}{b},  \sqrt a → \sqrt{a}
    s = re.sub(r"\\frac(?!\{)\s*([^{\s])\s*([^{\s])", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\sqrt(?!\{)\s*([^{\s])", r"\\sqrt{\1}", s)
    s = s.replace(" ", "").replace("\n", "").rstrip(".")
    s = re.sub(r"^\{(.*)\}$", r"\1", s)          # 전체를 감싼 중괄호
    s = re.sub(r"(\d),(\d{3})\b", r"\1\2", s)    # 1,000 → 1000
    if re.fullmatch(r"-?\.\d+", s):              # .5 → 0.5
        s = s.replace(".", "0.", 1)
    if re.fullmatch(r"-?\d+\.0+", s):            # 3.0 → 3
        s = s.split(".")[0]
    return s.lower()


_NUM = r"-?\d+(?:\.\d+)?"


def to_float(s):
    """정규화된 문자열을 수치로. \\frac{1}{2} 와 0.5 를 같게 보기 위한 최소 평가기.

    sympy 의 parse_latex 는 antlr 런타임이 없으면 조용히 실패하므로
    (설치돼 있어도 환경마다 다르다) 의존성 없는 경로를 먼저 둔다.
    """
    if s is None:
        return None
    s = s.strip()
    try:
        return float(s)
    except Exception:
        pass
    m = re.fullmatch(rf"\\frac\{{({_NUM})\}}\{{({_NUM})\}}", s)
    if m and float(m.group(2)) != 0:
        return float(m.group(1)) / float(m.group(2))
    m = re.fullmatch(rf"({_NUM})/({_NUM})", s)
    if m and float(m.group(2)) != 0:
        return float(m.group(1)) / float(m.group(2))
    m = re.fullmatch(rf"(-?)\\sqrt\{{({_NUM})\}}", s)
    if m and float(m.group(2)) >= 0:
        v = math.sqrt(float(m.group(2)))
        return -v if m.group(1) else v
    return None


def mat2tuple(s):
    """\\begin{pmatrix}a\\\\b\\end{pmatrix} → (a,b).

    probe 는 \\boxed{ 를 열어 두고 읽으므로 벡터를 튜플로 쓰는 반면
    최종 답은 행렬 환경으로 쓰는 경우가 많다. 표기만 다른데 '답이 바뀌었다'
    로 집계되면 any_probe_gold 와 commit_step 이 통째로 거짓이 된다.
    """
    if s is None:
        return None
    m = re.search(r"\\begin\{[pbvB]?matrix\}(.*?)\\end\{[pbvB]?matrix\}",
                  str(s), re.S)
    if not m:
        return s
    parts = [p.strip() for p in re.split(r"\\\\", m.group(1)) if p.strip()]
    return "(" + ",".join(parts) + ")" if parts else s


def eq_math(a, b, use_sympy=True):
    if _eq_raw(a, b, use_sympy):
        return True
    ta, tb = mat2tuple(a), mat2tuple(b)
    return (ta, tb) != (a, b) and _eq_raw(ta, tb, use_sympy)


def _eq_raw(a, b, use_sympy=True):
    na, nb = norm_math(a), norm_math(b)
    if na is None or nb is None:
        return False
    if na == nb:
        return True
    fa, fb = to_float(na), to_float(nb)
    if fa is not None and fb is not None:
        return abs(fa - fb) < 1e-6
    if use_sympy:
        try:
            from sympy.parsing.latex import parse_latex
            import sympy
            d = sympy.simplify(parse_latex(na) - parse_latex(nb))
            return bool(d == 0)
        except Exception:
            pass
    return False


def final_answer(text):
    """본 생성에서 모델의 최종 답. </think> 이후 구간의 마지막 boxed."""
    tail = text.rsplit("</think>", 1)[1] if "</think>" in text else text
    b = last_boxed(tail)
    if b is not None:
        return b
    lines = [l.strip() for l in tail.splitlines() if l.strip()]
    if not lines:
        return None
    s = lines[-1].replace("*", "").replace("#", "")
    s = re.sub(r"^\s*(final\s*answer|answer|the\s+answer\s+is)\s*[:\-–]?\s*",
               "", s, flags=re.I)
    return s.strip(" .") or None


# ──────────────────────── 캐시 롤백 ────────────────────────
def rollback(cache, target):
    """캐시를 target 길이로 정확히 되돌린다.

    crop 의 인자 의미가 버전마다 다르다.
      transformers 4.x : crop(max_length)      양수=남길 길이, 음수=제거 개수
      transformers 5.x : crop(tokens_to_remove) 음수=제거 개수,
                         양수는 deprecated 로 '남길 길이' 취급
    두 버전에서 같은 뜻을 갖는 형태는 **음수** 하나뿐이라 그것만 쓴다.
    그래도 매번 길이를 확인한다 — 여기가 틀리면 probe 가 본 생성을 오염시켜
    모든 숫자가 조용히 거짓이 된다.
    """
    cur = cache.get_seq_length()
    if cur <= target:
        return
    cache.crop(-(cur - target))
    got = cache.get_seq_length()
    if got != target:                      # 최후의 대비: 양수 keep 의미
        cache.crop(target)
        got = cache.get_seq_length()
    if got != target:
        import transformers
        raise RuntimeError(
            f"캐시 롤백 실패: {cur} → {got}, 목표 {target}. "
            f"transformers {transformers.__version__} 의 crop 의미가 "
            f"예상과 다르다. probe 를 끄고(--probe-every 0) 돌리거나 "
            f"버전을 확인할 것")


# ──────────────────────── 내부 신호 ────────────────────────
def _ent(p):
    return -(p * torch.log(p.clamp_min(1e-9))).sum(-1)


# MLP 중간활성(down_proj 입력, intermediate_size 차원)을 받아 두는 전역 버퍼.
# forward 중에 채워지고, 신호를 뽑는 시점에 읽는다. probe 는 신호를 뽑은
# **뒤에** 돌므로 probe 의 활성이 섞이지 않는다 (순서가 바뀌면 깨진다)
_MLP = {}


def attach_mlp_hooks(model):
    hs = []
    for i, lyr in enumerate(model.model.layers):
        def mk(i):
            def pre(mod, inp):
                _MLP[i] = inp[0]
            return pre
        hs.append(lyr.mlp.down_proj.register_forward_pre_hook(mk(i)))
    return hs


@torch.no_grad()
def mlp_snapshot(nl, proj):
    """레이어별 MLP 중간활성의 (random projection, 희소성 통계).

    원본은 intermediate_size(Qwen3-8B 는 12288)라 그대로 저장하면 n=500 에
    44GB 가 된다. SiE 에서 쓴 random projection 으로 128 차원에 넣으면
    방향 정보를 유지한 채 0.5GB 로 줄어든다 (Johnson-Lindenstrauss).
    통계는 NAD(2510.26277)가 쓴 '활성 뉴런 수' 계열이다.
    """
    P, st_ = [], []
    for i in range(nl):
        a = _MLP.get(i)
        if a is None:
            P.append(np.zeros(proj.shape[1], np.float16))
            st_.append(np.zeros(5, np.float32))
            continue
        v = a[0, -1].float()                      # 마지막 위치
        P.append((v @ proj).cpu().numpy().astype(np.float16))
        av = v.abs()
        s = av.sum()
        p = av / s.clamp_min(1e-9)
        k = min(32, av.numel())
        st_.append(np.array([
            float((av > 1e-3).float().mean()),    # 활성 뉴런 비율
            float(s), float(v.norm()),
            float(av.topk(k).values.sum() / s.clamp_min(1e-9)),   # top-32 질량
            float(-(p * torch.log(p.clamp_min(1e-12))).sum()),    # 활성 엔트로피
        ], np.float32))
    return np.stack(P), np.stack(st_)


@torch.no_grad()
def internal_signals(model, att, hs, layers, prev_h=None):
    """forward 산출물 하나에서 뽑을 수 있는 내부값 전부.

    att 가 None 이면(sdpa) attention 계열만 건너뛴다. hidden / logit 계열은
    sdpa 에서 전부 계산되고, output_hidden_states 는 이미 만들어진 텐서를
    돌려줄 뿐이라 추가 연산이 없다. 비싼 것은 logit lens 뿐이므로
    (lm_head 를 레이어마다 적용) layers 로 제한한다.

    반환: (신호 dict, 이번 스텝의 마지막 레이어 hidden)
          두 번째 값을 다음 호출의 prev_h 로 넘기면 Δh 가 인접 스텝 차이가 된다.
    """
    out = {}
    h = hs[-1][0, -1].float()

    # ── 상태의 크기와 변화. 루프드 LM(2510.25741)은 Δh 임계만으로
    #    학습된 게이트와 거의 동등한 성능을 냈다
    out["h_norm"] = float(h.norm())
    if prev_h is not None:
        out["h_delta"] = float((h - prev_h).norm())
        out["h_cos"] = float(torch.nn.functional.cosine_similarity(
            h, prev_h, dim=0))
        out["h_delta_rel"] = out["h_delta"] / (out["h_norm"] + 1e-9)

    # ── 레이어별 업데이트 크기: 어느 깊이가 일하고 있나
    nl = len(hs) - 1
    for fr, nm in ((.25, "q1"), (.5, "q2"), (.75, "q3")):
        i = max(1, int(nl * fr))
        a, b = hs[i - 1][0, -1].float(), hs[i][0, -1].float()
        out["dl_" + nm] = float((b - a).norm() / (a.norm() + 1e-9))

    # ── logit lens: 중간 레이어에 최종 레이어용 lm_head 를 갖다 대서
    #    "이 깊이에서는 무엇을 쓰려 했나" 를 읽는다
    # h 는 노름·코사인 정밀도를 위해 float32 로 들고 있으므로, lm_head 에
    # 넣기 직전에 모델 dtype(bf16)으로 되돌린다. 결과는 다시 float32

    # hs[-1] 은 HF Qwen 구현에서 이미 model.model.norm() 을 통과한 상태다
    # (레이어 노름이 1462 → 165 로 꺾이는 것이 그 증거, lm_head(hs[-1]) 과
    #  모델 자신의 logits 가 bf16 오차 0.06 로 일치함을 확인했다).
    # 거기에 norm 을 또 걸면 RMSNorm 가중치가 두 번 곱해져 분포가 왜곡된다.
    # 중간 레이어(hs[-4..-2])는 norm 전이므로 lens 가 맞다 — 둘을 분리한다.
    _dt = model.lm_head.weight.dtype
    lens = lambda x: model.lm_head(model.model.norm(x.to(_dt))).float()  # 중간층
    fin = torch.softmax(model.lm_head(h.to(_dt)).float(), -1)            # 최종층
    top = torch.topk(fin, 5)
    out["tok_ent"] = float(_ent(fin))          # 다음 토큰 선택의 망설임
    out["top1"] = float(top.values[0])
    out["margin"] = float(top.values[0] - top.values[1])
    out["top5_mass"] = float(top.values.sum())
    fin_id = int(top.indices[0])

    kl = agree = 0.0
    lay_ent = []
    for li in layers[:-1]:
        q = torch.softmax(lens(hs[li][0, -1].float()), -1)
        kl += float((q * (q.clamp_min(1e-9).log()
                          - fin.clamp_min(1e-9).log())).sum())
        agree += float(int(q.argmax()) == fin_id)
        lay_ent.append(float(_ent(q)))
    m = max(len(layers) - 1, 1)
    out["layer_kl"] = kl / m                   # 깊이 방향 수렴 (작으면 일찍 정해짐)
    out["layer_agree"] = agree / m             # 중간층이 최종과 같은 토큰을 골랐나
    if lay_ent:
        out["layer_ent_mean"] = sum(lay_ent) / len(lay_ent)
        out["layer_ent_drop"] = lay_ent[0] - out["tok_ent"]   # 깊이 따라 확신이 붙는 폭

    # ── attention 계열 (eager 에서만 얻을 수 있다)
    if att is not None:
        js = en = sink = 0.0
        for li in layers:
            p = att[li][0, :, -1, :].float()           # (H, n) 디코드라 쿼리 1개
            js += float(_ent(p.mean(0)) - _ent(p).mean())
            en += float(_ent(p).mean())
            sink += float(p[:, :4].sum(-1).mean())     # 앞 4토큰 질량(sink)
        n = len(layers)
        out["head_jsd"] = js / n               # head 들이 같은 곳을 보나 (0=합의)
        out["attn_ent"] = en / n
        # attn_ent 는 길이 n 벡터의 엔트로피라 n 이 커지면 상한 자체가 올라간다
        # (실측 corr(log n)=+0.813). log n 으로 나눈 값도 같이 남겨 비교한다
        out["attn_ent_norm"] = out["attn_ent"] / math.log(att[-1].shape[-1] + 1)
        out["attn_sink"] = sink / n
    return out, h


# ──────────────────────── 생성 + probe ────────────────────────
@torch.no_grad()
def generate_probed(model, tok, ids, *, max_new, device, eos_ids, end_ids,
                    probe_ids, probe_every, probe_tokens, log_every=1,
                    want_signals=False, sig_every=0, sig_layers=None,
                    use_att=False, dump=False, proj=None,
                    arm_ids=None, probe_mode="grid", para_min_chars=200,
                    para_max_chars=1200, dump_layers=None, dump_mlp=False):
    """arm_ids: {팔이름: 접두 토큰열}. None 이면 probe_ids 하나만 쓴다(기존 동작).

    probe_mode
      grid : (s+1) % probe_every == 0        — 기존 동작, 재현용
      para : `\\n\\n` 경계에서, 직전 probe 이후 para_min_chars 이상 쌓였을 때
             (PUMA 입도. 생 `\\n\\n` 은 문단당 ~53토큰이라 probe 하나당 너무 짧다)
             para_max_chars 를 넘기면 경계가 없어도 발화 — 긴 문단이 통째로
             probe 없이 지나가는 것을 막는다
    """
    cache = DynamicCache()
    sig_layers = sig_layers or [-4, -3, -2, -1]
    end_set, eos_set = set(end_ids), set(eos_ids)
    if arm_ids is None:
        arm_ids = {PRIMARY_ARM: probe_ids} if probe_ids else {}
    # dump: 체크포인트마다 **선택한** 레이어 hidden 을 모아 둔다.
    # 37층 전부 저장하면 n=500 에 35GB 가 되고 실제로 쓰는 건 세 층뿐이었다.
    # Q/K/V 는 hidden 의 선형변환이라 사후 복원이 가능하므로 따로 안 뽑는다
    H, MP, MS, DFLAG = [], [], [], []

    def step(inp, want_att=False, want_hs=False):
        kv = cache.get_seq_length()
        pos = torch.arange(kv, kv + inp.shape[1], device=device)[None]
        return model(input_ids=inp, past_key_values=cache, use_cache=True,
                     position_ids=pos, cache_position=pos[0],
                     output_attentions=want_att,
                     output_hidden_states=want_hs)

    def probe_one(pids):
        # probe 안에서는 신호를 뽑지 않는다 (기본값 False). 본 생성의 prev_h 도
        # 건드리지 않으므로 Δh 가 오염되지 않는다
        o = step(torch.tensor([pids], device=device))
        got, txt = [], ""
        for _ in range(probe_tokens):
            t = int(o.logits[0, -1].argmax())
            if t in eos_set or t in end_set:
                break
            got.append(t)
            txt = tok.decode(got, skip_special_tokens=True)
            if close_brace(txt) is not None:
                break
            o = step(torch.tensor([[t]], device=device))
        cut = close_brace(txt)
        return (cut if cut is not None else txt).strip()

    def probe_all():
        """팔마다 포크 → 읽기 → **즉시 롤백**. 길이 검증은 rollback 안에 있다.

        팔이 늘어나면 포크/롤백도 그만큼 늘어난다. 한 팔이라도 롤백이 어긋나면
        본 생성이 오염되어 그 뒤 모든 숫자가 조용히 거짓이 되므로,
        팔 사이마다 복원하고 매번 길이를 확인한다.
        """
        L = cache.get_seq_length()
        res = {}
        for nm, pids in arm_ids.items():
            res[nm] = probe_one(pids)
            rollback(cache, L)
        return res

    out = step(torch.tensor([ids], device=device), want_hs=want_signals)
    prev_h = out.hidden_states[-1][0, -1].float() if want_signals else None
    # prefill 마지막 토큰 = CoT 를 시작하기 **전** 상태. 2603.01437 이
    # "추론 시작 전에 답이 이미 표상돼 있다" 를 본 지점이다
    # want_signals 가 꺼져 있으면 hidden_states 가 None 이다. 덤프도 신호도
    # 없는 경로(T4 의 clean 팔)에서 여기서 터지면 안 된다
    nhs = len(out.hidden_states) if out.hidden_states is not None else 0
    dl = dump_layers if dump_layers else list(range(nhs))
    prefill_h = (np.stack([out.hidden_states[li][0, -1]
                           .to(torch.float16).cpu().numpy() for li in dl])
                 if (dump and nhs) else None)
    gen, trace, sigs, fin = [], [], [], False
    probes = {nm: [] for nm in arm_ids}          # 팔별 [{t, ans}]
    stop_k = None
    para_chars, prev_piece, n_para = 0, "", 0

    for s in range(max_new):
        lg = out.logits[0, -1].float()
        if log_every and s % log_every == 0:
            p = torch.softmax(lg, dim=-1)
            top2 = torch.topk(p, 2)
            trace.append(dict(
                t=s, ent=float(-(p * torch.log(p.clamp_min(1e-12))).sum()),
                top1=float(top2.values[0]),
                margin=float(top2.values[0] - top2.values[1])))
        nxt = int(lg.argmax())

        if nxt in end_set and stop_k is None:
            stop_k = s
        gen.append(nxt)
        if nxt in eos_set:
            fin = True
            break

        # ── 문단 경계 검출. 토큰 하나가 "\n\n" 을 품는 경우와 "\n" 두 개가
        #    연달아 오는 경우를 둘 다 본다 (토크나이저마다 다르다)
        piece = tokstr(tok, nxt)
        para_chars += len(piece)
        at_para = ("\n\n" in piece
                   or (prev_piece.endswith("\n") and piece.startswith("\n")))
        prev_piece = piece
        if at_para:
            n_para += 1

        # hidden 은 매 스텝 받는다 (이미 계산된 텐서라 공짜). 그래야 Δh 가
        # 인접 스텝 차이가 된다 — 신호 스텝에서만 받으면 sig_every 간격의
        # 차이가 되어 의미가 달라진다. attention 은 비싸므로 신호 스텝만
        need = (want_signals and stop_k is None
                and sig_every and (s + 1) % sig_every == 0)
        out = step(torch.tensor([[nxt]], device=device),
                   want_att=(need and use_att), want_hs=want_signals)
        if want_signals:
            if need:
                sg, prev_h = internal_signals(
                    model, out.attentions if use_att else None,
                    out.hidden_states, sig_layers, prev_h)
                sg["t"] = s + 1
                sigs.append(sg)
            else:
                prev_h = out.hidden_states[-1][0, -1].float()
            # 덤프는 sig 격자 **와** 문단 경계 둘 다에서 뜬다. 격자가 있어야
            # 정지 규칙을 촘촘히 점수 매길 수 있고, 문단 경계가 있어야
            # 문단 대푯값을 근사 없이 쓸 수 있다. probe 가 돌기 **전에** 떠야 한다
            if dump and stop_k is None and (need or at_para):
                H.append(np.stack([out.hidden_states[li][0, -1]
                                   .to(torch.float16).cpu().numpy()
                                   for li in dl]))
                DFLAG.append(dict(t=s + 1, para=int(at_para), grid=int(need)))
                if dump_mlp:
                    p_, s_ = mlp_snapshot(len(model.model.layers), proj)
                    MP.append(p_)
                    MS.append(s_)

        # ── think 블록 안에서만 중간 답을 읽는다
        if probe_mode == "para":
            fire = (at_para and para_chars >= para_min_chars) or \
                   (para_chars >= para_max_chars)
        else:
            fire = bool(probe_every) and (s + 1) % probe_every == 0
        if stop_k is None and arm_ids and fire:
            saved = out            # 로짓은 이미 계산된 텐서라 캐시 변경과 무관
            ans = probe_all()      # 팔마다 포크·롤백·길이검증
            out = saved
            for nm, v in ans.items():
                probes[nm].append(dict(t=s + 1, ans=v))
            para_chars = 0

    text = tok.decode(gen, skip_special_tokens=True)
    if stop_k is None:
        fin = False
    return dict(gen=gen, text=text, trace=trace, probes=probes, sigs=sigs,
                fin=fin, stop_k=stop_k, prefill_h=prefill_h, n_para=n_para,
                dflag=DFLAG,
                H=(np.stack(H) if H else None),
                MP=(np.stack(MP) if MP else None),
                MS=(np.stack(MS) if MS else None))


# ──────────────────────── 특징 / 라벨 ────────────────────────
def equiv_classes(seq, use_sympy=True):
    """답 문자열 열을 수학적 동치 클래스 id 열로 바꾼다.

    문자열 정규화만으로 세면 \\frac{1}{2} → 0.5 같은 표기 변화가 '답이
    바뀌었다' 로 집계돼 n_flips 가 부풀고 commit 지점이 뒤로 밀린다.
    commit_frac 이 간판 특징이라 여기가 지저분하면 전부 노이즈가 된다.
    """
    reps, out = [], []
    for s in seq:
        for i, r in enumerate(reps):
            if eq_math(s, r, use_sympy):
                out.append(i)
                break
        else:
            reps.append(s)
            out.append(len(reps) - 1)
    return out, reps


def commit_features(probes, final, gold, think_len, use_sympy=True):
    """probe 열에서 확신 고정 시점과 두 실패 방향을 뽑는다."""
    f = dict(n_probes=len(probes), n_flips=0, commit_step=None,
             commit_frac=None, first_gold_frac=None, n_distinct=0,
             any_probe_gold=False, probe_final_match=False, tail3_stable=0.0)
    if not probes:
        return f
    seq = [p["ans"] for p in probes]
    ts = [p["t"] for p in probes]
    # 최종 답과 정답도 같은 클래스 체계에 넣어 한 번에 비교한다
    cls, reps = equiv_classes(seq + [final, gold], use_sympy)
    pc, fc, gc = cls[:len(seq)], cls[len(seq)], cls[len(seq) + 1]

    f["n_flips"] = sum(1 for i in range(1, len(pc)) if pc[i] != pc[i - 1])
    f["n_distinct"] = len(set(pc))

    # 끝까지 안 바뀌고 최종 답과 같은 최초 지점
    j = len(pc)
    while j > 0 and pc[j - 1] == fc:
        j -= 1
    if j < len(pc):
        f["commit_step"] = ts[j]
        f["commit_frac"] = ts[j] / max(think_len, 1)
        f["probe_final_match"] = True

    # 정답을 거쳐 갔는가 (조기 종료 상한의 근거)
    for t, c in zip(ts, pc):
        if c == gc:
            f["any_probe_gold"] = True
            f["first_gold_frac"] = t / max(think_len, 1)
            break

    # 온라인 특징 — GT 없이 계산 가능해야 한다
    f["tail3_stable"] = float(len(pc) >= 3 and len(set(pc[-3:])) == 1)
    return f


# ──────────────────────── 재채점 ────────────────────────
def rescore(path, no_sympy=False):
    """저장된 pred/gold/probes 로 correct 와 feat 를 다시 계산해 덮어쓴다.

    채점기를 고칠 때마다 긴 생성을 다시 돌릴 이유가 없다. probes 를
    저장해 둔 이유가 이것이다. 원본은 .bak 으로 남긴다.
    sigs / trace / text 는 건드리지 않는다.
    """
    import shutil
    rows = [json.loads(l) for l in open(path) if l.strip()]
    shutil.copy(path, path + ".bak")
    ch_c = ch_f = 0
    for r in rows:
        old = r.get("correct")
        new = eq_math(r.get("pred"), r.get("gold"), not no_sympy)
        if old != new:
            ch_c += 1
        r["correct"] = new
        f0 = r.get("feat", {})
        f = commit_features(r.get("probes", []), r.get("pred"), r.get("gold"),
                            r.get("think_len", 1), not no_sympy)
        for k in ("think_len", "ent_med", "margin_med"):
            if k in f0:
                f[k] = f0[k]
        if f.get("any_probe_gold") != f0.get("any_probe_gold") or \
           f.get("commit_step") != f0.get("commit_step"):
            ch_f += 1
        r["feat"] = f
    with open(path, "w") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"재채점 {len(rows)}행 → {path}  (원본은 {path}.bak)")
    print(f"  correct 판정이 바뀐 행 {ch_c}건, "
          f"commit/any_probe_gold 가 바뀐 행 {ch_f}건\n")


# ──────────────────────── 분석 ────────────────────────
def analyze(path, baseline=None):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if not rows:
        print("행이 없다.")
        return
    done = [r for r in rows if r.get("fin")]
    acc = sum(1 for r in done if r["correct"]) / max(len(done), 1)
    trunc = 1 - len(done) / len(rows)
    print(f"\n{'='*78}")
    print(f"{len(rows)}문항.  완주 {len(done)} (절단 {trunc:.1%}).  "
          f"정확도 {acc:.1%}")

    if not done:
        print("완주 0건 — 길이 통계와 정착 분석을 건너뛴다")
        return
    print(f"think 길이 중앙값 {st.median([r['think_len'] for r in done]):.0f} 토큰, "
          f"probe 중앙값 {st.median([r['feat']['n_probes'] for r in done]):.0f}회")

    # 신호가 기록돼 있으면 요약만 보고한다. 신호 분석 자체는 별도 스크립트에서
    # (탐색/확정 분할 + 랜덤 컷 순열 + 길이 귀무 모델이 필요하므로)
    sg = [r for r in rows if r.get("sigs")]
    if sg:
        ks = sorted(set(sg[0]["sigs"][0]) - {"t"})
        npc = st.median([len(r["sigs"]) for r in sg])
        print(f"내부 신호 {len(ks)}종 × 체크포인트 중앙 {npc:.0f}개 "
              f"({len(sg)}건에 기록)")
        print(f"  {', '.join(ks)}")

    # ── probe 팔 비교 (2026-11 추가). 이 실행의 주 목적이다
    arm_rows = [r for r in rows if r.get("probes_arms")]
    if arm_rows:
        arms = list(arm_rows[0]["probes_arms"])
        print(f"\n{'='*78}\n=== probe 팔 {len(arms)}개: 라벨이 표현에 의존하나 ===")
        ca = {nm: [r["feat"]["commit_arms"].get(nm) for r in arm_rows
                   if r.get("feat", {}).get("commit_arms")] for nm in arms}
        base = PRIMARY_ARM if PRIMARY_ARM in arms else arms[0]
        print(f"{'팔':>4} | {'commit 있음':>10} | {f'{base} 와 동일':>10} | "
              f"{'commit_frac 중앙':>16}")
        print("-" * 52)
        for nm in arms:
            v = [x for x in ca[nm] if x is not None]
            same = sum(1 for x, y in zip(ca[nm], ca[base])
                       if x is not None and x == y)
            fr = [x / max(r["think_len"], 1) for x, r in zip(ca[nm], arm_rows)
                  if x is not None]
            print(f"{nm:>4} | {len(v):>4}/{len(ca[nm]):<5} | {same:>10} | "
                  f"{(st.median(fr) if fr else float('nan')):>16.3f}")
        # 체인 앞쪽에서 더 흔들리는가 — Stable Answers §4.2 가 보고한 패턴
        for nm in arms:
            if nm == base:
                continue
            early, late = [], []
            for r in arm_rows:
                pa_ = r["probes_arms"].get(base) or []
                pb_ = r["probes_arms"].get(nm) or []
                for p1, p2 in zip(pa_, pb_):
                    if p1["t"] != p2["t"]:
                        continue                 # 같은 체크포인트끼리만 비교
                    d = 0 if eq_math(p1["ans"], p2["ans"]) else 1
                    f = p1["t"] / max(r["think_len"], 1)
                    if f < 0.1:
                        early.append(d)
                    elif f > 0.66:
                        late.append(d)
            if early or late:
                e_ = sum(early) / len(early) if early else float("nan")
                l_ = sum(late) / len(late) if late else float("nan")
                print(f"  {base} vs {nm} 답 불일치율: 앞쪽 1/10 {e_:.1%} "
                      f"(n={len(early)}),  뒤쪽 1/3 {l_:.1%} (n={len(late)})")
        print("  A vs B 가 크면 → 라벨이 표현에 의존한다. 그 위에 올라앉은 "
              "모든 숫자가 흔들린다")
        print("  C vs D 가 유일한 '적대적 섭동의 내용 효과' 다 "
              "(길이가 맞춰진 대조군)")

    if baseline:
        b = [json.loads(l) for l in open(baseline) if l.strip()]
        # 두 형태를 받는다: 시드별 집계행({"seed":..,"acc":..}) 또는
        # 예제별 행({"correct":..,"seed":..})
        agg = [x for x in b if "acc" in x]
        if agg:
            per = {x.get("seed", i): float(x["acc"]) for i, x in enumerate(agg)}
            tr = [x["frac_truncated"] for x in agg if "frac_truncated" in x]
            tk = [x["avg_think_tokens_generated"] for x in agg
                  if "avg_think_tokens_generated" in x]
        else:
            seeds = {}
            for x in b:
                if "correct" in x:
                    seeds.setdefault(x.get("seed", 0), []).append(
                        bool(x["correct"]))
            per = {k: sum(v) / len(v) for k, v in seeds.items()}
            tr, tk = [], []
        if per:
            print(f"\n기존 baseline: " +
                  ", ".join(f"seed{k} {v:.1%}" for k, v in sorted(per.items())))
            if len(per) > 1:
                sd = st.stdev(per.values())
                print(f"  시드간 표준편차 {sd:.2%}  → **어떤 이득도 "
                      f"{2*sd:.2%} 는 넘어야 신호다**")
            if tk:
                print(f"  baseline think 길이 평균 {st.mean(tk):.0f} 토큰"
                      + (f", 절단 {st.mean(tr):.1%}" if tr else ""))
            # 비교는 반드시 **같은 기준**으로. baseline 이 절단을 오답으로
            # 세고 있으면 완주 기준 정확도와 비교하는 것은 틀린 대조다.
            acc_tr = sum(1 for r in done if r["correct"]) / len(rows)
            d = acc_tr - st.mean(list(per.values()))
            print(f"  이번 실행: 완주 기준 {acc:.1%}, "
                  f"**절단=오답 기준 {acc_tr:.1%}** (차이 {d:+.1%})")
            print(f"  ※ baseline 쪽 절단 {st.mean(tr):.0%} 와 이번 "
                  f"{len(rows)-len(done)}/{len(rows)}={1-len(done)/len(rows):.0%} 를 "
                  f"먼저 맞춰 볼 것 — 예산이 다르면 정확도도 같이 움직인다"
                  if tr else "")
            if abs(d) >= 0.10:
                print(f"  차이가 {d:+.1%}로 크다. 순서대로 의심할 것: "
                      f"(1) 예산/절단 처리 (2) 그리디 vs 샘플링 (3) 채점")

    # 실패의 분모에는 절단도 들어간다. 절단은 최종 답이 없을 뿐 실패이고,
    # 조기 종료가 가장 크게 구조하는 집단이 바로 거기다. fin 만 보면
    # 지배적 실패 모드를 분모에서 빼고 상한을 재게 된다.
    wrong = [r for r in done if not r["correct"]]
    trunc = [r for r in rows if not r.get("fin")]
    fail = wrong + trunc
    if not fail:
        print("\n실패가 없다.")
        return

    gold_seen = lambda r: any(eq_math(p["ans"], r["gold"])
                              for p in r.get("probes", []))

    # ── 이득 상한 (프로토콜 1번)
    nw = sum(1 for r in wrong if gold_seen(r))
    nt = sum(1 for r in trunc if gold_seen(r))
    ceil_n, ceil = nw + nt, (nw + nt) / len(fail)
    print(f"\n{'='*78}\n=== 이득 상한 ===")
    print(f"{'집단':>14} | {'건수':>4} | {'중간에 정답 거침':>16}")
    print("-" * 44)
    print(f"{'완주 오답':>14} | {len(wrong):>4} | {nw:>8} "
          f"({nw/max(len(wrong),1):>5.0%})")
    print(f"{'절단':>14} | {len(trunc):>4} | {nt:>8} "
          f"({nt/max(len(trunc),1):>5.0%})")
    print(f"{'합계':>14} | {len(fail):>4} | {ceil_n:>8} ({ceil:>5.0%})")
    acc_all = sum(1 for r in done if r["correct"]) / len(rows)
    print(f"\n→ **조기 종료 상한 {ceil:.1%}**  "
          f"(절단=오답 기준 정확도 {acc_all:.1%} → {acc_all + ceil_n/len(rows):.1%}, "
          f"+{ceil_n/len(rows):.1%}p)")
    first = []
    for r in fail:
        for p in r.get("probes", []):
            if eq_math(p["ans"], r["gold"]):
                first.append(p["t"] / max(r["think_len"], 1))
                break
    if first:
        print(f"   실패 건이 정답에 처음 도달한 시점 중앙값: "
              f"체인의 {st.median(first):.0%} 지점")
    print("  probe 가 '그 시점에 멈췄다면 뭐라 답했을까' 를 이미 갖고 있으므로,")
    print("  어떤 조기 종료 탐지기도 이 값을 넘지 못한다.")
    if ceil < 0.10:
        print("  => 10% 미만. 완벽한 탐지기도 못 건지니 이 방향은 접는다")
    else:
        print("  => 10% 이상. 탐지기를 만들 가치가 있다")

    # ── 두 실패 방향
    corr = [r for r in done if r["correct"]]
    lw = [r for r in fail if not gold_seen(r)]
    ra = [r for r in wrong if gold_seen(r)]
    tr_g = [r for r in trunc if gold_seen(r)]
    print(f"\n=== 실패 방향 분해 (처방이 반대라 섞어 세면 안 된다) ===")
    print(f"  locked_wrong    정답을 한 번도 안 거침   {len(lw):>4}건 "
          f"({len(lw)/len(fail):.1%}) → 더 생각해야 함")
    print(f"  right_abandoned 정답을 거쳤는데 번복     {len(ra):>4}건 "
          f"({len(ra)/len(fail):.1%}) → 덜 생각해야 함")
    print(f"  truncated_right 정답을 거쳤는데 예산소진 {len(tr_g):>4}건 "
          f"({len(tr_g)/len(fail):.1%}) → 멈추기만 하면 됨")
    for nm, grp in (("번복", ra), ("절단", tr_g)):
        fg = []
        for r in grp:
            for p in r.get("probes", []):
                if eq_math(p["ans"], r["gold"]):
                    fg.append(p["t"] / max(r["think_len"], 1))
                    break
        if fg:
            print(f"  {nm} 건의 첫 정답 도달 시점 중앙값 "
                  f"{st.median(fg):.1%} 지점 — 이보다 뒤의 토큰은 순손실")

    # ── 조기 확신이 오답과 관계있나
    print(f"\n{'='*78}\n=== commit_frac (확신 고정 시점) ===")
    def cf(rs):
        v = [r["feat"]["commit_frac"] for r in rs
             if r["feat"]["commit_frac"] is not None]
        return v
    a, b = cf(corr), cf(wrong)
    if len(a) >= 5 and len(b) >= 5:
        print(f"  정답 {st.median(a):.3f} (n={len(a)})  vs  "
              f"오답 {st.median(b):.3f} (n={len(b)})   [중앙값]")
        try:
            from scipy import stats as sps
            print(f"  Mann-Whitney p = {sps.mannwhitneyu(a,b).pvalue:.2e}")
        except ImportError:
            pass
        print("  오답이 더 낮으면 '일찍 고정한 쪽이 더 틀린다' 가 성립한다")
    nm = [r for r in done if not r["feat"]["probe_final_match"]]
    print(f"  최종 답이 probe 열에 한 번도 안 나타난 건 {len(nm)} "
          f"({len(nm)/len(done):.1%}) — 답이 </think> 이후에 결정된 경우")

    # ── 온라인 탐지기 (GT 불필요한 특징만)
    #    대상은 상한과 **같은 집단**이어야 한다. 완주 행만 세면 지배적 모드인
    #    truncated_right 가 빠져, 상한은 17건 기준인데 탐지기는 2건 기준으로
    #    평가하게 된다.
    fail_ids = {id(r) for r in fail}
    tgt = lambda r: id(r) in fail_ids and gold_seen(r)
    done = rows                      # 절단 행도 feat 를 갖고 있다
    base = sum(1 for r in rows if tgt(r)) / len(rows)
    print(f"\n{'='*78}\n=== 온라인 탐지기: '지금 멈추면 이득' 탐지 ===")
    print(f"대상 = 실패(오답+절단) & 중간에 정답 있음.  기저율 {base:.1%} "
          f"({sum(1 for r in rows if tgt(r))}/{len(rows)})")
    print("※ 정밀도는 반드시 이 기저율과 비교. 리프트가 없으면 신호가 없는 것이다")
    FEATS = ["commit_frac", "n_flips", "n_distinct", "n_probes", "tail3_stable",
             "think_len", "ent_med", "margin_med"]
    # 임계·방향·특징 조합을 전부 훑으므로 다중비교로 리프트가 그냥 뜬다.
    # 난수 라벨에서도 최고 리프트가 +10%p 를 넘는 것을 확인했다.
    # 그래서 고정 임계가 아니라 **라벨을 섞은 귀무분포**와 비교한다.
    grid = []
    for f in FEATS:
        vals = sorted({r["feat"][f] for r in done
                       if r.get("feat", {}).get(f) is not None})
        if len(vals) < 10:
            continue
        for sign in (1, -1):
            for q in (0.1, 0.2, 0.3, 0.4):
                th = vals[int(len(vals) * (1 - q))] if sign > 0 \
                    else vals[int(len(vals) * q)]
                idx = [i for i, r in enumerate(done)
                       if r.get("feat", {}).get(f) is not None
                       and (r["feat"][f] >= th if sign > 0
                            else r["feat"][f] <= th)]
                if len(idx) >= 10:
                    grid.append((f, "높음" if sign > 0 else "낮음", th, idx))

    lab = [bool(tgt(r)) for r in done]
    npos = sum(lab)

    def scan(labels):
        out = []
        for f, d, th, idx in grid:
            hit = sum(1 for i in idx if labels[i])
            out.append((hit / len(idx) - base, f, d, th, len(idx),
                        hit / len(idx), hit / max(npos, 1)))
        out.sort(reverse=True)
        return out

    best = scan(lab)
    print(f"\n{'특징':>14} {'방향':>4} | {'임계':>9} | {'발화':>5} | "
          f"{'정밀도':>6} | {'리프트':>7} | {'재현':>5}")
    print("-" * 68)
    for lift, f, d, th, n, prec, rec in best[:8]:
        print(f"{f:>14} {d:>4} | {th:>9.3f} | {n:>5} | {prec:>6.1%} | "
              f"{lift:>+7.1%} | {rec:>5.0%}")
    if not best:
        return

    import random as _rnd
    rng = _rnd.Random(0)
    null = []
    for _ in range(300):
        sh = lab[:]
        rng.shuffle(sh)
        null.append(scan(sh)[0][0])
    null.sort()
    p95 = null[int(0.95 * len(null))]
    obs = best[0][0]
    pval = sum(1 for x in null if x >= obs) / len(null)
    print(f"\n라벨 순열 귀무분포 ({len(grid)}개 조합, 300회): "
          f"최고 리프트 중앙값 {null[len(null)//2]:+.1%}, 95% 지점 {p95:+.1%}")
    print(f"관측 최고 리프트 {obs:+.1%} ({best[0][1]}), 순열 p = {pval:.3f}")
    # 95% 지점과의 대소가 아니라 p값으로 판정 (경계에서 부등호가 뒤집힌다)
    print(f"  => 유의 (p<0.05). 다변량 결합과 홀드아웃으로 진행" if pval < 0.05
          else f"  => **다중비교를 못 넘는다** (p={pval:.3f}) — 접거나 표본을 "
               f"늘릴 것")
    print("※ think_len 과 n_probes 는 귀무 모델이다. 다른 특징이 이 둘을 "
          "못 이기면 신호는 길이에 대해 잉여다")
    print("※ 절단 건은 정의상 think_len=max_new, n_probes=최대이므로 이 둘이")
    print("  '실패' 를 거의 완벽히 맞힌다. 이 표에서 둘이 1위면 동어반복이다")


# ──────────────────────── 자체 검증 ────────────────────────
def selftest(tok, model, data, dev, eos, end_ids, probe_ids, budget,
             probe_tokens, want_signals=False, sig_every=0, sig_layers=None,
             use_att=False, arm_ids=None, probe_mode="grid",
             para_min_chars=200, para_max_chars=1200):
    print("=== 자체 검증 ===")
    ok = True

    c1 = bool(end_ids)
    print(f"  [{'PASS' if c1 else 'FAIL'}] T1 </think> 토큰 id        {end_ids}")
    ok &= c1

    # T2 boxed 추출
    cases = [(r"x = \boxed{\frac{1}{2}}", r"\frac{1}{2}"),
             (r"\boxed{\frac{\sqrt{3}}{2}}", r"\frac{\sqrt{3}}{2}"),
             (r"\boxed{1} and \boxed{2}", "2"),
             (r"\boxed{\text{(A)}}", r"\text{(A)}")]
    bad = [(s, last_boxed(s), e) for s, e in cases if last_boxed(s) != e]
    # probe 는 \boxed{ 가 열린 뒤부터 읽으므로 깊이 추적이 따로 필요하다.
    # 첫 '}' 에서 끊으면 \frac{1}{2} 가 '\frac{1' 로 잘려 상한이 과소추정된다.
    pc = [(r"\frac{1}{2}} and more", r"\frac{1}{2}"),
          (r"\begin{pmatrix}1\\2\end{pmatrix}}$", r"\begin{pmatrix}1\\2\end{pmatrix}"),
          (r"42}", "42"),
          (r"\frac{1", None)]
    bad += [(s, close_brace(s), e) for s, e in pc if close_brace(s) != e]
    print(f"  [{'PASS' if not bad else 'FAIL'}] T2 boxed 추출 / 괄호 깊이  "
          f"{len(cases)+len(pc)-len(bad)}/{len(cases)+len(pc)}")
    for s, g, e in bad:
        print(f"         {s!r} → {g!r}, 기대 {e!r}")
    ok &= not bad

    # T3 정규화 동치 / 비동치
    same = [(r"\dfrac{1}{2}", r"\frac{1}{2}"), (".5", "0.5"),
            (r"50\%", "50"), (r"\left(1,2\right)", "(1,2)"),
            ("3.0", "3"), (r"\text{apples}", "apples"), ("1,000", "1000")]
    diff = [("1/2", "1/3"), ("2", "-2"), (r"\frac{1}{2}", r"\frac{1}{3}")]
    e1 = [(a, b) for a, b in same if norm_math(a) != norm_math(b)]
    e2 = [(a, b) for a, b in diff if norm_math(a) == norm_math(b)]
    print(f"  [{'PASS' if not e1 and not e2 else 'FAIL'}] T3 정규화          "
          f"        동치 {len(same)-len(e1)}/{len(same)}, "
          f"구분 {len(diff)-len(e2)}/{len(diff)}")
    for a, b in e1:
        print(f"         같아야 하는데 다름: {a!r} → {norm_math(a)!r} / "
              f"{b!r} → {norm_math(b)!r}")
    for a, b in e2:
        print(f"         달라야 하는데 같음: {a!r} / {b!r}")
    ok &= not e1 and not e2

    # T4 **probe 가 본 생성을 오염시키지 않는가** — 가장 중요한 검사.
    #    신호를 켠 상태로 돌려야 신호 경로가 캐시를 건드리지 않는지도 같이 본다
    ids = build_prompt(tok, data[0])
    kw = dict(max_new=budget, device=dev, eos_ids=eos, end_ids=end_ids,
              probe_tokens=probe_tokens, log_every=0,
              want_signals=want_signals, sig_every=sig_every,
              sig_layers=sig_layers, use_att=use_att,
              para_min_chars=para_min_chars, para_max_chars=para_max_chars)
    clean = generate_probed(model, tok, ids, probe_ids=None, arm_ids={},
                            probe_every=0, probe_mode="grid", **kw)
    # 실제로 쓸 팔 전부를 켜고 돌린다. 팔이 늘면 포크/롤백도 그만큼 늘어나므로
    # 단일팔로 통과해도 다중팔에서 깨질 수 있다
    withp = generate_probed(model, tok, ids, probe_ids=probe_ids,
                            arm_ids=arm_ids, probe_mode=probe_mode,
                            probe_every=max(32, budget // 6), **kw)
    c4 = clean["gen"] == withp["gen"]
    nfire = len(next(iter(withp["probes"].values()))) if withp["probes"] else 0
    print(f"  [{'PASS' if c4 else 'FAIL'}] T4 probe 가 본 생성을 안 건드림 "
          f"팔 {len(arm_ids or {})}개 × 발화 {nfire}회, "
          f"{'토큰열 완전 동일' if c4 else '토큰열 불일치 — 캐시 롤백 실패'}")
    if not c4:
        import transformers
        n = next((i for i, (x, y) in enumerate(zip(clean["gen"], withp["gen"]))
                  if x != y), min(len(clean["gen"]), len(withp["gen"])))
        print(f"         첫 불일치 스텝 {n} (길이 {len(clean['gen'])} vs "
              f"{len(withp['gen'])}).  transformers "
              f"{transformers.__version__} 의 crop 동작 확인 필요 — "
              f"probe 가 본 생성을 오염시키고 있으니 절대 그냥 돌리지 말 것")
    ok &= c4

    # T5 probe 가 읽을 수 있는 답을 내는가 — 팔마다 본다
    allp = [a for lst in withp["probes"].values() for a in
            [p["ans"] for p in lst]]
    nonempty = [a for a in allp if a]
    c5 = len(nonempty) >= max(1, len(allp) // 2)
    print(f"  [{'PASS' if c5 else 'FAIL'}] T5 probe 가 답을 읽어냄       "
          f"{len(nonempty)}/{len(allp)} 비어있지 않음")
    for nm, lst in withp["probes"].items():
        print(f"         [{nm}] {[p['ans'] for p in lst][:5]}")
    ok &= c5

    # T7 문단 경계가 실제로 검출되는가 (probe_mode=para 일 때만 의미 있음)
    npara = withp.get("n_para", 0)
    glen = len(withp["gen"])
    c7 = npara >= 1 if probe_mode == "para" else True
    print(f"  [{'PASS' if c7 else 'FAIL'}] T7 문단 경계 검출            "
          f"{npara}개 / {glen}토큰"
          + (f" (문단당 {glen/max(npara,1):.0f}토큰)" if npara else
             "  ← 0개면 토크나이저의 개행 표현을 확인할 것"))
    if probe_mode == "para" and npara and nfire:
        print(f"         발화 {nfire}회 = 문단 {npara}개를 "
              f"{para_min_chars}자 기준으로 병합한 결과")
    ok &= c7

    # T6 신호가 실제로 기록되고, 상수가 아닌가
    if want_signals:
        sg = withp["sigs"]
        c6 = bool(sg)
        rng_ok = []
        if sg:
            for k in sorted(set(sg[0]) - {"t"}):
                v = [x[k] for x in sg if k in x]
                if len(v) >= 3 and max(v) - min(v) < 1e-9:
                    rng_ok.append(k)
            c6 = c6 and not rng_ok
        print(f"  [{'PASS' if c6 else 'FAIL'}] T6 내부 신호 기록           "
              f"{len(sg)}회 × {len(set(sg[0])-{'t'}) if sg else 0}종")
        if sg:
            k0 = sorted(set(sg[0]) - {"t"})
            print(f"         {', '.join(k0)}")
        if rng_ok:
            print(f"         상수로 죽은 신호: {rng_ok} — 쓸모없으니 확인할 것")
        ok &= c6

    print(f"  => {'통과' if ok else '실패 — 중단할 것'}\n")
    return ok


# ──────────────────────────── main ────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", default=None)
    ap.add_argument("--rescore", default=None,
                    help="기존 jsonl 의 correct/feat 를 저장된 pred/gold/probes "
                         "로 다시 계산한다. 모델 불필요, GPU 불필요")
    ap.add_argument("--baseline", default=None,
                    help="기존 seed 실행 jsonl (correct/seed 필드). 교차확인용")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--data-file", default=None)
    ap.add_argument("--hf-name", default=None)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--levels", type=int, nargs="*", default=None)
    ap.add_argument("--max-new", type=int, default=8192)
    ap.add_argument("--probe-every", type=int, default=256)
    ap.add_argument("--probe-tokens", type=int, default=40,
                    help="probe 당 최대 생성 토큰. 행렬/pmatrix 답은 24 로 "
                         "모자라 잘린다")
    ap.add_argument("--probe-prefix", default=PROBE_PREFIX)
    ap.add_argument("--arms", default="A,B,C,D",
                    help=f"쓸 probe 팔. 가능: {','.join(PROBE_ARMS)}. "
                         "A 는 기존 데이터와의 다리이므로 빼지 말 것. "
                         "'A' 하나만 주면 기존 실행과 동일")
    ap.add_argument("--probe-mode", default="para", choices=["para", "grid"],
                    help="para = `\\n\\n` 경계 + 문자수 병합(PUMA 입도). "
                         "grid = 기존 probe_every 격자(재현용)")
    ap.add_argument("--para-min-chars", type=int, default=200,
                    help="직전 probe 이후 이만큼 쌓인 뒤의 첫 `\\n\\n` 에서 발화. "
                         "생 `\\n\\n` 은 문단당 ~53토큰이라 probe 하나당 너무 짧다")
    ap.add_argument("--para-max-chars", type=int, default=1200,
                    help="경계가 안 와도 이만큼 쌓이면 발화. 긴 문단이 통째로 "
                         "probe 없이 지나가는 것을 막는다")
    ap.add_argument("--dump-layers", type=int, nargs="*",
                    default=[9, 18, 27, 31, 35, 36],
                    help="덤프할 hidden 레이어. 37층 전부 저장하면 n=500 에 "
                         "35GB 이고 실제로 쓴 건 세 층뿐이었다. "
                         "빈 리스트면 전부")
    ap.add_argument("--dump-mlp", action="store_true",
                    help="MLP 중간활성 projection·통계도 저장(2.3GB). "
                         "지금까지 분석에 쓰지 않았으므로 기본은 끔")
    ap.add_argument("--log-every", type=int, default=8,
                    help="엔트로피/마진 기록 간격. 0 이면 끔")
    ap.add_argument("--no-sympy", action="store_true")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", default="math_commit.jsonl")
    ap.add_argument("--save-probes", action="store_true", default=True)
    ap.add_argument("--save-text", action="store_true", default=True,
                    help="생성 텍스트 저장. 엔트로피 스파이크 지점의 토큰을 "
                         "나중에 확인하려면 필요하다")
    ap.add_argument("--selftest-budget", type=int, default=400)
    ap.add_argument("--skip-selftest", action="store_true")
    ap.add_argument("--attn-impl", default="sdpa", choices=["sdpa", "eager"],
                    help="eager 라야 attention 확률을 받을 수 있으나, 부동소수점 "
                         "연산 순서가 달라 sdpa 로 만든 기존 jsonl 과는 생성 "
                         "토큰열 자체가 갈린다(실측 111스텝에서 분기)")
    ap.add_argument("--signals", action="store_true",
                    help="본 생성 스텝에서 내부 신호를 기록. sdpa 면 hidden/"
                         "logit 계열만, eager 면 attention 계열까지")
    ap.add_argument("--sig-every", type=int, default=16)
    ap.add_argument("--sig-layers", type=int, nargs="*", default=[-4, -3, -2, -1],
                    help="logit lens 를 적용할 레이어. lm_head 를 레이어마다 "
                         "돌리므로 늘릴수록 느려진다")
    ap.add_argument("--dump-hidden", action="store_true",
                    help="체크포인트마다 전 레이어 hidden(fp16) + MLP 중간활성의 "
                         "random projection + 희소성 통계 + prefill 상태를 "
                         "별도 바이너리로 저장한다. 스칼라로 뭉개면 방향 정보가 "
                         "사라지므로, 재실행 없이 사후 분석하려면 필요하다")
    ap.add_argument("--proj-dim", type=int, default=256,
                    help="MLP 중간활성(12288차원)을 넣을 random projection 차원. "
                         "128 이면 거리 보존 오차가 최대 20%% 까지 벌어지고 "
                         "256 이면 절반으로 준다. 전체 용량의 3%% 라 256 이 낫다")
    ap.add_argument("--resume", action="store_true",
                    help="--out 이 이미 있으면 이어서 쓴다 (기본은 덮어쓰기)")
    a = ap.parse_args()

    if a.rescore:
        rescore(a.rescore, a.no_sympy)
        if not a.analyze:
            analyze(a.rescore, a.baseline)
        return

    if a.analyze:
        analyze(a.analyze, a.baseline)
        return

    use_att = (a.attn_impl == "eager")
    if a.signals and not use_att:
        print("※ sdpa 이므로 attention 계열(head_jsd/attn_ent/attn_sink)은 "
              "건너뛴다.\n"
              "  hidden·logit 계열은 전부 기록되고, 생성은 기존 sdpa jsonl 과 "
              "동일하므로 교차 확인이 가능하다")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    data = load_math500(a.data_file, a.hf_name, a.n, levels=a.levels)
    print(f"{len(data)}문항  max_new {a.max_new}  "
          f"probe {a.probe_every}토큰마다 {a.probe_tokens}토큰"
          + (f"  신호 {a.sig_every}토큰마다 ({a.attn_impl})" if a.signals else ""))

    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=getattr(torch, a.dtype), device_map=dev,
        attn_implementation=a.attn_impl).eval()

    e = tok.convert_tokens_to_ids("</think>")
    end_ids = [e] if isinstance(e, int) and e >= 0 else \
        tok("</think>", add_special_tokens=False)["input_ids"][:1]
    eos = {tok.eos_token_id}
    for t in ("<|im_end|>", "<|endoftext|>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0:
            eos.add(i)
    probe_ids = tok(a.probe_prefix, add_special_tokens=False)["input_ids"]

    # ── probe 팔 구성
    names = [x.strip() for x in a.arms.split(",") if x.strip()]
    bad = [x for x in names if x not in PROBE_ARMS]
    if bad:
        raise SystemExit(f"모르는 팔 {bad}. 가능: {list(PROBE_ARMS)}")
    if PRIMARY_ARM not in names:
        print(f"※ 경고: 기준 팔 '{PRIMARY_ARM}' 이 빠졌다. "
              f"기존 데이터와 비교가 안 되고 하위 호환 필드도 비게 된다")
    arm_ids = {nm: tok(PROBE_ARMS[nm], add_special_tokens=False)["input_ids"]
               for nm in names}
    print(f"probe 팔 {len(arm_ids)}개, 모드 {a.probe_mode}"
          + (f" (문단 {a.para_min_chars}~{a.para_max_chars}자)"
             if a.probe_mode == "para" else f" ({a.probe_every}토큰 격자)"))
    for nm, pids in arm_ids.items():
        print(f"  [{nm}] {len(pids):>2}토큰  {PROBE_ARMS[nm]!r}")

    if not a.skip_selftest:
        if not selftest(tok, model, data, dev, eos, end_ids, probe_ids,
                        min(a.selftest_budget, a.max_new), a.probe_tokens,
                        a.signals, a.sig_every, a.sig_layers, use_att,
                        arm_ids, a.probe_mode, a.para_min_chars,
                        a.para_max_chars):
            raise SystemExit("자체 검증 실패 — 중단")

    # ── 원본 덤프 준비
    proj, dump_f = None, None
    nl_all = len(model.model.layers) + 1          # 임베딩 + 블록
    dump_layers = [li % nl_all for li in a.dump_layers] if a.dump_layers \
        else list(range(nl_all))
    if a.dump_hidden:
        nl = len(model.model.layers)
        hid = model.config.hidden_size
        inter = model.config.intermediate_size
        if a.dump_mlp:
            attach_mlp_hooks(model)
            # 투영 행렬은 재현 가능해야 한다 (seed 고정 + 파일로 저장)
            g = torch.Generator(device=dev).manual_seed(0)
            proj = (torch.randn(inter, a.proj_dim, generator=g, device=dev,
                                dtype=torch.float32) / math.sqrt(a.proj_dim))
            np.save(a.out + ".proj.npy", proj.cpu().numpy())
        mode = "ab" if (a.resume and os.path.exists(a.out + ".h.f16")) else "wb"
        dump_f = dict(h=open(a.out + ".h.f16", mode),
                      pf=open(a.out + ".prefill.f16", mode),
                      mt=open(a.out + ".meta.jsonl", "a" if mode == "ab" else "w"))
        if a.dump_mlp:
            dump_f["mp"] = open(a.out + ".mlp.f16", mode)
            dump_f["ms"] = open(a.out + ".mlpstat.f32", mode)
        # 다운스트림이 [행, 층, hidden] 으로 reshape 하려면 층 목록이 필요하다
        json.dump(dict(dump_layers=dump_layers, n_layers_all=nl_all,
                       hidden=hid, dump_mlp=bool(a.dump_mlp),
                       arms=list(arm_ids), probe_mode=a.probe_mode,
                       para_min_chars=a.para_min_chars,
                       sig_every=a.sig_every),
                  open(a.out + ".dumpinfo.json", "w"), indent=1)
        per = len(dump_layers) * hid * 2 + (
            nl * a.proj_dim * 2 + nl * 5 * 4 if a.dump_mlp else 0)
        # 격자 + 문단 경계 둘 다 뜨므로 체크포인트가 늘어난다
        ckpt = 3100 / max(a.sig_every, 1) + 60
        print(f"덤프: 레이어 {dump_layers} ({len(dump_layers)}개) × hidden {hid}"
              + (f", MLP {inter}→{a.proj_dim}" if a.dump_mlp else ", MLP 끔"))
        print(f"  체크포인트당 {per/1024:.0f}KB → 예제당 약 "
              f"{per*ckpt/1e6:.0f}MB, {len(data)}건이면 약 "
              f"{per*ckpt*len(data)/1e9:.1f}GB")
        print(f"  파일: {a.out}.h.f16 / .prefill.f16 / .meta.jsonl / "
              f".dumpinfo.json"
              + (" / .mlp.f16 / .mlpstat.f32 / .proj.npy\n" if a.dump_mlp
                 else "\n"))

    # 한 줄씩 덧붙인다. 매 예제마다 전체를 다시 쓰면 sigs/trace/text 가 붙은
    # n=500 에서 누적 수십 GB 를 쓰게 되고, 생성보다 저장이 더 걸린다
    done_ids = set()
    if a.resume and os.path.exists(a.out):
        for l in open(a.out):
            if l.strip():
                done_ids.add(json.loads(l).get("ex_id"))
        print(f"이어서 쓴다: 이미 {len(done_ids)}건 완료")
    else:
        open(a.out, "w").close()

    t0, n_done = time.time(), 0
    for n_ex, ex in enumerate(data):
        if n_ex in done_ids:
            continue
        ids = build_prompt(tok, ex)
        o = generate_probed(model, tok, ids, max_new=a.max_new, device=dev,
                            eos_ids=eos, end_ids=end_ids, probe_ids=probe_ids,
                            probe_every=a.probe_every,
                            probe_tokens=a.probe_tokens,
                            log_every=a.log_every,
                            want_signals=a.signals, sig_every=a.sig_every,
                            sig_layers=a.sig_layers, use_att=use_att,
                            dump=a.dump_hidden, proj=proj,
                            arm_ids=arm_ids, probe_mode=a.probe_mode,
                            para_min_chars=a.para_min_chars,
                            para_max_chars=a.para_max_chars,
                            dump_layers=dump_layers, dump_mlp=a.dump_mlp)
        think_len = o["stop_k"] if o["stop_k"] is not None else len(o["gen"])
        fa = final_answer(o["text"])
        gold = ex["answer"]
        correct = eq_math(fa, gold, not a.no_sympy)

        # ── 하위 호환: feat / probes 는 **기준 팔(A)** 기준으로 그대로 둔다.
        #    probe_commit.py 와 quartile_paired.py 가 r["probes"] 와
        #    feat["commit_step"] 을 읽으므로, 여기 모양이 바뀌면 전부 깨진다
        prim = o["probes"].get(PRIMARY_ARM) or \
            (next(iter(o["probes"].values())) if o["probes"] else [])
        feat = commit_features(prim, fa, gold, think_len, not a.no_sympy)
        feat["think_len"] = think_len
        feat["n_para"] = o["n_para"]
        # 팔별 commit — A vs B 가 wording 검사, C−D 가 적대적 섭동의 내용 효과
        feat["commit_arms"] = {}
        feat["flips_arms"] = {}
        for nm, lst in o["probes"].items():
            fa_ = commit_features(lst, fa, gold, think_len, not a.no_sympy)
            feat["commit_arms"][nm] = fa_["commit_step"]
            feat["flips_arms"][nm] = fa_["n_flips"]
        if o["trace"]:
            feat["ent_med"] = st.median([t["ent"] for t in o["trace"]])
            feat["margin_med"] = st.median([t["margin"] for t in o["trace"]])
        r = dict(ex_id=n_ex, uid=ex.get("unique_id"), level=ex.get("level"),
                 subject=ex.get("subject"), fin=o["fin"],
                 gen_len=len(o["gen"]), prompt_len=len(ids),
                 think_len=think_len, pred=fa, gold=gold, correct=correct,
                 feat=feat)
        if a.save_probes:
            r["probes"] = prim                 # 기존 형식 그대로
            r["probes_arms"] = o["probes"]     # 새 데이터는 여기
            r["trace"] = o["trace"]        # tok_ent 원본 (변동성 계산용)
        if a.save_text:
            r["text"] = o["text"]          # 스파이크 지점 토큰 확인용
        if a.signals:
            r["sigs"] = o["sigs"]

        # 바이너리 → 메타 → jsonl 순서로 쓰고 매번 flush 한다. 중간에 죽어도
        # meta 행 수가 바이너리 행 수를 넘지 않으므로 정렬이 깨지지 않는다
        if a.dump_hidden:
            if o["prefill_h"] is not None:
                dump_f["pf"].write(o["prefill_h"].tobytes())
            if o["H"] is not None:
                dump_f["h"].write(o["H"].tobytes())
                if a.dump_mlp and o["MP"] is not None:
                    dump_f["mp"].write(o["MP"].tobytes())
                    dump_f["ms"].write(o["MS"].tobytes())
            for f_ in dump_f.values():
                f_.flush()
            # meta 는 **덤프 행과 1:1** 이어야 한다. 이제 덤프가 sig 격자와
            # 문단 경계 둘 다에서 뜨므로 sigs 가 아니라 dflag 를 따라간다
            for d_ in o["dflag"]:
                dump_f["mt"].write(json.dumps(
                    dict(ex_id=n_ex, t=d_["t"], para=d_["para"],
                         grid=d_["grid"])) + "\n")
            dump_f["mt"].flush()

        with open(a.out, "a") as fo:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")

        n_done += 1
        cf = feat.get("commit_frac")
        el = time.time() - t0
        left = len(data) - len(done_ids) - n_done
        eta = el / n_done * left
        ca = feat.get("commit_arms", {})
        agree = (sum(1 for nm in ca if nm != PRIMARY_ARM
                     and ca[nm] == ca.get(PRIMARY_ARM)), max(len(ca) - 1, 0))
        print(f"  {n_ex+1}/{len(data)} L{ex.get('level','?')} "
              f"{'O' if correct else 'X'} len {len(o['gen'])} "
              f"para {feat.get('n_para',0)} probe {feat['n_probes']} "
              f"flip {feat['n_flips']} "
              f"commit {cf if cf is None else round(cf,2)} "
              f"팔일치 {agree[0]}/{agree[1]} "
              f"gold거침 {int(feat['any_probe_gold'])} "
              f"| {el/60:.0f}분 경과, 남은 {eta/3600:.1f}시간", flush=True)

    analyze(a.out, a.baseline)


if __name__ == "__main__":
    main()