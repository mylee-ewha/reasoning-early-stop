"""MATH-500 채점기 보강.

run_commit_math.py 의 norm_math/eq_math 가 놓친 여섯 가지를 더 지운다.
n=500c 에서 완주·오답 19건 중 16건이 이 여섯 가지 때문에 오답으로 집계됐다.

    1. 진법 아래첨자      52_8, 4210_{5}      → 52, 4210
    2. 단위               864 \\mbox{inches}^2 → 864      (화이트리스트만)
    3. 객관식 괄호         \\text{(C)}          → C
    4. \\frac 인자 축약     \\frac9{19}          → \\frac{9}{19}
    5. 좌변               x=5,  x \\in [-2,7]  → 5, [-2,7]
    6. 튜플·행렬 원소별 비교  (\\dfrac13, …) vs (1/3, …)

다섯째까지는 표기만 지우고, 여섯째와 대수 동치(9a+11 vs 11+9a)는 비교
단계에서 처리한다. sympy 의 parse_latex 는 antlr 런타임을 요구하고 이
환경에 없으므로, LaTeX 를 sympify 가 읽는 꼴로 직접 바꾼 경로를 쓴다.

**과허용 점검이 필수다.** 채점기를 느슨하게 만들면 상한이 공짜로 올라간다.
selftest() 가 (a) 19건을 기대대로 가르는지 (b) 다른 문제의 정답으로 채점한
우연 일치율(hit_null)이 0 근처인지를 함께 본다. 둘 다 통과해야 쓴다.
"""
import re

try:                                  # 분석용으로 떼어둔 모듈이 있으면 그걸
    from grade import eq_math as eq_base
    from grade import mat2tuple, norm_math
except ImportError:                   # 없으면 본체에서 직접 (torch 를 끌고 온다)
    from run_commit_math import eq_math as eq_base
    from run_commit_math import mat2tuple, norm_math

# 단위는 화이트리스트로만 지운다. "2x" 의 x 를 단위로 보면 2 와 같아져 버린다
_UNIT = (r"(?:inch(?:es)?|cm|mm|km|ft|feet|foot|yards?|miles?|meters?|"
         r"units?|squareunits?|sq|degrees?|radians?|dollars?|cents?|"
         r"seconds?|minutes?|hours?|days?|weeks?|months?|years?|"
         r"liters?|litres?|gallons?|grams?|kg|pounds?|lbs?|"
         r"students?|people|persons?|ways?|times?|cubic\w*|square\w*)")


def _strip_units(s):
    m = re.fullmatch(rf"(-?[\d.,/\\{{}}]+?){_UNIT}(?:\^\{{?\d+\}}?)?", s)
    return m.group(1) if m else s


def _strip_base(s):
    """진법 아래첨자. 어간이 그 진법의 유효 숫자로만 돼 있을 때만 지운다."""
    m = re.fullmatch(r"([0-9a-z]+)_\{?(\d{1,2})\}?", s)
    if not m:
        return s
    stem, base = m.group(1), int(m.group(2))
    if not 2 <= base <= 36:
        return s
    try:
        int(stem, base)
    except ValueError:
        return s
    return stem


def _strip_lhs(s):
    """x=5 → 5,  x\\in[-2,7] → [-2,7]. 좌변이 짧은 변수일 때만."""
    for op in (r"\\in", "="):
        parts = re.split(op, s, maxsplit=1)
        if len(parts) == 2 and re.fullmatch(r"[a-z](?:\^?\d)?(?:\(\w\))?",
                                            parts[0]) and parts[1]:
            return parts[1]
    return s


def _strip_choice(s):
    m = re.fullmatch(r"\(([a-e])\)", s)
    return m.group(1) if m else s


def _fix_frac(s):
    """\\frac9{19} / \\frac{9}19 → \\frac{9}{19}. norm_math 는 둘 다 못 잡는다."""
    s = re.sub(r"\\frac(?!\{)\s*([^{\s\\])\s*(\{[^{}]*\})", r"\\frac{\1}\2", s)
    s = re.sub(r"\\frac(\{[^{}]*\})\s*([^{\s\\])", r"\\frac\1{\2}", s)
    return s


def norm2(s):
    if s is None:
        return None
    s = norm_math(s)
    if s is None:
        return None
    for f in (_fix_frac, _strip_choice, _strip_lhs, _strip_base, _strip_units):
        s = f(s)
    return re.sub(r"^\{(.*)\}$", r"\1", s)


# ── 대수 동치: LaTeX → sympify 가 읽는 꼴 ────────────────────────────
def _to_sympy_src(s):
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
    s = s.replace("\\pi", "pi").replace("\\cdot", "*").replace("\\times", "*")
    s = re.sub(r"\\left|\\right", "", s)
    s = re.sub(r"\^\{([^{}]*)\}", r"**(\1)", s)
    s = re.sub(r"\^(\w)", r"**\1", s)
    if re.search(r"\\[a-z]", s):          # 못 바꾼 매크로가 남으면 포기
        return None
    return s


def _sym_eq(a, b):
    src = [_to_sympy_src(x) for x in (a, b)]
    if None in src:
        return False
    try:
        import sympy
        from sympy.parsing.sympy_parser import (parse_expr,
                                                standard_transformations)
        from sympy.parsing.sympy_parser import \
            implicit_multiplication_application as imul
        tr = standard_transformations + (imul,)
        ea, eb = (parse_expr(x, transformations=tr, evaluate=True) for x in src)
        return bool(sympy.simplify(ea - eb) == 0)
    except Exception:
        return False


# ── 튜플·행렬 원소별 ──────────────────────────────────────────────
def _split_top(s):
    """최상위 쉼표로 분해. 중괄호·괄호 안의 쉼표는 건드리지 않는다."""
    out, buf, d = [], "", 0
    for c in s:
        if c in "{([":
            d += 1
        elif c in "})]":
            d -= 1
        if c == "," and d == 0:
            out.append(buf); buf = ""
        else:
            buf += c
    out.append(buf)
    return [x for x in out if x != ""]


def _unwrap(s):
    m = re.fullmatch(r"[\(\[]([^\[\]]*)[\)\]]", s)
    return m.group(1) if m else s


def eq_math2(a, b, use_sympy=True):
    """eq_math 의 상위 호환. 기존이 맞다고 하면 그대로 맞다고 한다."""
    if eq_base(a, b, use_sympy):
        return True
    na, nb = norm2(mat2tuple(a)), norm2(mat2tuple(b))
    if na is None or nb is None:
        return False
    if na == nb:
        return True

    ea, eb = _split_top(_unwrap(na)), _split_top(_unwrap(nb))
    if len(ea) == len(eb) and len(ea) > 1:
        return all(eq_math2(x, y, use_sympy) for x, y in zip(ea, eb))
    if len(ea) == len(eb) == 1:
        if eq_base(ea[0], eb[0], use_sympy):
            return True
        return _sym_eq(ea[0], eb[0]) if use_sympy else False
    return False


# ── 자체 점검 ────────────────────────────────────────────────────
#   n=500c 의 완주·오답 19건. 마지막 3건은 모델이 답을 하나만 낸 진짜 오답이라
#   채점기를 고쳐도 오답으로 남아야 한다 — 여기가 True 로 바뀌면 과허용이다
CASES = [
    ("5", "x=5", True), ("52", "52_8", True), ("40", "40_9", True),
    ("2516", "2516_8", True), ("4210", "4210_{5}", True),
    ("4343", "4343_6", True), ("c", "\\text{(C)}", True),
    ("E", "\\text{(E)}", True), ("864", "864 \\mbox{ inches}^2", True),
    ("15", "15\\mbox{ cm}^2", True),
    ("\\dfrac{9}{19}", "\\frac9{19}", True),
    ("\\dfrac{9a + 11}{20}", "\\frac{11+9a}{20}", True),
    ("[-2, 7]", "x \\in [-2,7]", True),
    ("\\begin{pmatrix} -\\dfrac{1}{3} \\\\ \\dfrac{2}{3} \\\\ \\dfrac{5}{3} \\end{pmatrix}",
     "\\begin{pmatrix} -1/3 \\\\ 2/3 \\\\ 5/3 \\end{pmatrix}", True),
    ("\\begin{pmatrix} \\dfrac{1}{5} \\\\ -\\dfrac{18}{5} \\end{pmatrix}",
     "\\begin{pmatrix} 1/5 \\\\ -18/5 \\end{pmatrix}", True),
    ("\\begin{pmatrix} \\dfrac{16}{49} \\\\ \\dfrac{48}{49} \\\\ \\dfrac{24}{49} \\end{pmatrix}",
     "\\begin{pmatrix} 16/49 \\\\ 48/49 \\\\ 24/49 \\end{pmatrix}", True),
    ("1", "1,-2", False),
    ("7", "3, 5, 7", False),
    ("1 - \\sqrt{19}", "1 \\pm \\sqrt{19}", False),
]

# 과허용 점검용 — 서로 다른 답은 끝까지 달라야 한다
NEG = [("1/2", "1/3"), ("(1,2)", "(2,1)"), ("5", "x=6"), ("52", "53_8"),
       ("864", "865 \\mbox{ inches}^2"), ("2x", "2"), ("a_1", "a"),
       ("[-2,7]", "x \\in [-2,8]"), ("\\frac{9}{19}", "\\frac{9}{18}"),
       ("c", "\\text{(D)}"), ("3", "3_2")]


def selftest(verbose=True):
    bad = []
    for a, b, want in CASES:
        got = eq_math2(a, b)
        if got != want:
            bad.append(("CASE", a, b, want, got))
    for a, b in NEG:
        if eq_math2(a, b):
            bad.append(("NEG", a, b, False, True))
    if verbose:
        print(f"CASES {len(CASES)}건 + NEG {len(NEG)}건 → "
              f"{'통과' if not bad else f'실패 {len(bad)}건'}")
        for t, a, b, w, g in bad:
            print(f"  [{t}] {a!r} vs {b!r}: 기대 {w}, 결과 {g}")
    return not bad


if __name__ == "__main__":
    selftest()