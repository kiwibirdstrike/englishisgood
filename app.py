import io
import os
import math
import numpy as np
import pandas as pd
import streamlit as st

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

st.set_page_config(page_title="층화가중 리커트 분포", layout="wide")

# =============================
# 0) 층화 변수 매핑(개념 -> (표본컬럼, 모집단컬럼))
# =============================
STRATA_MAP = {
    "소속대학": ("소속 단과대학", "대학"),
    "학년": ("학년", "학년"),
    "레벨반": ("소속 레벨반", "레벨반"),
}

# =============================
# 1) Matplotlib 한글 폰트(Windows) 강제 설정
# =============================
@st.cache_resource
def setup_korean_font():
    font_files = [
        r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\malgunsl.ttf",
    ]
    for fp in font_files:
        if os.path.exists(fp):
            fm.fontManager.addfont(fp)
            name = fm.FontProperties(fname=fp).get_name()
            mpl.rcParams["font.family"] = name
            mpl.rcParams["axes.unicode_minus"] = False
            return name

    candidates = ["Malgun Gothic", "맑은 고딕", "AppleGothic", "NanumGothic"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            mpl.rcParams["font.family"] = name
            mpl.rcParams["axes.unicode_minus"] = False
            return name

    mpl.rcParams["axes.unicode_minus"] = False
    return None

setup_korean_font()

# =============================
# 2) IO
# =============================
def read_csv_robust(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except Exception:
            pass
    return pd.read_csv(io.BytesIO(raw), encoding_errors="ignore")

def load_pop_xlsx(file, label: str) -> pd.DataFrame:
    raw = file.getvalue()
    xf = pd.ExcelFile(io.BytesIO(raw))
    sheet = st.selectbox(f"{label} 시트 선택", xf.sheet_names, key=f"{label}_sheet")
    return pd.read_excel(io.BytesIO(raw), sheet_name=sheet)

# =============================
# 3) Normalization / Likert coercion
# =============================
def normalize_cat(s: pd.Series) -> pd.Series:
    s0 = s.copy()
    x = pd.to_numeric(s0, errors="coerce")
    ok = x.notna().mean()

    if ok >= 0.8:
        xi = x.round()
        if (np.abs(x - xi) < 1e-9).mean() >= 0.95:
            return xi.astype("Int64").astype("string")
        return x.astype("string")

    return (
        s0.astype("string")
          .str.replace(r"\s+", " ", regex=True)
          .str.strip()
          .str.casefold()
    )

def coerce_likert(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().mean() >= 0.6:
        return x

    s_norm = (
        s.astype("string")
         .str.strip()
         .str.replace(r"\s+", "", regex=True)
         .str.replace(r"[^0-9가-힣]", "", regex=True)
    )
    ko_map = {
        "전혀아니다": 1,
        "아니다": 2,
        "보통이다": 3,
        "보통": 3,
        "그렇다": 4,
        "매우그렇다": 5,
    }
    mapped = s_norm.map(ko_map)
    if mapped.notna().any():
        return mapped

    z = s.astype("string").str.extract(r"([1-5])", expand=False)
    return pd.to_numeric(z, errors="coerce")

# =============================
# 4) Weighting
# =============================
def compute_poststrat_weights(sample: pd.DataFrame, pop: pd.DataFrame, samp_col: str, pop_col: str):
    s = sample.copy()
    p = pop.copy()

    s["__STRATA__"] = normalize_cat(s[samp_col])
    p["__STRATA__"] = normalize_cat(p[pop_col])

    pop_ct = p["__STRATA__"].value_counts(dropna=False)
    samp_ct = s["__STRATA__"].value_counts(dropna=False)

    w_map = (pop_ct / samp_ct).replace([np.inf, -np.inf], np.nan).to_dict()
    s["_w"] = s["__STRATA__"].map(w_map)

    return s

# =============================
# 5) Stats (mean/sd only)
# =============================
def weighted_mean_sd(x: np.ndarray, w: np.ndarray):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    ok = np.isfinite(x) & np.isfinite(w)
    x = x[ok]
    w = w[ok]
    if len(x) == 0:
        return np.nan, np.nan
    wsum = float(np.sum(w))
    if wsum <= 0:
        return np.nan, np.nan
    mean = float(np.sum(w * x) / wsum)
    var = float(np.sum(w * (x - mean) ** 2) / wsum)
    sd = float(np.sqrt(var))
    return mean, sd

def prepare_question(df_w: pd.DataFrame, q_col: str, use_weights: bool):
    """
    반환:
      pivot_counts: index=1..5, columns=strata (빈도 또는 가중합)
      overall: (mean, sd)
      strata_lines: [(층명, mean, sd), ...]
    """
    tmp = df_w[["__STRATA__", q_col, "_w"]].copy()
    tmp[q_col] = coerce_likert(tmp[q_col])
    tmp = tmp.dropna(subset=[q_col, "__STRATA__"])
    tmp = tmp[tmp[q_col].between(1, 5)]
    tmp[q_col] = tmp[q_col].astype(int)

    if len(tmp) == 0:
        return pd.DataFrame(), (np.nan, np.nan), []

    # pivot (빈도/가중합)
    if use_weights:
        val = tmp.groupby([q_col, "__STRATA__"])["_w"].sum().reset_index(name="value")
    else:
        val = tmp.groupby([q_col, "__STRATA__"]).size().reset_index(name="value")

    pivot = (
        val.pivot_table(index=q_col, columns="__STRATA__", values="value", aggfunc="sum")
           .reindex([1, 2, 3, 4, 5])
           .fillna(0)
    )

    # overall mean/sd
    x_all = tmp[q_col].astype(float).to_numpy()
    w_all = tmp["_w"].astype(float).to_numpy() if use_weights else np.ones_like(x_all, dtype=float)
    overall = weighted_mean_sd(x_all, w_all)

    # strata mean/sd
    strata_lines = []
    for strata, g in tmp.groupby("__STRATA__"):
        x = g[q_col].astype(float).to_numpy()
        w = g["_w"].astype(float).to_numpy() if use_weights else np.ones_like(x, dtype=float)
        m, s = weighted_mean_sd(x, w)
        strata_lines.append((str(strata), m, s))
    strata_lines.sort(key=lambda t: t[0])

    return pivot, overall, strata_lines

# =============================
# 6) Plot (비율로 보기=전체합 100% 강제)
# =============================
def plot_pivot(pivot_counts: pd.DataFrame, as_percent: bool, barmode: str, use_weights: bool):
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    x_labels = ["1", "2", "3", "4", "5"]

    if pivot_counts.empty or pivot_counts.shape[1] == 0:
        ax.set_xticks(range(5))
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("응답(1~5)")
        ax.set_ylabel("비율(%)" if as_percent else ("가중빈도" if use_weights else "빈도"))
        ax.text(0.5, 0.5, "유효 응답 없음", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylim(bottom=0)  # ✅ 0부터만 시작
        fig.tight_layout()
        return fig

    y = pivot_counts.copy()

    # ✅ 비율로 보기: 전체합=100% (항상)
    if as_percent:
        total = y.values.sum()
        y = (y / total * 100.0) if total > 0 else y * 0.0

    if barmode == "group":
        n = len(y.columns)
        x = np.arange(len(y.index))
        width = 0.8 / max(n, 1)
        for i, col in enumerate(y.columns):
            ax.bar(x + (i - (n - 1) / 2) * width, y[col].values, width=width, label=str(col))
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
    else:  # stack
        bottom = np.zeros(len(y))
        for col in y.columns:
            ax.bar(x_labels, y[col].values, bottom=bottom, label=str(col))
            bottom += y[col].values

    ax.set_xlabel("응답(1~5)")
    ax.set_ylabel("비율(%) (전체합=100)" if as_percent else ("가중빈도" if use_weights else "빈도"))

    ax.set_ylim(bottom=0)  # ✅ 자동 스케일 + 0부터 시작 (0~100 고정 안 함)
    ax.legend(title="층", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


# =============================
# 7) UI
# =============================
st.title("층화가중 기반 리커트 분포")

c1, c2 = st.columns(2)
with c1:
    sample_file = st.file_uploader("표본(CSV) 업로드", type=["csv"], key="sample")
with c2:
    pop_file = st.file_uploader("모집단(XLSX) 업로드", type=["xlsx", "xls"], key="pop")

if sample_file is None or pop_file is None:
    st.info("표본(CSV)과 모집단(XLSX)을 모두 업로드하면 시작합니다.")
    st.stop()

df_samp = read_csv_robust(sample_file)
df_pop = load_pop_xlsx(pop_file, "모집단")

# 사이드바: 표시지표/막대방식 유지 + 비율로 보기(전체합=100%) + 층화 선택
st.sidebar.header("설정")
strata_key = st.sidebar.selectbox("층화 변수(개념)", list(STRATA_MAP.keys()))
metric_mode = st.sidebar.radio("표시 지표", ["가중치 적용(모집단 추정치)", "표본 그대로(미가중)"], index=0)
use_weights = (metric_mode == "가중치 적용(모집단 추정치)")
barmode = st.sidebar.radio("막대 방식", ["stack", "group"], index=0)
as_percent = st.sidebar.checkbox("비율로 보기", value=True)

samp_strata_col, pop_strata_col = STRATA_MAP[strata_key]

# 컬럼 검증
missing = []
if samp_strata_col not in df_samp.columns:
    missing.append(f"표본에 '{samp_strata_col}' 없음")
if pop_strata_col not in df_pop.columns:
    missing.append(f"모집단에 '{pop_strata_col}' 없음")
if missing:
    st.error(" / ".join(missing))
    st.stop()

# 문항: 표본 6~27열
if df_samp.shape[1] < 27:
    st.error("표본 데이터 컬럼 수가 27개 미만입니다. (6~27열 문항 가정)")
    st.stop()
q_cols = list(df_samp.columns[5:27])

# 가중치 계산(항상 계산해두고, use_weights에 따라 사용/미사용만 분기)
df_samp_w = compute_poststrat_weights(df_samp, df_pop, samp_strata_col, pop_strata_col)

# =============================
# Pagination (page number buttons)
# =============================
PER_PAGE = 4
GRID_COLS = 2

total_pages = math.ceil(len(q_cols) / PER_PAGE)
if "page" not in st.session_state:
    st.session_state.page = 1

st.markdown(f"**페이지 선택 (현재: {st.session_state.page} / {total_pages})**")

BTN_PER_ROW = 10
for p0 in range(1, total_pages + 1, BTN_PER_ROW):
    p1 = min(total_pages, p0 + BTN_PER_ROW - 1)
    cols = st.columns(p1 - p0 + 1)
    for i, p in enumerate(range(p0, p1 + 1)):
        if cols[i].button(f"{p}", key=f"pagebtn_{p}", disabled=(p == st.session_state.page)):
            st.session_state.page = p
            st.rerun()

page = int(st.session_state.page)
start = (page - 1) * PER_PAGE
end = min(page * PER_PAGE, len(q_cols))
page_qs = q_cols[start:end]

st.subheader(f"층화: {strata_key} | 문항 {start+1}~{end}")

# =============================
# Render 4 per page (2x2): 시각화 + 평균/표준편차
# =============================
rows = math.ceil(len(page_qs) / GRID_COLS)
idx = 0

for _ in range(rows):
    row_cols = st.columns(GRID_COLS)
    for j in range(GRID_COLS):
        if idx >= len(page_qs):
            break

        q = page_qs[idx]
        pivot, (m_all, s_all), strata_lines = prepare_question(df_samp_w, q, use_weights=use_weights)

        with row_cols[j]:
            st.markdown(f"#### {q}")
            left, right = st.columns([3, 2])

            with left:
                fig = plot_pivot(pivot, as_percent=as_percent, barmode=barmode, use_weights=use_weights)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            with right:
                if not np.isfinite(m_all):
                    st.write("유효 응답 없음")
                else:
                    st.markdown("**전체(평균/표준편차)**")
                    st.write(f"- 평균: **{m_all:.2f}**")
                    st.write(f"- 표준편차: **{s_all:.2f}**")

                    st.markdown("**층별(평균/표준편차)**")
                    if len(strata_lines) == 0:
                        st.write("층별 유효 응답 없음")
                    else:
                        lines = [f"- {name}: 평균 {m:.2f}, 표준편차 {s:.2f}" for name, m, s in strata_lines]
                        st.markdown("\n".join(lines))

        idx += 1
