# -*- coding: utf-8 -*-
import os, re, io, time
from urllib.parse import urlparse, parse_qs
from collections import Counter

import requests
import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

# ====================== Настройка ======================
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
st.set_page_config(page_title="Анализ тональности", page_icon="💬", layout="wide")
# Универсальные стили для метрик (чтобы выглядело красиво и на белой теме)
# 💅 Стили метрик (чтобы выглядело корректно и на белой теме)
st.markdown("""
    <style>
    /* Контейнер метрик */
    div[data-testid="stMetricValue"] {
        color: #111 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #333 !important;
        font-weight: 600 !important;
    }
    /* Весь блок метрики (фон и тень) */
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #ddd !important;
        border-radius: 12px !important;
        padding: 15px !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    /* Чтобы убрать возможный прозрачный overlay от темы */
    [data-testid="stMetricDelta"] {
        color: #444 !important;
    }
    </style>
""", unsafe_allow_html=True)


VK_API_VERSION = "5.131"

# ====================== Настройки модели и API ======================
with st.expander("⚙️ Настройки API и модели"):
    vk_token_ui = st.text_input("VK_TOKEN", os.getenv("VK_TOKEN", ""), type="password")
    yt_ui = st.text_input("YOUTUBE_API_KEY", os.getenv("YOUTUBE_API_KEY", ""), type="password")
    model_choice = st.selectbox(
        "Модель для анализа тональности:",
        [
            "cointegrated/rubert-tiny-sentiment-balanced (быстрая)",
            "cointegrated/rubert-base-cased-sentiment-balanced (сбалансированная)",
            "blanchefort/rubert-base-cased-sentiment (точная)"
        ],
        index=0
    )

# ====================== Модель ======================
@st.cache_resource(show_spinner=False)
def load_pipeline(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    return TextClassificationPipeline(model=mdl, tokenizer=tok, top_k=None, truncation=True, device=-1)

def classify_many(texts: list[str], model_name: str) -> list[str]:
    clf = load_pipeline(model_name)
    out_labels = []
    for t in texts:
        t = (t or "").strip()
        if not t:
            out_labels.append("Нейтрал")
            continue
        out = clf(t[:512])[0]
        best = max(out, key=lambda x: x["score"])
        lab = best["label"].upper()
        if lab == "POSITIVE":
            out_labels.append("Позитив")
        elif lab == "NEGATIVE":
            out_labels.append("Негатив")
        else:
            out_labels.append("Нейтрал")
    return out_labels

# ====================== Вспомогательные ======================
def detect_platform(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    if "vk.com" in host:
        return "VK"
    if "youtube.com" in host or "youtu.be" in host:
        return "YOUTUBE"
    return "UNKNOWN"

# VK
VK_WALL_RE = re.compile(r"wall(?P<owner>-?\d+)_(?P<post>\d+)")
def vk_extract_ids(url: str):
    m = VK_WALL_RE.search(url)
    if not m:
        q = parse_qs(urlparse(url).query).get("w", [""])[0]
        m = VK_WALL_RE.search(q)
    if not m:
        raise ValueError("VK: не удалось извлечь owner_id и post_id.")
    return int(m.group("owner")), int(m.group("post"))

def vk_call(token: str, method: str, **params):
    params.update({"access_token": token, "v": VK_API_VERSION})
    r = requests.get(f"https://api.vk.com/method/{method}", params=params, timeout=30)
    j = r.json()
    if "error" in j:
        e = j["error"]
        raise RuntimeError(f"VK API error {e.get('error_code')}: {e.get('error_msg')}")
    return j["response"]

def fetch_vk_comments(token: str, url: str):
    owner_id, post_id = vk_extract_ids(url)
    all_comments, offset = [], 0
    while True:
        resp = vk_call(
            token, "wall.getComments",
            owner_id=owner_id, post_id=post_id,
            count=100, offset=offset,
            sort="asc", thread_items_count=10, extended=0
        )
        items = resp.get("items", [])
        if not items:
            break
        all_comments.extend(items)
        offset += len(items)
        if len(items) < 100:
            break
        time.sleep(0.25)

    texts, ids, parents, dates = [], [], [], []
    for it in all_comments:
        t = (it.get("text") or "").strip()
        if t:
            texts.append(t)
            ids.append(it.get("id"))
            parents.append(None)
            dates.append(pd.to_datetime(it.get("date", 0), unit="s", utc=True))
        for r in (it.get("thread") or {}).get("items", []):
            tr = (r.get("text") or "").strip()
            if tr:
                texts.append(tr)
                ids.append(r.get("id"))
                parents.append(it.get("id"))
                dates.append(pd.to_datetime(r.get("date", 0), unit="s", utc=True))
    return ids, parents, texts, dates

# YouTube
def extract_youtube_id(url: str) -> str:
    q = parse_qs(urlparse(url).query)
    if "v" in q:
        return q["v"][0]
    m = re.search(r"youtu\.be/([A-Za-z0-9_-]+)", url)
    if m:
        return m.group(1)
    raise ValueError("Невозможно извлечь YouTube video ID.")

def fetch_youtube_comments(api_key: str, url: str):
    video_id = extract_youtube_id(url)
    comments, ids, parents, dates = [], [], [], []
    next_page = None
    while True:
        r = requests.get(
            "https://www.googleapis.com/youtube/v3/commentThreads",
            params={
                "part": "snippet",
                "videoId": video_id,
                "maxResults": 100,
                "key": api_key,
                "pageToken": next_page,
            },
            timeout=30,
        ).json()
        for item in r.get("items", []):
            sn = item["snippet"]["topLevelComment"]["snippet"]
            txt = (sn.get("textDisplay") or "").strip()
            if txt:
                comments.append(txt)
                ids.append(item["id"])
                parents.append(None)
                dates.append(pd.to_datetime(sn["publishedAt"], utc=True))
        next_page = r.get("nextPageToken")
        if not next_page:
            break
        time.sleep(0.2)
    return ids, parents, comments, dates

# ====================== Анализ ======================
def analyze_one(url: str, vk_token: str, yt_key: str, model_name: str):
    platform = detect_platform(url)
    if platform == "VK":
        ids, parents, texts, dates = fetch_vk_comments(vk_token, url)
    elif platform == "YOUTUBE":
        ids, parents, texts, dates = fetch_youtube_comments(yt_key, url)
    else:
        raise RuntimeError("Неподдерживаемая платформа.")
    labels = classify_many(texts, model_name)
    df = pd.DataFrame({
        "Платформа": platform,
        "Ссылка": url,
        "Комментарий ID": ids,
        "Родитель ID": parents,
        "Текст": texts,
        "Дата": dates,
        "Тональность": labels
    })
    cnt = Counter(labels)
    total = max(1, len(labels))
    summary = {
        "Позитив (%)": round(100 * cnt.get("Позитив", 0) / total, 1),
        "Нейтрал (%)": round(100 * cnt.get("Нейтрал", 0) / total, 1),
        "Негатив (%)": round(100 * cnt.get("Негатив", 0) / total, 1),
        "Позитив": cnt.get("Позитив", 0),
        "Нейтрал": cnt.get("Нейтрал", 0),
        "Негатив": cnt.get("Негатив", 0),
        "Комментариев": len(texts),
    }
    return summary, df

# ====================== UI ======================
urls_raw = st.text_area(
    "Ссылки (VK, YouTube — по одной в строке)",
    height=150,
    placeholder="https://vk.com/wall-141155426_420521\nhttps://www.youtube.com/watch?v=dQw4w9WgXcQ",
)
show_table = st.checkbox("Показывать таблицу комментариев", value=False)
go = st.button("Анализировать", type="primary")

if "summary_df" not in st.session_state:
    st.session_state["summary_df"] = None
if "all_df" not in st.session_state:
    st.session_state["all_df"] = None

if go:
    urls = [u.strip() for u in (urls_raw or "").splitlines() if u.strip()]
    progress = st.progress(0.0)
    per_link, frames = [], []
    for i, url in enumerate(urls, start=1):
        try:
            s, df = analyze_one(url, vk_token_ui, yt_ui, model_choice.split()[0])
            per_link.append({"Ссылка": url, "Платформа": df.iloc[0]["Платформа"], **s})
            frames.append(df)
        except Exception as e:
            per_link.append({"Ссылка": url, "Платформа": "?", "Ошибка": str(e)})
        progress.progress(i / len(urls))
    if frames:
        st.session_state["summary_df"] = pd.DataFrame(per_link)
        st.session_state["all_df"] = pd.concat(frames, ignore_index=True)

# ====================== Отображение результатов ======================
if st.session_state["summary_df"] is not None:
    summary_df = st.session_state["summary_df"]
    all_df = st.session_state["all_df"]

    # карточки общей сводки
    st.subheader("📊 Общая сводка")
    counts = all_df["Тональность"].value_counts().to_dict()
    total = max(1, len(all_df))
    totals = {
        "Позитив": (round(100 * counts.get("Позитив", 0) / total, 1), counts.get("Позитив", 0), "green"),
        "Нейтрал": (round(100 * counts.get("Нейтрал", 0) / total, 1), counts.get("Нейтрал", 0), "gold"),
        "Негатив": (round(100 * counts.get("Негатив", 0) / total, 1), counts.get("Негатив", 0), "red"),
    }

    # три горизонтальные колонки
    c1, c2, c3 = st.columns(3)
    for col, (label, (p, n, color)) in zip([c1, c2, c3], totals.items()):
        col.markdown(
            f"""
            <style>
            @media (prefers-color-scheme: light) {{
                .metric-card {{
                    background-color: #ffffff;
                    border: 1px solid #e6e6e6;
                    color: #333333;
                }}
            }}
            @media (prefers-color-scheme: dark) {{
                .metric-card {{
                    background-color: #2b2b2b;
                    border: 1px solid #444444;
                    color: #dddddd;
                }}
            }}
            </style>

            <div class="metric-card" style="
                border-radius:12px;
                padding:1.5rem;
                text-align:center;
                box-shadow:0 2px 6px rgba(0,0,0,0.15);
                transition:transform 0.2s ease;
            ">
                <h4 style="color:{color}; margin-bottom:0.4rem;">{label}</h4>
                <h2 style="color:{color}; margin:0; font-size:2.2rem;">{p}%</h2>
                <p style="margin-top:0.3rem; font-size:14px;">{n} комментариев</p>
            </div>
            """,
            unsafe_allow_html=True
        )




    # Круговая диаграмма
    pie_df = pd.DataFrame({"Тональность": list(totals.keys()), "Процент": [v[0] for v in totals.values()]})
    fig = px.pie(
        pie_df, names="Тональность", values="Процент", hole=0.35,
        color="Тональность", color_discrete_map={"Позитив":"green","Нейтрал":"gold","Негатив":"red"},
        title="Общая тональность по всем комментариям"
    )
    st.plotly_chart(fig, use_container_width=True)


    # ---------- График по ссылкам ----------
    st.subheader("📈 Распределение по ссылкам")
    fig2 = px.bar(
        summary_df.melt(id_vars=["Ссылка", "Платформа"], value_vars=["Позитив", "Нейтрал", "Негатив"]),
        x="Ссылка", y="value", color="variable", barmode="stack",
        color_discrete_map={"Позитив":"green","Нейтрал":"gold","Негатив":"red"},
        labels={"value":"Количество","variable":"Тональность"}
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ---------- График по времени ----------
    st.subheader("🕒 Распределение комментариев по времени")
    if "Дата" in all_df.columns:
        tmp = all_df.dropna(subset=["Дата"]).copy()
        if tmp["Дата"].dt.tz is None:
            tmp["Дата"] = tmp["Дата"].dt.tz_localize("UTC")
        tmp["Дата"] = tmp["Дата"].dt.tz_convert(None)
        tmp["Дата_день"] = tmp["Дата"].dt.date
        time_df = tmp.groupby(["Дата_день", "Тональность"]).size().reset_index(name="count")
        fig_time = px.line(
            time_df, x="Дата_день", y="count", color="Тональность",
            color_discrete_map={"Позитив":"green","Нейтрал":"gold","Негатив":"red"},
            markers=True
        )
        st.plotly_chart(fig_time, use_container_width=True)

    # ---------- Сводка и таблица ----------
    st.subheader("📊 Сводка по ссылкам")
    st.dataframe(summary_df, use_container_width=True, height=300)
    if show_table:
        st.subheader("💬 Все комментарии")
        st.dataframe(all_df, use_container_width=True, height=430)

    # ---------- Выгрузка ----------
    st.markdown("### 💾 Выгрузка")
    csv_data = all_df.to_csv(index=False).encode("utf-8")
    st.download_button("📄 Скачать CSV", data=csv_data, file_name="sentiment_comments.csv", mime="text/csv")

    # Excel (2 листа)
    xls_buf = io.BytesIO()
    try:
        if all_df["Дата"].dt.tz is not None:
            all_df["Дата"] = all_df["Дата"].dt.tz_convert(None)
    except Exception:
        pass
    with pd.ExcelWriter(xls_buf, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Сводка", index=False)
        all_df.to_excel(writer, sheet_name="Комментарии", index=False)
    xls_buf.seek(0)
    st.download_button(
        "📊 Скачать Excel",
        data=xls_buf,
        file_name="sentiment_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info("Добавьте ссылки и нажмите «Анализировать».")
