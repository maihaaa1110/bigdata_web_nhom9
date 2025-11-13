# etl_three_functions.py
import os
import time
from io import BytesIO
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

import pandas as pd
import numpy as np
import requests

# CONFIG
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "KXHaneBxKmIC0_oLJdUKqhRh4if7DsCz")
PROCESSED_CONTAINER_SAS_URL = os.environ.get(
    "PROCESSED_CONTAINER_SAS_URL",
    None  # set if you want azure fallback for news
)
MODEL_NAME = os.environ.get("SENT_MODEL", "yiyanghkust/finbert-tone")

SYMBOL_MAP = {
    '^GSPC': 'sp500',
    'SPY': 'spy',
    '^VIX': 'vix',
    'GC=F': 'gold',
    'CL=F': 'oil',
    'DX-Y.NYB': 'usd_index',
    'UUP': 'uup'
}
TICKERS = list(SYMBOL_MAP.keys())
BASE_FIELDS = ['open', 'high', 'low', 'close', 'volume']

# small helpers
def safe_request(url, params=None, headers=None, max_retry=4, sleep_sec=2, timeout=30):
    for attempt in range(1, max_retry+1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code in (429, 500, 502, 503):
                wait = sleep_sec * (2 ** (attempt-1))
                print(f"safe_request: status {r.status_code}. retry {attempt}/{max_retry} after {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            wait = sleep_sec * (2 ** (attempt-1))
            print(f"safe_request: failed attempt {attempt}/{max_retry}: {e}. wait {wait}s")
            time.sleep(wait)
    return None

# -----------------------
# 1) FINANCE LOADER -> df1
# -----------------------
def load_finance_data(date_str: str) -> pd.DataFrame:
    """
    Return DataFrame with rows for each ticker (symbol,date,open,high,low,close,volume)
    Keeps the robust fallback behavior: bulk download -> single download -> last-7d fallback.
    """
    import yfinance as yf
    date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
    start = date_str
    end = (date_obj + timedelta(days=1)).strftime("%Y-%m-%d")

    df_bulk = None
    try:
        # avoid FutureWarning by setting auto_adjust explicitly
        df_bulk = yf.download(TICKERS, start=start, end=end, group_by='ticker', progress=False, threads=False, auto_adjust=False)
    except Exception as e:
        print("load_finance_data: bulk download failed:", e)
        df_bulk = None

    def _safe_float(v):
        try:
            if v is None:
                return None
            if v != v:
                return None
            return float(v)
        except Exception:
            return None

    def _parse_yf_row(row: dict):
        return {
            "open": _safe_float(row.get("Open")),
            "high": _safe_float(row.get("High")),
            "low": _safe_float(row.get("Low")),
            "close": _safe_float(row.get("Close")),
            "volume": int(row.get("Volume")) if row.get("Volume") is not None and row.get("Volume") == row.get("Volume") else None
        }

    results = []
    for sym in TICKERS:
        row_vals = {"symbol": sym, "date": date_str, "open": None, "high": None, "low": None, "close": None, "volume": None}
        if df_bulk is not None and not df_bulk.empty:
            try:
                if isinstance(df_bulk.columns, pd.MultiIndex) and sym in list(df_bulk.columns.get_level_values(0)):
                    sub = df_bulk[sym]
                    if sub is not None and not sub.empty:
                        series_row = sub.dropna(how='all').iloc[0]
                        row_vals.update(_parse_yf_row(dict(series_row)))
                else:
                    # single-symbol bulk
                    series_row = df_bulk.dropna(how='all').iloc[0]
                    row_vals.update(_parse_yf_row(dict(series_row)))
            except Exception:
                pass

        # fallback single ticker
        if row_vals["close"] is None:
            try:
                df_single = yf.download(sym, start=start, end=end, progress=False, threads=False, auto_adjust=False)
                if df_single is not None and not df_single.empty:
                    series_row = df_single.dropna(how='all').iloc[0]
                    row_vals.update(_parse_yf_row(dict(series_row)))
            except Exception:
                pass

        # final fallback last 7 days
        if row_vals["close"] is None:
            try:
                df_hist = yf.download(sym, end=end, period="7d", progress=False, threads=False, auto_adjust=False)
                if df_hist is not None and not df_hist.empty:
                    series_row = df_hist.dropna(how='all').iloc[-1]
                    row_vals.update(_parse_yf_row(dict(series_row)))
            except Exception:
                pass

        results.append(row_vals)
        time.sleep(0.02)

    df_fin = pd.DataFrame(results)
    print(f"load_finance_data: finished for {date_str}, rows={len(df_fin)}")
    return df_fin

# -----------------------
# 2) NEWS LOADER -> df2
#    supports 'polygon' (default) or 'azure_blob' if PROCESSED_CONTAINER_SAS_URL provided.
# -----------------------
def split_container_sas_url(container_sas_url: str) -> Tuple[str, str]:
    if "?" not in container_sas_url:
        raise ValueError("container SAS URL must contain ? and query string")
    base, query = container_sas_url.split("?", 1)
    return base.rstrip("/"), query

def load_news_data(date_str: str, source: str = "polygon", azure_sas_url: Optional[str] = None) -> pd.DataFrame:
    """
    Return DataFrame of raw article records for date_str (YYYY-MM-DD).
    source: "polygon" (call Polygon API) or "azure_blob" (download from processed blob container).
    If using azure_blob, azure_sas_url must be provided (full container SAS URL).
    """
    # polygon path
    if source == "polygon":
        url = "https://api.polygon.io/v2/reference/news"
        headers = {"Authorization": f"Bearer {POLYGON_API_KEY}"} if POLYGON_API_KEY else None
        params = {
            "published_utc.gte": f"{date_str}T00:00:00Z",
            "published_utc.lte": f"{date_str}T23:59:59Z",
            "limit": 50
        }
        resp = safe_request(url, params=params, headers=headers)
        articles = []
        if resp and "results" in resp and resp["results"]:
            for a in resp["results"]:
                a["source"] = "Polygon"
                a["published_date"] = a.get("published_utc", "")[:10]
                articles.append(a)
            print(f"load_news_data: Polygon fetched {len(articles)} articles for {date_str}")
        else:
            print(f"load_news_data: Polygon returned no articles for {date_str}")
        return pd.DataFrame(articles)

    # azure blob path
    elif source == "azure_blob":
        if azure_sas_url is None:
            azure_sas_url = PROCESSED_CONTAINER_SAS_URL
        if azure_sas_url is None:
            raise ValueError("azure_sas_url must be provided for source='azure_blob'")

        # lazy import to avoid requirement unless used
        try:
            from azure.storage.blob import BlobServiceClient
        except Exception as e:
            raise ImportError("azure-storage-blob is required for source='azure_blob'. Install with `pip install azure-storage-blob`") from e

        base, sas = split_container_sas_url(azure_sas_url)
        # container name is last path segment after base
        base_no_proto = base.replace("https://", "").rstrip("/")
        parts = base_no_proto.split("/")
        account_url = "https://" + parts[0]
        container_name = parts[1] if len(parts) > 1 else ""
        if not container_name:
            raise ValueError("Cannot parse container name from SAS URL")

        bsc = BlobServiceClient(account_url=account_url, credential=sas)
        container_client = bsc.get_container_client(container_name)

        # look for blobs under sentiment/ or sentiment/date=YYYY-MM-DD
        prefix_candidates = [f"sentiment/date={date_str}", f"sentiment/{date_str}", f"sentiment/{date_str}/"]
        blobs = []
        for pref in prefix_candidates:
            try:
                it = container_client.list_blobs(name_starts_with=pref)
                for b in it:
                    if b.name and not b.name.endswith("/"):
                        blobs.append(b.name)
            except Exception:
                pass
            if blobs:
                break

        articles = []
        for blob_name in blobs:
            try:
                data = container_client.download_blob(blob_name).readall()
                # try parse as parquet / csv / json lines
                try:
                    # try parquet
                    from io import BytesIO
                    tmp = BytesIO(data)
                    df = pd.read_parquet(tmp)
                    articles.extend(df.to_dict(orient="records"))
                except Exception:
                    try:
                        txt = data.decode("utf-8")
                        # try jsonlines or csv
                        try:
                            for line in txt.splitlines():
                                if line.strip():
                                    articles.append(pd.io.json.loads(line))
                        except Exception:
                            try:
                                df = pd.read_csv(BytesIO(data))
                                articles.extend(df.to_dict(orient="records"))
                            except Exception:
                                pass
                    except Exception:
                        pass
            except Exception as e:
                print("load_news_data azure: failed to download blob", blob_name, e)

        if articles:
            for a in articles:
                if isinstance(a, dict):
                    a.setdefault("source", "AzureBlob")
                    a.setdefault("published_date", a.get("published_date") or a.get("published_utc","")[:10])
            print(f"load_news_data: Azure fetched {len(articles)} articles for {date_str} from blobs")
        else:
            print(f"load_news_data: Azure returned no articles for {date_str}")

        return pd.DataFrame(articles)

    else:
        raise ValueError("Unknown source for load_news_data: " + str(source))

# -----------------------
# 3) SCORE SENTIMENT -> df3 (articles + sentiment columns)
# -----------------------
def load_sentiment_pipeline(model_name: str = MODEL_NAME):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    print("Loading sentiment model:", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=-1)
    return nlp

def infer_sentiment_for_texts(nlp, texts: List[str], batch_size: int = 16) -> List[tuple]:
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        cleaned = [t if (t is not None and str(t).strip() != "") else "" for t in batch]
        try:
            preds = nlp(cleaned)
            for p in preds:
                label = p.get("label","NEUTRAL").upper()
                score = float(p.get("score",0.0))
                score_num = {"POSITIVE":1, "NEUTRAL":0, "NEGATIVE":-1}.get(label, 0)
                results.append((label, score, score_num))
        except Exception as e:
            print("infer_sentiment_for_texts: batch inference failed:", e)
            for _ in batch:
                results.append(("NEUTRAL", 0.0, 0))
    return results

def score_sentiment(df_news: pd.DataFrame, nlp=None, load_model: bool = True, batch_size: int = 16) -> pd.DataFrame:
    """
    Input: df_news (raw articles), possibly empty
    Output: df_scored with additional columns:
      - text_for_sentiment
      - sentiment_label (POSITIVE/NEUTRAL/NEGATIVE)
      - sentiment_score_prob (float)
      - sentiment_score (1/0/-1)
    If nlp is None and load_model True -> load model once.
    """
    if df_news is None:
        df_news = pd.DataFrame()

    # ensure DataFrame (some callers may pass list)
    if not isinstance(df_news, pd.DataFrame):
        df_news = pd.DataFrame(df_news)

    # ensure needed columns and text extraction
    if df_news.empty:
        # return empty df with sentiment columns to keep schema consistent
        cols = list(df_news.columns) + ["text_for_sentiment", "sentiment_label", "sentiment_score_prob", "sentiment_score"]
        return pd.DataFrame(columns=cols)

    def prepare_text_row(row):
        # use title/headline and description/summary
        title = row.get("title") or row.get("headline") or ""
        desc = row.get("description") or row.get("summary") or ""
        return f"{title}. {desc}".strip()

    df = df_news.copy()
    df["text_for_sentiment"] = df.apply(lambda r: prepare_text_row(r.to_dict()), axis=1)
    texts = df["text_for_sentiment"].astype(str).tolist()

    if nlp is None and load_model:
        nlp = load_sentiment_pipeline()

    if nlp is not None and len(texts) > 0:
        preds = infer_sentiment_for_texts(nlp, texts, batch_size=batch_size)
        labels, probs, nums = zip(*preds)
        df["sentiment_label"] = labels
        df["sentiment_score_prob"] = probs
        df["sentiment_score"] = nums
    else:
        df["sentiment_label"] = "NEUTRAL"
        df["sentiment_score_prob"] = 0.0
        df["sentiment_score"] = 0

    return df

# -----------------------
# Helper: aggregate scored -> summary dict (same behavior as previous aggregate_daily_sentiment)
# -----------------------
def get_sentiment_agg_from_scored(df_scored: pd.DataFrame, date_str: Optional[str] = None) -> Dict:
    """
    From scored articles DataFrame produce summary dict with keys:
      date, n_articles, n_positive, n_neutral, n_negative, prop_*, mean_sentiment_score, mean_sentiment_prob, ...
    If df_scored empty -> return neutral default with date=date_str
    """
    if df_scored is None or df_scored.empty:
        return {
            "date": date_str,
            "n_articles": 0, "n_positive": 0, "n_neutral": 0, "n_negative": 0,
            "prop_positive": float("nan"), "prop_neutral": float("nan"), "prop_negative": float("nan"),
            "mean_sentiment_score": float("nan"), "mean_sentiment_prob": float("nan"),
            "median_sentiment_prob": float("nan"), "std_sentiment_score": float("nan"),
            "weighted_sentiment_score": float("nan"), "avg_text_len": float("nan"), "median_text_len": float("nan")
        }

    df = df_scored.copy()
    labels_u = df["sentiment_label"].astype(str).str.upper().fillna("NEUTRAL")
    n = len(df)
    n_pos = int((labels_u == "POSITIVE").sum())
    n_neu = int((labels_u == "NEUTRAL").sum())
    n_neg = int((labels_u == "NEGATIVE").sum())

    prop_pos = n_pos / n if n > 0 else float("nan")
    prop_neu = n_neu / n if n > 0 else float("nan")
    prop_neg = n_neg / n if n > 0 else float("nan")

    scores = pd.to_numeric(df["sentiment_score"], errors="coerce").fillna(0.0).astype(float)
    probs_num = pd.to_numeric(df["sentiment_score_prob"], errors="coerce").fillna(0.0).astype(float)

    mean_score = float(scores.mean()) if n > 0 else float("nan")
    mean_prob = float(probs_num.mean()) if n > 0 else float("nan")
    median_prob = float(probs_num.median()) if n > 0 else float("nan")
    std_score = float(scores.std(ddof=0)) if n > 0 else float("nan")
    denom = probs_num.sum()
    weighted_score = float((scores * probs_num).sum() / denom) if denom > 0 else mean_score

    text_lens = df["text_for_sentiment"].astype(str).map(len)
    avg_text_len = float(text_lens.mean()) if n > 0 else float("nan")
    median_text_len = float(text_lens.median()) if n > 0 else float("nan")

    return {
        "date": date_str,
        "n_articles": int(n),
        "n_positive": n_pos,
        "n_neutral": n_neu,
        "n_negative": n_neg,
        "prop_positive": prop_pos,
        "prop_neutral": prop_neu,
        "prop_negative": prop_neg,
        "mean_sentiment_score": mean_score,
        "mean_sentiment_prob": mean_prob,
        "median_sentiment_prob": median_prob,
        "std_sentiment_score": std_score,
        "weighted_sentiment_score": weighted_score,
        "avg_text_len": avg_text_len,
        "median_text_len": median_text_len
    }

# -----------------------
# Example orchestration using the three functions
# -----------------------
if __name__ == "__main__":
    # choose date
    date = "2025-11-10"   # sample date that actually has news in your previous run
    # 1) finance -> df1
    df1 = load_finance_data(date)
    print("df1 (finance):")
    print(df1.head())

    # 2) news -> df2 (try polygon first; if you prefer azure set source='azure_blob' and pass SAS)
    df2 = load_news_data(date, source="polygon")
    print("df2 (raw news) shape:", df2.shape)
    print(df2.head())

    # 3) score -> df3
    # If you process many dates, load_sentiment_pipeline() once and pass nlp to score_sentiment to avoid reloading
    nlp = None
    try:
        nlp = load_sentiment_pipeline()
    except Exception as e:
        print("Warning: cannot load model locally, proceeding with load_model fallback per-call:", e)
        nlp = None

    df3 = score_sentiment(df2, nlp=nlp, load_model=(nlp is None))
    print("df3 (scored) sample:")
    print(df3[["text_for_sentiment","sentiment_label","sentiment_score_prob","sentiment_score"]].head())

    # get summary
    sent_agg = get_sentiment_agg_from_scored(df3, date_str=date)
    print("sent_agg:", sent_agg)
