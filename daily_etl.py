# etl/daily_etl.py
"""
Daily ETL: fetch finance (yfinance) + news (Polygon) for TWO days (target_date and prev_date)
→ compute features (59 variables, excludes market_direction target) →
→ upload single-row parquet for target_date to Azure Blob (processed container).
"""
import os
import time
import math
import requests
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

# -----------------------
# CONFIG - set via ENV or replace here
# -----------------------
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "KXHaneBxKmIC0_oLJdUKqhRh4if7DsCz")
PROCESSED_CONTAINER_SAS_URL = os.environ.get(
    "PROCESSED_CONTAINER_SAS_URL",
    "https://finnewsstorageproject123.blob.core.windows.net/processed-news?sp=racwdli&st=2025-11-03T12:02:36Z&se=2025-11-14T20:17:36Z&spr=https&sv=2024-11-04&sr=c&sig=yHHatzKgqEDuettuOHPdw%2Fj94jh%2FbEcP9meaeY%2B81F0%3D"
)
MODEL_NAME = os.environ.get("SENT_MODEL", "yiyanghkust/finbert-tone")  # finbert-tone or appropriate
# Tickers mapping: ticker -> short name used in final columns
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

# HTTP / upload timeouts
READ_TIMEOUT = 60
UPLOAD_TIMEOUT = 180

# -----------------------
# Helpers: SAS parsing + upload
# -----------------------
def split_container_sas_url(container_sas_url: str) -> Tuple[str, str]:
    if "?" not in container_sas_url:
        raise ValueError("container SAS URL must contain ? and query string")
    base, query = container_sas_url.split("?", 1)
    return base.rstrip("/"), query

PROCESSED_BASE, PROCESSED_SAS = split_container_sas_url(PROCESSED_CONTAINER_SAS_URL)

def upload_bytes_to_blob(bytes_data: bytes, dest_blob_name: str) -> bool:
    url = f"{PROCESSED_BASE}/{dest_blob_name}?{PROCESSED_SAS}"
    headers = {"x-ms-blob-type":"BlockBlob","Content-Type":"application/octet-stream"}
    try:
        r = requests.put(url, data=bytes_data, headers=headers, timeout=UPLOAD_TIMEOUT)
        if r.status_code in (200,201):
            print(f"Uploaded to {dest_blob_name}")
            return True
        else:
            print("Upload failed:", r.status_code, r.text[:300])
            return False
    except Exception as e:
        print("Upload exception:", e)
        return False

# -----------------------
# Safe request
# -----------------------
def safe_request(url, params=None, headers=None, max_retry=4, sleep_sec=2, timeout=30):
    for attempt in range(1, max_retry+1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code in (429, 500, 502, 503):
                wait = sleep_sec * (2 ** (attempt-1))
                print(f"Request {url} status {r.status_code}. Retry {attempt}/{max_retry} after {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            wait = sleep_sec * (2 ** (attempt-1))
            print(f"Request failed attempt {attempt}/{max_retry}: {e}. Retry after {wait}s")
            time.sleep(wait)
    return None

# -----------------------
# Crawl Polygon for one day
# -----------------------
def crawl_polygon_single_day(date_str: str) -> List[Dict]:
    articles = []
    url = "https://api.polygon.io/v2/reference/news"
    headers = {"Authorization": f"Bearer {POLYGON_API_KEY}"}
    params = {
        "published_utc.gte": f"{date_str}T00:00:00Z",
        "published_utc.lte": f"{date_str}T23:59:59Z",
        "limit": 50
    }
    resp = safe_request(url, params=params, headers=headers)
    if resp and "results" in resp:
        for a in resp["results"]:
            a["source"] = "Polygon"
            a["published_date"] = a.get("published_utc", "")[:10]
        articles.extend(resp["results"])
        print(f"Polygon {date_str}: fetched {len(resp['results'])} articles")
    else:
        print(f"Polygon {date_str}: no results")
    return articles

# -----------------------
# Finance fetch (yfinance) - resilient
# -----------------------
def _safe_float(v):
    try:
        if v is None:
            return None
        if v != v:
            return None
        return float(v)
    except Exception:
        return None

def _parse_yf_row(row: dict) -> Dict:
    return {
        "open": _safe_float(row.get("Open")),
        "high": _safe_float(row.get("High")),
        "low": _safe_float(row.get("Low")),
        "close": _safe_float(row.get("Close")),
        "volume": int(row.get("Volume")) if row.get("Volume") is not None and row.get("Volume") == row.get("Volume") else None
    }

def fetch_finance_for_date(date_obj: datetime, tickers: List[str]) -> List[Dict]:
    import yfinance as yf
    results = []
    date_str = date_obj.strftime("%Y-%m-%d")
    start = date_str
    end = (date_obj + timedelta(days=1)).strftime("%Y-%m-%d")
    df_bulk = None
    try:
        df_bulk = yf.download(tickers, start=start, end=end, group_by='ticker', progress=False, threads=False)
    except Exception as e:
        print("Bulk download failed:", e)
        df_bulk = None

    for sym in tickers:
        row_vals = {"symbol": sym, "date": date_str, "open": None, "high": None, "low": None, "close": None, "volume": None}
        if df_bulk is not None and not df_bulk.empty:
            try:
                if isinstance(df_bulk.columns, pd.MultiIndex):
                    if sym in list(df_bulk.columns.get_level_values(0)):
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

        # fallback: single ticker
        if row_vals["close"] is None:
            try:
                df_single = yf.download(sym, start=start, end=end, progress=False, threads=False)
                if df_single is not None and not df_single.empty:
                    series_row = df_single.dropna(how='all').iloc[0]
                    row_vals.update(_parse_yf_row(dict(series_row)))
            except Exception:
                pass

        # final fallback: last available within 7 days
        if row_vals["close"] is None:
            try:
                df_hist = yf.download(sym, end=end, period="7d", progress=False, threads=False)
                if df_hist is not None and not df_hist.empty:
                    series_row = df_hist.dropna(how='all').iloc[-1]
                    row_vals.update(_parse_yf_row(dict(series_row)))
            except Exception:
                pass

        results.append(row_vals)
        time.sleep(0.03)
    print(f"Finished finance fetch for {date_str}, {len(results)} tickers")
    return results

# -----------------------
# Sentiment: load pipeline + inference
# Note: loading transformer model is heavy. Consider using external inference service.
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
            print("Inference batch failed:", e)
            for _ in batch:
                results.append(("NEUTRAL", 0.0, 0))
    return results

def aggregate_daily_sentiment(records: List[Dict], load_model: bool = True) -> Dict:
    if not records:
        return {
            "date": None, "n_articles": 0, "n_positive":0, "n_neutral":0, "n_negative":0,
            "prop_positive": float("nan"), "prop_neutral": float("nan"), "prop_negative": float("nan"),
            "mean_sentiment_score": float("nan"), "mean_sentiment_prob": float("nan"),
            "median_sentiment_prob": float("nan"), "std_sentiment_score": float("nan"),
            "weighted_sentiment_score": float("nan"), "avg_text_len": float("nan"), "median_text_len": float("nan")
        }
    df = pd.json_normalize(records)
    # prepare text
    def prepare_text_row(r):
        title = r.get("title") or r.get("headline") or ""
        desc = r.get("description") or r.get("summary") or ""
        return f"{title}. {desc}".strip()
    df["text_for_sentiment"] = df.apply(lambda row: prepare_text_row(row.to_dict()), axis=1)
    texts = df["text_for_sentiment"].astype(str).tolist()

    if load_model:
        nlp = load_sentiment_pipeline()
    else:
        nlp = None

    if nlp is not None and len(texts) > 0:
        preds = infer_sentiment_for_texts(nlp, texts, batch_size=16)
        labels, probs, nums = zip(*preds)
        df["sentiment_label"] = labels
        df["sentiment_score_prob"] = probs
        df["sentiment_score"] = nums
    else:
        df["sentiment_label"] = "NEUTRAL"
        df["sentiment_score_prob"] = 0.0
        df["sentiment_score"] = 0

    n = len(df)
    labels_u = df["sentiment_label"].astype(str).str.upper().fillna("NEUTRAL")
    n_pos = int((labels_u == "POSITIVE").sum())
    n_neu = int((labels_u == "NEUTRAL").sum())
    n_neg = int((labels_u == "NEGATIVE").sum())

    prop_pos = n_pos / n if n>0 else float("nan")
    prop_neu = n_neu / n if n>0 else float("nan")
    prop_neg = n_neg / n if n>0 else float("nan")

    scores = pd.to_numeric(df["sentiment_score"], errors="coerce").fillna(0.0).astype(float)
    probs_num = pd.to_numeric(df["sentiment_score_prob"], errors="coerce").fillna(0.0).astype(float)

    mean_score = float(scores.mean()) if n>0 else float("nan")
    mean_prob = float(probs_num.mean()) if n>0 else float("nan")
    median_prob = float(probs_num.median()) if n>0 else float("nan")
    std_score = float(scores.std(ddof=0)) if n>0 else float("nan")
    denom = probs_num.sum()
    weighted_score = float((scores * probs_num).sum() / denom) if denom>0 else mean_score

    text_lens = df["text_for_sentiment"].astype(str).map(len)
    avg_text_len = float(text_lens.mean()) if n>0 else float("nan")
    median_text_len = float(text_lens.median()) if n>0 else float("nan")

    return {
        "date": df["published_date"].iloc[0] if "published_date" in df.columns else None,
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
# Build final row (59 features) with lag1 filled from prev_fin_rows
# -----------------------
def build_final_row(fin_rows_today: List[Dict], fin_rows_prev: List[Dict], sent_agg: Dict) -> Dict:
    # initialize
    row = {"date": sent_agg.get("date") or (fin_rows_today[0]["date"] if fin_rows_today else None)}
    # set default finance fields to None
    for k in SYMBOL_MAP.values():
        for f in BASE_FIELDS:
            row[f"{k}_{f}"] = None

    # map today's finance
    today_map = {r["symbol"]: r for r in fin_rows_today}
    prev_map = {r["symbol"]: r for r in fin_rows_prev} if fin_rows_prev else {}

    for sym, mapped in SYMBOL_MAP.items():
        t = today_map.get(sym)
        if t:
            for f in BASE_FIELDS:
                row[f"{mapped}_{f}"] = t.get(f)

    # derived features for today
    def safe_pct_change(a,b):
        try:
            if a is None or b is None or a==0:
                return None
            return (b-a)/a
        except:
            return None

    row['sp500_return'] = safe_pct_change(row.get('sp500_open'), row.get('sp500_close'))
    row['sp500_range'] = None if (row.get('sp500_high') is None or row.get('sp500_low') is None) else (row['sp500_high'] - row['sp500_low'])
    row['gold_return'] = safe_pct_change(row.get('gold_open'), row.get('gold_close'))
    row['gold_range'] = None if (row.get('gold_high') is None or row.get('gold_low') is None) else (row['gold_high'] - row['gold_low'])
    row['oil_return'] = safe_pct_change(row.get('oil_open'), row.get('oil_close'))
    row['oil_range'] = None if (row.get('oil_high') is None or row.get('oil_low') is None) else (row['oil_high'] - row['oil_low'])

    # copy closes (already exist as e.g. vix_close); ensure keys exist
    row['vix_close'] = row.get('vix_close')
    row['usd_index_close'] = row.get('usd_index_close')
    row['uup_close'] = row.get('uup_close')

    # compute lag1 from prev_map: if prev exists compute prev_return and assign
    def compute_return_from_map(m, mapped_name):
        rec = m.get(mapped_name)
        if not rec:
            return None
        a = rec.get("Open")
        b = rec.get("Close")
        if a is None or b is None:
            # try other keys if different naming
            a = rec.get("open") or rec.get("Open")
            b = rec.get("close") or rec.get("Close")
        try:
            if a is None or b is None or a == 0:
                return None
            return (b - a) / a
        except:
            return None

    # prev returns for specific symbols
    prev_sp500_return = compute_return_from_map(prev_map, '^GSPC') or compute_return_from_map(prev_map, 'SPY')  # try both
    prev_gold_return = compute_return_from_map(prev_map, 'GC=F')
    prev_vix_close = None
    prev_vix_rec = prev_map.get('^VIX')
    if prev_vix_rec:
        prev_vix_close = prev_vix_rec.get('Close') or prev_vix_rec.get('close') or prev_vix_rec.get('Close')
        # ensure float
        try:
            prev_vix_close = float(prev_vix_close) if prev_vix_close is not None else None
        except:
            prev_vix_close = None

    row['sp500_return_lag1'] = prev_sp500_return
    row['vix_close_lag1'] = prev_vix_close
    row['gold_return_lag1'] = prev_gold_return

    # attach sentiment aggregates
    row['n_articles'] = sent_agg.get('n_articles', 0)
    row['n_positive'] = sent_agg.get('n_positive', 0)
    row['n_neutral'] = sent_agg.get('n_neutral', 0)
    row['n_negative'] = sent_agg.get('n_negative', 0)
    row['prop_positive'] = sent_agg.get('prop_positive')
    row['prop_neutral'] = sent_agg.get('prop_neutral')
    row['prop_negative'] = sent_agg.get('prop_negative')
    row['mean_sentiment_score'] = sent_agg.get('mean_sentiment_score')
    row['mean_sentiment_prob'] = sent_agg.get('mean_sentiment_prob')
    row['median_sentiment_prob'] = sent_agg.get('median_sentiment_prob')
    row['std_sentiment_score'] = sent_agg.get('std_sentiment_score')
    row['weighted_sentiment_score'] = sent_agg.get('weighted_sentiment_score')
    row['avg_text_len'] = sent_agg.get('avg_text_len')
    row['median_text_len'] = sent_agg.get('median_text_len')

    # final column list (59 features) - order kept similar to your list but WITHOUT market_direction
    final_cols = [
        "date",
        "sp500_open","sp500_high","sp500_low","sp500_close","sp500_volume",
        "spy_open","spy_high","spy_low","spy_close","spy_volume",
        "vix_open","vix_high","vix_low","vix_close","vix_volume",
        "gold_open","gold_high","gold_low","gold_close","gold_volume",
        "oil_open","oil_high","oil_low","oil_close","oil_volume",
        "usd_index_open","usd_index_high","usd_index_low","usd_index_close","usd_index_volume",
        "uup_open","uup_high","uup_low","uup_close","uup_volume",
        "sp500_return","sp500_range","gold_return","gold_range","oil_return","oil_range",
        "sp500_return_lag1","vix_close_lag1","gold_return_lag1",
        # note: market_direction intentionally omitted
        "n_articles","n_positive","n_neutral","n_negative",
        "prop_positive","prop_neutral","prop_negative",
        "mean_sentiment_score","mean_sentiment_prob","median_sentiment_prob","std_sentiment_score","weighted_sentiment_score",
        "avg_text_len","median_text_len"
    ]
    # ensure keys exist
    for c in final_cols:
        if c not in row:
            row[c] = None

    # reorder
    out = {c: row.get(c) for c in final_cols}
    return out

# -----------------------
# Main run function
# -----------------------
def run_daily_etl(target_date: Optional[str] = None, upload_prefix: str = "finance"):
    """
    target_date: 'YYYY-MM-DD' or None -> defaults to today's date (UTC)
    The ETL will fetch finance & news for: target_date and prev_date = target_date - 1 day
    and fill lag1 features using prev_date values.
    """
    if target_date is None:
        dt = datetime.utcnow().date()
        date_str = dt.strftime("%Y-%m-%d")
    else:
        # validate format
        date_str = target_date

    # parse dates
    date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
    prev_date_obj = date_obj - timedelta(days=1)
    prev_date_str = prev_date_obj.strftime("%Y-%m-%d")

    print("Running ETL for target_date:", date_str, "prev_date:", prev_date_str)

    # 1) Fetch news for target_date (Polygon)
    news_records = crawl_polygon_single_day(date_str)

    # 2) Aggregate sentiment for target_date
    sent_agg = aggregate_daily_sentiment(news_records, load_model=True)
    if sent_agg.get("date") is None:
        sent_agg["date"] = date_str

    # 3) Fetch finance for target_date and prev_date
    fin_rows_today = fetch_finance_for_date(datetime.strptime(date_str, "%Y-%m-%d"), TICKERS)
    fin_rows_prev = fetch_finance_for_date(datetime.strptime(prev_date_str, "%Y-%m-%d"), TICKERS)

    # 4) Build final row (59 cols) using prev for lag1
    final_row = build_final_row(fin_rows_today, fin_rows_prev, sent_agg)
    df_final = pd.DataFrame([final_row])

    # 5) Save to parquet and upload
    tmp = BytesIO()
    df_final.to_parquet(tmp, index=False)
    tmp.seek(0)
    dest_blob = f"{upload_prefix}/date={date_str}/daily_features_{date_str}.parquet"
    ok = upload_bytes_to_blob(tmp.read(), dest_blob)
    if not ok:
        raise RuntimeError("Upload failed")

    print("ETL completed for", date_str)
    return df_final

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--date", help="YYYY-MM-DD (default=today UTC)", default=None)
    p.add_argument("--prefix", help="upload prefix", default="finance")
    args = p.parse_args()
    df_final = run_daily_etl(args.date, upload_prefix=args.prefix)

    df_final.to_parquet("df_final.parquet", engine="pyarrow", index=False)

