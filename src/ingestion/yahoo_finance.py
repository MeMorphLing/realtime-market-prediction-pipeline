"""Yahoo Finance OHLCV ingestion.

Downloads historical price data with `yfinance` and stores it as Parquet
files inside ``data/raw/prices/``.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

RAW_PRICES_DIR = Path("data/raw/prices")


def fetch_price_data(
    ticker: str,
    period: str = "7d",
    interval: str = "1h",
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Fetch OHLCV bars for ``ticker`` and persist them as Parquet.

    Args:
        ticker: Yahoo Finance symbol (e.g. ``"AAPL"``).
        period: yfinance period string (``"1d"``, ``"7d"``, ``"1mo"``, ...).
        interval: Bar resolution (``"1m"``, ``"5m"``, ``"1h"``, ``"1d"``).
        output_dir: Optional override for the output directory.

    Returns:
        A DataFrame with columns ``[timestamp, open, high, low, close, volume, ticker]``.
    """
    out_dir = Path(output_dir) if output_dir else RAW_PRICES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching %s prices (period=%s, interval=%s)", ticker, period, interval)

    try:
        df = yf.download(
            tickers=ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
        )
    except Exception:
        logger.exception("yfinance download failed for %s", ticker)
        raise

    if df.empty:
        logger.warning("No price rows returned for %s", ticker)
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume", "ticker"]
        )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    df = df.reset_index().rename(columns={df.index.name or "Datetime": "timestamp"})
    if "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "timestamp"})
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "timestamp"})

    df["ticker"] = ticker
    df = df[["timestamp", "open", "high", "low", "close", "volume", "ticker"]]

    today = datetime.utcnow().strftime("%Y%m%d")
    out_path = out_dir / f"{ticker}_{today}.parquet"

    try:
        df.to_parquet(out_path, index=False)
        logger.info("Saved %d rows to %s", len(df), out_path)
    except Exception:
        logger.exception("Failed to write parquet to %s", out_path)
        raise

    return df


if __name__ == "__main__":
    fetch_price_data("AAPL")
