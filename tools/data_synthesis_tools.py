import math
import os
import pathlib

from agents import function_tool
from utils.audit_logger import log_event

# All uploaded PDFs are stored under this directory
UPLOAD_DIR = pathlib.Path("uploads")


@function_tool
def extract_pdf_text(filename: str) -> dict:
    """Extract and return full text content from an uploaded PDF file.
    Pass only the filename (e.g. 'report.pdf'), not a full path.
    """
    log_event("DataAgent", "extract_pdf_text", {"filename": filename})
    try:
        import pypdf
    except ImportError:
        return {"error": "pypdf package not installed. Run: pip install pypdf"}

    # Security: strip any path components to prevent directory traversal
    safe_name = pathlib.Path(filename).name
    file_path = UPLOAD_DIR / safe_name

    if not file_path.exists():
        return {"error": f"File '{safe_name}' not found. Please upload it first via the attachment button."}

    try:
        reader = pypdf.PdfReader(str(file_path))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append({"page": i + 1, "text": text.strip()})

        full_text = "\n\n".join(p["text"] for p in pages if p["text"])
        return {
            "filename": safe_name,
            "total_pages": len(reader.pages),
            "full_text": full_text[:12000],  # cap to avoid token overflow
            "pages_preview": pages[:5],
        }
    except Exception as e:
        return {"error": f"Failed to read PDF: {str(e)}"}


@function_tool
def fetch_fred_series(series_id: str, limit: int = 10) -> dict:
    """Fetch macroeconomic data from the Federal Reserve FRED API.

    Common series IDs relevant to banking:
      FEDFUNDS     - Federal Funds Rate (monthly)
      DFF          - Federal Funds Effective Rate (daily)
      MORTGAGE30US - 30-Year Fixed Mortgage Average Rate
      MORTGAGE15US - 15-Year Fixed Mortgage Average Rate
      CPIAUCSL     - Consumer Price Index (Inflation, monthly)
      UNRATE       - Civilian Unemployment Rate
      GDP          - Gross Domestic Product (quarterly)
      T10Y2Y       - 10-Year minus 2-Year Treasury Yield Spread
      DEXUSEU      - USD/EUR Exchange Rate
    """
    log_event("DataAgent", "fetch_fred_series", {"series_id": series_id, "limit": limit})

    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        return {
            "error": (
                "FRED_API_KEY is not set. Add it to your .env file. "
                "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
            )
        }

    try:
        from fredapi import Fred
    except ImportError:
        return {"error": "fredapi package not installed. Run: pip install fredapi"}

    try:
        fred = Fred(api_key=api_key)
        info = fred.get_series_info(series_id)
        data = fred.get_series(series_id)
        observations = data.tail(limit)

        return {
            "series_id": series_id,
            "title": info.get("title", series_id),
            "units": info.get("units", ""),
            "frequency": info.get("frequency", ""),
            "last_updated": info.get("last_updated", ""),
            "observations": [
                {
                    "date": str(date.date()),
                    "value": round(float(val), 4) if not math.isnan(float(val)) else None,
                }
                for date, val in observations.items()
            ],
        }
    except Exception as e:
        return {"error": f"FRED API error for series '{series_id}': {str(e)}"}


@function_tool
def fetch_stock_data(ticker: str, period: str = "1mo") -> dict:
    """Fetch stock or ETF market data from Yahoo Finance.

    ticker: Stock symbol, e.g. 'JPM', 'BAC', 'WFC', 'GS', 'AAPL', 'SPY'
    period: Data range — '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y'

    Returns current price, key fundamentals, and recent OHLCV history.
    No API key required.
    """
    log_event("DataAgent", "fetch_stock_data", {"ticker": ticker, "period": period})

    try:
        import yfinance as yf
    except ImportError:
        return {"error": "yfinance package not installed. Run: pip install yfinance"}

    valid_periods = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"}
    if period not in valid_periods:
        return {"error": f"Invalid period '{period}'. Choose from: {', '.join(sorted(valid_periods))}"}

    try:
        stock = yf.Ticker(ticker.upper())
        hist = stock.history(period=period)

        if hist.empty:
            return {"error": f"No data found for ticker '{ticker.upper()}'. Verify the symbol is correct."}

        info = stock.info
        recent = hist.tail(5)

        return {
            "ticker": ticker.upper(),
            "company_name": info.get("longName", ticker.upper()),
            "sector": info.get("sector", "N/A"),
            "currency": info.get("currency", "USD"),
            "current_price": round(float(hist["Close"].iloc[-1]), 2),
            "previous_close": info.get("previousClose"),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "dividend_yield": info.get("dividendYield"),
            "recent_prices": [
                {
                    "date": str(idx.date()),
                    "open": round(float(row["Open"]), 2),
                    "high": round(float(row["High"]), 2),
                    "low": round(float(row["Low"]), 2),
                    "close": round(float(row["Close"]), 2),
                    "volume": int(row["Volume"]),
                }
                for idx, row in recent.iterrows()
            ],
        }
    except Exception as e:
        return {"error": f"Yahoo Finance error for '{ticker}': {str(e)}"}
