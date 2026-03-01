#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone

UPBIT_BASE = "https://api.upbit.com/v1"
OKX_BASE = "https://www.okx.com/api/v5"
USER_AGENT = "Mozilla/5.0"
CACHE_DIR = os.path.expanduser("~/.cache/okx_from_upbit")
RATE_LIMIT_RETRY_SEC = 15
DEFAULT_MARKET_PREFIX = "KRW"
CANDLE_LIMIT = 100


def send_notification(title: str, body: str = "") -> None:
    notifier = shutil.which("notify-send")
    if notifier is None:
        return
    args = [notifier, title]
    if body:
        args.append(body)
    try:
        subprocess.run(args, check=False)
    except Exception:
        pass


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def cache_path(url: str) -> str:
    import hashlib

    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{digest}.json")


def read_cache(url: str):
    path = cache_path(url)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def write_cache(url: str, payload: object) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = cache_path(url)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    except Exception:
        pass


def clear_cache() -> None:
    if not os.path.isdir(CACHE_DIR):
        return
    for name in os.listdir(CACHE_DIR):
        p = os.path.join(CACHE_DIR, name)
        if os.path.isfile(p):
            try:
                os.remove(p)
            except Exception:
                pass


def to_float(value):
    try:
        return float(value)
    except Exception:
        return None


def fetch_json(base_url: str, path: str, params: dict[str, str] | None = None, retries: int = 4):
    query = ""
    if params:
        query = "?" + urllib.parse.urlencode(params)
    url = f"{base_url}{path}{query}"
    cached = read_cache(url)
    if cached is not None:
        return cached

    last_error: Exception | None = None
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }

    for i in range(retries + 1):
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                raw = resp.read().decode("utf-8")
                data = json.loads(raw)
                write_cache(url, data)
                return data
        except urllib.error.HTTPError as e:
            if e.code == 429 and i < retries:
                clear_cache()
                print(f"Rate limit hit on {url}, wait {RATE_LIMIT_RETRY_SEC}s")
                time.sleep(RATE_LIMIT_RETRY_SEC)
                last_error = e
                continue
            if e.code in {500, 502, 503, 504} and i < retries:
                wait = min(2.0 * (2**i), 8.0)
                print(f"Server error {e.code}, retry in {wait}s")
                time.sleep(wait)
                last_error = e
                continue
            detail = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else ""
            raise RuntimeError(f"HTTP {e.code} on {url}: {detail}") from e
        except urllib.error.URLError as e:
            if i < retries:
                wait = min(1.0 * (2**i), 8.0)
                print(f"Network error on {url}, retry in {wait}s")
                time.sleep(wait)
                last_error = e
                continue
            raise RuntimeError(f"Network error on {url}: {e}") from e
    raise RuntimeError(f"Request failed after retries: {url}") from last_error


def chunked(values: list[str], size: int) -> list[list[str]]:
    return [values[i:i + size] for i in range(0, len(values), size)]


def load_top_upbit_markets(top_n: int, market_prefix: str) -> list[dict]:
    """Fetch upbit markets and return top N by 24h quote volume."""
    markets = fetch_json(UPBIT_BASE, "/market/all", {"isDetails": "false"})
    if not isinstance(markets, list):
        raise RuntimeError("Invalid response from upbit /market/all")

    codes = [m["market"] for m in markets if m.get("market", "")]
    if market_prefix:
        p = market_prefix.upper().strip()
        codes = [c for c in codes if c == p or c.startswith(f"{p}-")]

    rows: list[dict] = []
    for chunk in chunked(codes, 100):
        joined = ",".join(chunk)
        tickers = fetch_json(UPBIT_BASE, "/ticker", {"markets": joined})
        if not isinstance(tickers, list):
            continue
        for t in tickers:
            code = t.get("market")
            if not code:
                continue
            rows.append(
                {
                    "market": code,
                    "trade_price": to_float(t.get("trade_price")),
                    "trade_volume": to_float(t.get("acc_trade_price_24h")),
                }
            )

    rows.sort(key=lambda x: x.get("trade_volume") or 0.0, reverse=True)
    return rows[:top_n]


def build_okx_instrument_set() -> dict[str, dict]:
    data = fetch_json(OKX_BASE, "/public/instruments", {"instType": "SPOT"})
    if data.get("code") != "0" or "data" not in data:
        raise RuntimeError(f"Invalid response from OKX instruments: {data}")

    mapping: dict[str, dict] = {}
    for row in data["data"]:
        inst = row.get("instId")
        if not inst:
            continue
        mapping[inst] = row
    return mapping


def map_upbit_to_okx(upbit_market: str, okx_insts: dict[str, dict], fallback_quotes: list[str]) -> str | None:
    if "-" not in upbit_market:
        return None
    base = upbit_market.split("-", 1)[1]
    if not base or base == "KRW":
        return None

    for q in fallback_quotes:
        inst = f"{base}-{q}"
        info = okx_insts.get(inst)
        if info is None:
            continue
        if info.get("state", "") != "live":
            continue
        return inst
    return None


def utc_to_local_utc_iso(ts_ms: int) -> str:
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def to_kst_iso(ts_ms: int) -> str:
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc) + timedelta(hours=9)
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "KST")

def iter_okx_candles(inst_id: str, bar: str = "1D", limit: int = CANDLE_LIMIT, max_rows: int | None = None):
    fetched: list[list[str]] = []
    seen: set[str] = set()
    after: str | None = None

    while True:
        params = {
            "instId": inst_id,
            "bar": bar,
            "limit": str(limit),
        }
        if after:
            params["after"] = after

        payload = fetch_json(OKX_BASE, "/market/candles", params)
        if payload.get("code") != "0":
            raise RuntimeError(f"OKX candle error for {inst_id}: {payload}")
        page = payload.get("data") or []
        if not page:
            break

        # payload returns newest -> oldest
        added = 0
        for row in page:
            # row: [ts, open, high, low, close, vol, volCcy, volCcyQuote, confirm]
            if not isinstance(row, (list, tuple)) or len(row) < 8:
                continue
            ts = row[0]
            if ts in seen:
                continue
            seen.add(ts)
            fetched.append(row)
            added += 1

        if max_rows is not None and len(fetched) >= max_rows:
            break

        if len(page) < limit:
            break

        # move cursor older than current oldest ts
        oldest_ts = page[-1][0]
        try:
            after = str(int(oldest_ts) - 1)
        except Exception:
            break

        if added == 0:
            break

    # normalize to ascending (oldest -> newest)
    fetched.sort(key=lambda row: int(row[0]))
    return fetched


def write_okx_day_csv(path: str, upbit_market: str, okx_inst: str, rows: list[list[str]]) -> int:
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(["market", "candle_date_time_utc", "candle_date_time_kst", "timestamp", "trade_price", "opening_price", "high_price", "low_price", "candle_acc_trade_volume", "candle_acc_trade_price"])
        return 0

    with open(path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "market",
                "okx_inst",
                "candle_date_time_utc",
                "candle_date_time_kst",
                "timestamp",
                "opening_price",
                "high_price",
                "low_price",
                "trade_price",
                "candle_acc_trade_volume",
                "candle_acc_trade_price",
            ],
        )
        writer.writeheader()
        for row in rows:
            ts = int(row[0])
            writer.writerow(
                {
                    "market": upbit_market,
                    "okx_inst": okx_inst,
                    "candle_date_time_utc": utc_to_local_utc_iso(ts),
                    "candle_date_time_kst": to_kst_iso(ts),
                    "timestamp": row[0],
                    "opening_price": to_float(row[1]),
                    "high_price": to_float(row[2]),
                    "low_price": to_float(row[3]),
                    "trade_price": to_float(row[4]),
                    "candle_acc_trade_volume": to_float(row[5]),
                    "candle_acc_trade_price": to_float(row[6]),
                }
            )
    return len(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch OKX day candles for Upbit top-N symbols.")
    p.add_argument("--top", type=int, default=50, help="Top-N Upbit symbols by 24h volume")
    p.add_argument("--market", type=str, default=DEFAULT_MARKET_PREFIX, help="Upbit market prefix (e.g. KRW)")
    p.add_argument(
        "--candidate-quotes",
        type=str,
        default="USDT,USDC,BTC,ETH",
        help="Fallback quote list for OKX symbol mapping in order",
    )
    p.add_argument("--out-dir", type=str, default="/tmp/okx_from_upbit_top", help="Output directory")
    p.add_argument("--candle-limit", type=int, default=CANDLE_LIMIT, help="Candles per OKX request")
    p.add_argument("--notify", action="store_true", help="Progress notification via notify-send")
    p.add_argument("--notify-every", type=int, default=10, help="Notify every N processed symbols (0 disables)")
    p.add_argument("--summary", type=str, default="", help="Optional summary JSON path")
    p.add_argument("--all-history", action="store_true", help="Fetch all available history (default on) until exhausted")
    p.add_argument("--max-rows", type=int, default=0, help="Optional cap per market")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.top < 1:
        raise SystemExit("--top must be >= 1")
    if args.candle_limit < 1 or args.candle_limit > 300:
        raise SystemExit("--candle-limit must be between 1 and 300")

    ensure_dir(args.out_dir)
    fallback_quotes = [q.strip().upper() for q in args.candidate_quotes.split(",") if q.strip()]

    top_markets = load_top_upbit_markets(args.top, args.market)
    if not top_markets:
        raise SystemExit("No markets found from Upbit")

    okx_insts = build_okx_instrument_set()

    mapped = []
    skipped = []
    for row in top_markets:
        m = row["market"]
        inst = map_upbit_to_okx(m, okx_insts, fallback_quotes)
        if inst is None:
            skipped.append({"market": m, "reason": "no_okx_mapping"})
            continue
        mapped.append((m, inst))

    summary: dict = {
        "upbit_prefix": args.market.upper(),
        "requested_top": args.top,
        "mapped_count": len(mapped),
        "skipped_count": len(skipped),
        "mapped": [],
        "skipped": skipped,
        "markets": [],
    }

    start = time.time()
    saved_files: list[str] = []
    for idx, (market, inst) in enumerate(mapped, start=1):
        rows = iter_okx_candles(inst, bar="1D", limit=args.candle_limit, max_rows=args.max_rows if args.max_rows > 0 else None)
        out_path = os.path.join(args.out_dir, f"{market}_day.csv")
        count = write_okx_day_csv(out_path, market, inst, rows)
        saved_files.append(out_path)

        first_ts = None
        last_ts = None
        if rows:
            first_ts = rows[0][0]
            last_ts = rows[-1][0]

        summary["markets"].append(
            {
                "upbit_market": market,
                "okx_inst": inst,
                "file": out_path,
                "rows": count,
                "start_utc": utc_to_local_utc_iso(int(first_ts)) if first_ts else None,
                "end_utc": utc_to_local_utc_iso(int(last_ts)) if last_ts else None,
            }
        )

        print(f"[{idx:02d}/{len(mapped)}] {market} -> {inst}, rows={count}, file={out_path}")
        if args.notify and (args.notify_every == 0 or idx % max(args.notify_every, 1) == 0):
            elapsed = timedelta(seconds=int(time.time() - start))
            send_notification("OKX 일봉 수집 진행", f"{idx}/{len(mapped)} 완료 (매핑{idx}/{len(top_markets)}): {market} -> {inst}\n경과: {elapsed}")

    # summarize periods
    def _date_from_ms(ms: str) -> datetime:
        return datetime.fromtimestamp(int(ms) / 1000)

    dates = []
    for m in summary["markets"]:
        p = m["file"]
        if m["rows"] <= 0:
            continue
        with open(p, encoding="utf-8") as fp:
            rdr = csv.DictReader(fp)
            local_dates = [datetime.fromisoformat(r["candle_date_time_utc"].replace("Z", "+00:00")).date() for r in rdr if r.get("candle_date_time_utc")]
        if local_dates:
            dates.append((min(local_dates), max(local_dates), m["rows"]))

    if dates:
        union_min = min(x[0] for x in dates)
        union_max = max(x[1] for x in dates)
        inter_start = max(x[0] for x in dates)
        inter_end = min(x[1] for x in dates)
        summary["period"] = {
            "mapped_count": len(dates),
            "union_start": union_min.isoformat(),
            "union_end": union_max.isoformat(),
            "intersection_start": inter_start.isoformat(),
            "intersection_end": inter_end.isoformat(),
        }
    else:
        summary["period"] = None

    if args.summary:
        with open(args.summary, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.notify:
        elapsed = timedelta(seconds=int(time.time() - start))
        send_notification("OKX 일봉 수집 완료", f"맵핑 완료: {len(mapped)} / 요청: {args.top}\n소요시간: {elapsed}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
