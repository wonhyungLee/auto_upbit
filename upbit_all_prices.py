#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Iterable


BASE_URL = "https://api.upbit.com/v1"
CHUNK_SIZE = 100
CANDLE_COUNT_MAX = 200
CACHE_DIR = os.path.expanduser("~/.cache/upbit_all_prices")
RATE_LIMIT_RETRY_DELAY_SEC = 15

HISTORICAL_TIME_UNITS = {
    "day": "/candles/days",
}


def load_dotenv_like(path: str) -> dict[str, str]:
    env: dict[str, str] = {}
    pattern = re.compile(r"^([A-Z][A-Z0-9_]*)\s*=\s*\"(.*)\"$")
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("- "):
                continue
            m = pattern.match(line)
            if m:
                env[m.group(1)] = m.group(2)
    return env


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def cache_path_for_url(url: str) -> str:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{digest}.json")


def read_cache(url: str) -> dict | list | None:
    path = cache_path_for_url(url)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def write_cache(url: str, payload: object) -> None:
    ensure_dir(CACHE_DIR)
    path = cache_path_for_url(url)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    except Exception:
        pass


def clear_cache() -> None:
    if not os.path.isdir(CACHE_DIR):
        return
    try:
        for entry in os.listdir(CACHE_DIR):
            full = os.path.join(CACHE_DIR, entry)
            if os.path.isfile(full):
                try:
                    os.remove(full)
                except Exception:
                    pass
    except Exception:
        pass


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


def fetch_json(path: str, params: dict[str, str] | None = None, retries: int = 4) -> list[dict] | dict:
    query = ""
    if params:
        query = "?" + urllib.parse.urlencode(params)
    url = f"{BASE_URL}{path}{query}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    cached = read_cache(url)
    if cached is not None:
        return cached
    last_error: Exception | None = None
    delay = 1.0
    for _ in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                raw = resp.read().decode("utf-8")
                data = json.loads(raw)
                write_cache(url, data)
                return data
        except urllib.error.HTTPError as e:
            if e.code == 429 and _ < retries:
                clear_cache()
                print(f"Rate limit hit on {url}. Wait {RATE_LIMIT_RETRY_DELAY_SEC}s and retry.")
                time.sleep(RATE_LIMIT_RETRY_DELAY_SEC)
                delay = min(delay * 2, 8.0)
                last_error = e
                continue
            if e.code in {500, 502, 503, 504} and _ < retries:
                time.sleep(delay)
                delay = min(delay * 2, 8.0)
                last_error = e
                continue
            detail = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Upbit API error {e.code} on {url}: {detail}") from e
        except urllib.error.URLError as e:
            if _ < retries:
                time.sleep(delay)
                delay = min(delay * 2, 8.0)
                last_error = e
                continue
            raise RuntimeError(f"Network error on {url}: {e.reason}") from e
    raise RuntimeError(f"Upbit API request failed after retries: {url}") from last_error


def chunked(values: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(values), size):
        yield values[i : i + size]


def fetch_markets(with_details: bool = True) -> list[dict]:
    data = fetch_json(
        "/market/all",
        {"isDetails": "true" if with_details else "false"},
    )
    if not isinstance(data, list):
        raise RuntimeError("Invalid response from /market/all")
    return data


def fetch_tickers(markets: list[str]) -> list[dict]:
    if not markets:
        return []
    joined = ",".join(markets)
    data = fetch_json("/ticker", {"markets": joined})
    if not isinstance(data, list):
        raise RuntimeError("Invalid response from /ticker")
    return data


def to_float(value):
    try:
        return float(value)
    except Exception:
        return None


def combine_market_and_ticker(markets: list[dict], tickers: list[dict]) -> list[dict]:
    ticker_map = {item.get("market", ""): item for item in tickers}
    merged: list[dict] = []
    for market in markets:
        code = market.get("market", "")
        t = ticker_map.get(code, {})
        merged.append(
            {
                "market": code,
                "korean_name": market.get("korean_name", ""),
                "english_name": market.get("english_name", ""),
                "market_warning": market.get("market_warning", ""),
                "trade_price": to_float(t.get("trade_price")),
                "trade_volume": to_float(t.get("acc_trade_price_24h")),
                "change": t.get("signed_change_rate"),
                "timestamp": t.get("timestamp"),
            }
        )
    return merged


def print_table(rows: list[dict], top: int | None = None) -> None:
    if not rows:
        print("No data.")
        return

    selected = rows[:top] if top else rows
    print(
        f"{'MARKET':<12} {'KRW-Name/KR':<16} {'EN':<20} {'PRICE':>16} {'24h_VOL(KRW)':>18} {'CHANGE_RATE':>12}"
    )
    print("-" * 100)
    for r in selected:
        price = r["trade_price"]
        volume = r["trade_volume"]
        change = r["change"]
        print(
            f"{r['market']:<12} "
            f"{(r['korean_name'] or '-'): <16} "
            f"{(r['english_name'] or '-'): <20} "
            f"{(f'{price:,.2f}' if isinstance(price, (int, float)) else ' - '):>16} "
            f"{(f'{volume:,.2f}' if isinstance(volume, (int, float)) else ' - '):>18} "
            f"{(f'{float(change)*100:>10.2f}%' if change is not None else ' - '):>12}"
        )


def write_csv(path: str, rows: list[dict]) -> None:
    fieldnames = [
        "market",
        "korean_name",
        "english_name",
        "market_warning",
        "trade_price",
        "acc_trade_price_24h",
        "signed_change_rate",
        "signed_change_price",
        "trade_timestamp",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            # Merge back with original ticker fields not shown in table output.
            payload = {
                "market": row["market"],
                "korean_name": row["korean_name"],
                "english_name": row["english_name"],
                "market_warning": row["market_warning"],
                "trade_price": row["trade_price"],
                "acc_trade_price_24h": row["trade_volume"],
                "signed_change_rate": row["change"],
                "signed_change_price": None,
                "trade_timestamp": row["timestamp"],
            }
            writer.writerow(payload)


def parse_timestamp_ms(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    raw = value.strip()
    if raw.isdigit():
        return int(raw)
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def to_utc_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def fetch_candles(
    market: str,
    unit: str,
    count: int = CANDLE_COUNT_MAX,
    to: str | int | None = None,
) -> list[dict]:
    if unit not in HISTORICAL_TIME_UNITS:
        raise ValueError(f"Unsupported candle unit: {unit}")
    path = HISTORICAL_TIME_UNITS[unit]
    params = {"market": market, "count": str(count)}
    if to is not None:
        params["to"] = str(to)
    data = fetch_json(path, params)
    if not isinstance(data, list):
        raise RuntimeError(f"Invalid response from {path}")
    return data


def fetch_all_candles(
    market: str,
    unit: str,
    count: int,
    to: str | None = None,
    since: str | None = None,
    max_pages: int = 0,
) -> list[dict]:
    candles: list[dict] = []
    for page in iter_candles(market=market, unit=unit, count=count, to=to, since=since, max_pages=max_pages):
        candles.extend(page)
    return candles


def iter_candles(
    market: str,
    unit: str,
    count: int,
    to: str | None = None,
    since: str | None = None,
    max_pages: int = 0,
) -> Iterable[list[dict]]:
    since_ms = parse_timestamp_ms(since)
    cursor = to
    pages = 0
    seen_ts: set[int] = set()
    while True:
        if max_pages > 0 and pages >= max_pages:
            break
        page = fetch_candles(market=market, unit=unit, count=count, to=cursor)
        if not page:
            break
        pages += 1

        current: list[dict] = []
        for row in page:
            ts = row.get("timestamp")
            try:
                ts_int = int(ts)
            except Exception:
                continue
            if ts_int in seen_ts:
                continue
            if since_ms is not None and ts_int < since_ms:
                continue
            seen_ts.add(ts_int)
            current.append(row)

        if current:
            yield current

        oldest = page[-1].get("timestamp")
        try:
            oldest_ms = int(oldest)
        except Exception:
            break
        cursor = to_utc_iso(oldest_ms - 1)

        if len(page) < count:
            break
        if since_ms is not None and oldest_ms < since_ms:
            break

def print_candle_table(rows: list[dict], top: int | None = None) -> None:
    if not rows:
        print("No candle data.")
        return

    selected = rows[:top] if top else rows
    print(f"{'DateTime':<26} {'Market':<12} {'Open':>12} {'Close':>12} {'High':>12} {'Low':>12} {'Volume':>14}")
    print("-" * 100)
    for r in selected:
        dt = r.get("candle_date_time_kst") or r.get("candle_date_time_utc") or ""
        open_v = to_float(r.get("opening_price"))
        close_v = to_float(r.get("trade_price"))
        high_v = to_float(r.get("high_price"))
        low_v = to_float(r.get("low_price"))
        vol = to_float(r.get("candle_acc_trade_volume"))
        print(
            f"{dt:<26} "
            f"{r.get('market', ''):<12} "
            f"{(f'{open_v:,.2f}' if isinstance(open_v, (int, float)) else ' - '):>12} "
            f"{(f'{close_v:,.2f}' if isinstance(close_v, (int, float)) else ' - '):>12} "
            f"{(f'{high_v:,.2f}' if isinstance(high_v, (int, float)) else ' - '):>12} "
            f"{(f'{low_v:,.2f}' if isinstance(low_v, (int, float)) else ' - '):>12} "
            f"{(f'{vol:,.4f}' if isinstance(vol, (int, float)) else ' - '):>14}"
        )


def write_candles_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = sorted(set().union(*[set(row.keys()) for row in rows]))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def iter_intervals(args_interval: str) -> list[str]:
    if args_interval != "day":
        raise SystemExit("Only day interval is supported. Use --interval day.")
    return ["day"]


def write_candle_pages_csv(path: str, pages: Iterable[list[dict]]) -> int:
    fieldnames: list[str] | None = None
    row_count = 0
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer: csv.DictWriter | None = None
        for page in pages:
            if not page:
                continue
            if fieldnames is None:
                collected: set[str] = set()
                for row in page:
                    collected.update(row.keys())
                fieldnames = sorted(collected)
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            for row in page:
                writer.writerow(row)
                row_count += 1
    if fieldnames is None:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["market", "candle_date_time_kst", "timestamp", "trade_price"])
            writer.writeheader()
    return row_count


def _format_progress_seconds(seconds: float) -> str:
    secs = int(seconds)
    mins, sec = divmod(secs, 60)
    hours, mins = divmod(mins, 60)
    return f"{hours:02d}:{mins:02d}:{sec:02d}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch all Upbit listings and price info.")
    parser.add_argument("--market", type=str, default="", help="Optional prefix filter, e.g. KRW or BTC")
    parser.add_argument("--all-markets", action="store_true", help="Fetch history for all listed markets")
    parser.add_argument("--history", action="store_true", help="Fetch historical candles instead of current ticker")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    parser.add_argument("--csv", type=str, default="", help="Save CSV file")
    parser.add_argument("--top", type=int, default=0, help="Print only top N rows")
    parser.add_argument("--top-volume", type=int, default=0, help="Fetch only top N markets by 24h volume (history mode)")
    parser.add_argument("--creds", type=str, default="개인정보", help="Path to local credential file")
    parser.add_argument("--no-details", action="store_true", help="Skip market details lookup")
    parser.add_argument("--interval", type=str, default="day", choices=sorted(HISTORICAL_TIME_UNITS.keys()), help="History interval (only day)")
    parser.add_argument("--candle-count", type=int, default=CANDLE_COUNT_MAX, help="Candles per request (1-200)")
    parser.add_argument("--all-history", action="store_true", help="Fetch all available historical pages (may take long)")
    parser.add_argument("--to", type=str, default=None, help="History end time (timestamp ms or ISO8601, e.g. 2026-02-28T00:00:00+00:00)")
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="Stop when candles are older than this time (timestamp ms or ISO8601)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Max pages for history mode (0 = all pages)",
    )
    parser.add_argument("--output-dir", type=str, default="upbit_all_history", help="Directory for per-market history output")
    parser.add_argument("--notify", action="store_true", help="Send desktop notifications during all-markets history collection")
    parser.add_argument("--notify-every", type=int, default=25, help="Send progress notification every N markets (0: disable)")
    args = parser.parse_args()

    if args.candle_count < 1 or args.candle_count > CANDLE_COUNT_MAX:
        raise SystemExit("Error: --candle-count must be between 1 and 200")
    if args.max_pages < 0:
        raise SystemExit("Error: --max-pages must be >= 0")
    if args.top_volume < 0:
        raise SystemExit("Error: --top-volume must be >= 0")
    if args.notify_every < 0:
        raise SystemExit("Error: --notify-every must be >= 0")
    if args.all_history:
        args.max_pages = 0

    try:
        env = load_dotenv_like(args.creds)
        if env.get("UPBIT_KEY") and env.get("UPBIT_SECRET"):
            pass
        else:
            print("UPBIT API credential is not fully set. Continuing with public endpoints.")
    except FileNotFoundError:
        print("Credential file not found. Continuing with public endpoints.")

    markets = fetch_markets(with_details=not args.no_details)
    markets = [m for m in markets if m.get("market")]

    if args.market:
        prefix = args.market.upper().strip()
        markets = [m for m in markets if m["market"] == prefix or m["market"].startswith(f"{prefix}-")]

    if args.history:
        intervals = iter_intervals(args.interval)

        # Choose top-N markets by 24h volume if requested.
        if args.top_volume > 0:
            market_codes = [m["market"] for m in markets]
            merged: list[dict] = []
            for chunk in chunked(market_codes, CHUNK_SIZE):
                tickers = fetch_tickers(chunk)
                merged.extend(combine_market_and_ticker(markets=[{"market": c} for c in chunk], tickers=tickers))
            merged.sort(key=lambda row: row.get("trade_volume") or 0.0, reverse=True)
            ranked = []
            picked: set[str] = set()
            market_by_code = {m["market"]: m for m in markets}
            for row in merged:
                code = row.get("market")
                if not code or code in picked:
                    continue
                if code in market_by_code:
                    ranked.append(market_by_code[code])
                    picked.add(code)
                if len(ranked) >= args.top_volume:
                    break
            if len(ranked) < args.top_volume:
                for m in markets:
                    if m["market"] in picked:
                        continue
                    ranked.append(m)
                    if len(ranked) >= args.top_volume:
                        break
            markets = ranked[:args.top_volume]
            print(f"Top volume filter applied: {args.top_volume} markets selected.")

        if args.all_markets:
            if not markets:
                raise SystemExit("No markets matched the current filter.")
            if args.json:
                raise SystemExit("JSON output is not supported for --all-markets. Use --csv or disable --all-markets.")
            ensure_dir(args.output_dir)
            total_markets = len(markets)
            start_ts = time.time()
            saved_total = 0
            if args.notify:
                send_notification("업비트 일봉 수집 시작", f"총 {total_markets}개 코인")
            for idx, market in enumerate(markets, start=1):
                market_code = market["market"]
                for interval in intervals:
                    print(f"Fetching {market_code} / {interval} ...")
                    out_path = os.path.join(args.output_dir, f"{market_code}_{interval}.csv")
                    pages = iter_candles(
                        market=market_code,
                        unit=interval,
                        count=args.candle_count,
                        to=args.to,
                        since=args.since,
                        max_pages=args.max_pages,
                    )
                    total = write_candle_pages_csv(out_path, pages)
                    saved_total += total
                    print(f"  saved {total} rows -> {out_path}")

                if args.notify and (
                    args.notify_every == 0
                    or idx % max(args.notify_every, 1) == 0
                    or idx == total_markets
                ):
                    done_rate = (idx / total_markets) * 100
                    elapsed = _format_progress_seconds(time.time() - start_ts)
                    send_notification(
                        "업비트 일봉 진행",
                        f"진행: {idx}/{total_markets} ({done_rate:.1f}%)\n누적 row: {saved_total}\n경과시간: {elapsed}",
                    )
            if args.notify:
                elapsed = _format_progress_seconds(time.time() - start_ts)
                send_notification(
                    "업비트 일봉 수집 완료",
                    f"총 {total_markets}개 코인\n총 저장 row: {saved_total}\n총 소요시간: {elapsed}",
                )
            return 0

        if len(markets) != 1:
            raise SystemExit("History mode needs exactly one market. Use --market e.g. KRW-BTC or --all-markets.")
        market_code = markets[0]["market"]
        for interval in intervals:
            candles = fetch_all_candles(
                market=market_code,
                unit=interval,
                count=args.candle_count,
                to=args.to,
                since=args.since,
                max_pages=args.max_pages,
            )
            if args.json:
                print(json.dumps(candles, ensure_ascii=False, indent=2))
            else:
                print(f"Fetched {len(candles)} candles for {market_code} ({interval})")
                top = args.top if args.top > 0 else None
                print_candle_table(candles, top=top)

            if args.csv:
                out_path = f"{args.csv}_{interval}.csv" if len(intervals) > 1 else args.csv
                write_candles_csv(out_path, sorted(candles, key=lambda row: row.get("timestamp", 0)))
                print(f"Saved CSV: {out_path}")
        return 0

    market_codes = [m["market"] for m in markets]

    tickers: list[dict] = []
    for chunk in chunked(market_codes, CHUNK_SIZE):
        tickers.extend(fetch_tickers(chunk))

    merged = combine_market_and_ticker(markets, tickers)

    if args.json:
        print(json.dumps(merged, ensure_ascii=False, indent=2))
    else:
        print(f"Fetched {len(merged)} markets at {datetime.now().isoformat(timespec='seconds')}")
        top = args.top if args.top > 0 else None
        print_table(merged, top=top)

    if args.csv:
        write_csv(args.csv, merged)
        print(f"Saved CSV: {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
