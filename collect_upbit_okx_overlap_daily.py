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
from datetime import datetime, timezone

UPBIT_BASE_URL = "https://api.upbit.com/v1"
OKX_BASE_URL = "https://www.okx.com/api/v5"
USER_AGENT = "Mozilla/5.0"
RATE_LIMIT_SLEEP_SECONDS = 15
UPBIT_CACHE_DIR = os.path.expanduser("~/.cache/upbit_okx_overlap/upbit")
OKX_CACHE_DIR = os.path.expanduser("~/.cache/upbit_okx_overlap/okx")
UPBIT_CANDLE_LIMIT = 200
OKX_CANDLE_LIMIT = 100


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_cache(cache_dir: str, url: str):
    digest = __import__("hashlib").sha1(url.encode("utf-8")).hexdigest()
    path = os.path.join(cache_dir, f"{digest}.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:
        return None


def write_cache(cache_dir: str, url: str, payload: object) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{__import__('hashlib').sha1(url.encode('utf-8')).hexdigest()}.json")
    try:
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False)
    except Exception:
        pass


def clear_all_cache() -> None:
    for cache_dir in (UPBIT_CACHE_DIR, OKX_CACHE_DIR):
        if not os.path.isdir(cache_dir):
            continue
        for item in os.listdir(cache_dir):
            p = os.path.join(cache_dir, item)
            if os.path.isfile(p):
                try:
                    os.remove(p)
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


def to_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _to_iso(ts_ms: int) -> str:
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def fetch_json(
    base_url: str,
    path: str,
    params: dict[str, str] | None = None,
    retries: int = 5,
    cache_dir: str | None = None,
):
    query = ""
    if params:
        query = "?" + urllib.parse.urlencode(params)
    url = f"{base_url}{path}{query}"

    if cache_dir is not None:
        cached = read_cache(cache_dir, url)
        if cached is not None:
            return cached

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                raw = resp.read().decode("utf-8")
                payload = json.loads(raw)
                if cache_dir is not None:
                    write_cache(cache_dir, url, payload)
                return payload
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries:
                clear_all_cache()
                print(f"HTTP 429 on {url}. clear cache and wait {RATE_LIMIT_SLEEP_SECONDS}s")
                time.sleep(RATE_LIMIT_SLEEP_SECONDS)
                last_error = e
                continue
            if e.code in {500, 502, 503, 504} and attempt < retries:
                wait = min(2.0 * (2**attempt), 8.0)
                print(f"HTTP {e.code} on {url}. retry in {wait}s")
                time.sleep(wait)
                last_error = e
                continue
            detail = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else ""
            raise RuntimeError(f"HTTP {e.code} on {url}: {detail}") from e
        except urllib.error.URLError as e:
            if attempt < retries:
                wait = min(1.0 * (2**attempt), 8.0)
                print(f"Network error on {url}. retry in {wait}s")
                time.sleep(wait)
                last_error = e
                continue
            raise RuntimeError(f"Network error on {url}: {e}") from e
    raise RuntimeError(f"Request failed after retries: {url}") from last_error


def fetch_upbit_markets() -> list[dict]:
    markets = fetch_json(UPBIT_BASE_URL, "/market/all", {"isDetails": "false"}, cache_dir=UPBIT_CACHE_DIR)
    if not isinstance(markets, list):
        raise RuntimeError("Invalid /market/all response")
    return markets


def fetch_upbit_tickers(markets: list[str]) -> list[dict]:
    if not markets:
        return []
    joined = ",".join(markets)
    return fetch_json(UPBIT_BASE_URL, "/ticker", {"markets": joined}, cache_dir=UPBIT_CACHE_DIR)


def fetch_upbit_candles_all(market: str, count: int, since_ms: int | None = None, max_pages: int = 0) -> list[dict]:
    out: list[dict] = []
    cursor: str | None = None
    pages = 0
    seen: set[int] = set()

    while True:
        if max_pages > 0 and pages >= max_pages:
            break

        params = {
            "market": market,
            "count": str(count),
        }
        if cursor is not None:
            params["to"] = cursor

        page = fetch_json(UPBIT_BASE_URL, "/candles/days", params, cache_dir=UPBIT_CACHE_DIR)
        if not isinstance(page, list) or not page:
            break

        pages += 1
        newest_ts = None
        for row in page:
            ts = row.get("timestamp")
            try:
                ts_i = int(ts)
            except Exception:
                continue

            if ts_i in seen:
                continue
            if since_ms is not None and ts_i < since_ms:
                continue
            seen.add(ts_i)
            out.append(row)
            newest_ts = ts_i if newest_ts is None else max(newest_ts, ts_i)

        if len(page) < count:
            break
        try:
            last_ts = int(page[-1].get("timestamp"))
        except Exception:
            break
        cursor = _to_iso(last_ts - 1)
        if newest_ts is not None and since_ms is not None and newest_ts <= since_ms:
            break

    out.sort(key=lambda row: int(row.get("timestamp", 0)))
    return out


def build_upbit_by_base(markets: list[dict], quote_preference: list[str]) -> dict[str, dict]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in markets:
        market = row.get("market", "")
        if not market or "-" not in market:
            continue
        quote, base = market.split("-", 1)
        grouped[base].append(market)

    selected: dict[str, dict] = {}

    for base, items in grouped.items():
        chosen = None
        for pref in quote_preference:
            for m in items:
                if m.startswith(f"{pref}-"):
                    chosen = m
                    break
            if chosen:
                break
        if chosen is None:
            chosen = sorted(items)[0]

        selected[base] = {
            "market": chosen,
            "all_markets": sorted(items),
            "base": base,
        }

    return selected


def fetch_okx_spot_instruments() -> dict[str, list[dict]]:
    payload = fetch_json(OKX_BASE_URL, "/public/instruments", {"instType": "SPOT"}, cache_dir=OKX_CACHE_DIR)
    if payload.get("code") != "0" or "data" not in payload:
        raise RuntimeError(f"Invalid response from /public/instruments: {payload}")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in payload["data"]:
        inst = row.get("instId", "")
        if "-" not in inst:
            continue
        base, quote = inst.split("-", 1)
        if not base or not quote:
            continue
        if row.get("state") != "live":
            continue
        grouped[base].append(row)
    return grouped


def choose_okx_instruments(base: str, okx_by_base: dict[str, list[dict]], quote_preference: list[str]) -> dict | None:
    candidates = okx_by_base.get(base, [])
    if not candidates:
        return None

    by_quote = {row.get("instId", "").split("-", 1)[1]: row for row in candidates if "-" in row.get("instId", "")}
    for q in quote_preference:
        cand = by_quote.get(q)
        if cand is not None:
            return cand

    return sorted(candidates, key=lambda row: row.get("instId", ""))[0] if candidates else None


def fetch_okx_candles_all(inst_id: str, bar: str, count: int, max_rows: int | None = None) -> list[list[str]]:
    all_rows: list[list[str]] = []
    if max_rows is not None and max_rows <= 0:
        max_rows = None
    seen: set[str] = set()
    after: str | None = None

    while True:
        params = {
            "instId": inst_id,
            "bar": bar,
            "limit": str(count),
        }
        if after:
            params["after"] = after

        payload = fetch_json(OKX_BASE_URL, "/market/candles", params, cache_dir=OKX_CACHE_DIR)
        if payload.get("code") != "0":
            raise RuntimeError(f"OKX API error for {inst_id}: {payload}")

        page = payload.get("data") or []
        if not page:
            break

        for row in page:
            if not isinstance(row, (list, tuple)) or len(row) < 8:
                continue
            ts = row[0]
            if ts in seen:
                continue
            seen.add(ts)
            all_rows.append(list(row))

        if max_rows is not None and len(all_rows) >= max_rows:
            break
        if len(page) < count:
            break

        oldest = page[-1][0]
        try:
            after = str(int(oldest) - 1)
        except Exception:
            break

    all_rows.sort(key=lambda row: int(row[0]))
    if max_rows is not None:
        all_rows = all_rows[-max_rows:]
    return all_rows


def write_upbit_csv(path: str, market: str, rows: list[dict]) -> int:
    ensure_dir(os.path.dirname(path))
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=[
                "market",
                "candle_date_time_utc",
                "candle_date_time_kst",
                "timestamp",
                "opening_price",
                "high_price",
                "low_price",
                "trade_price",
                "candle_acc_trade_volume",
                "candle_acc_trade_price",
            ])
            writer.writeheader()
        return 0

    with open(path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=[
            "market",
            "candle_date_time_utc",
            "candle_date_time_kst",
            "timestamp",
            "opening_price",
            "high_price",
            "low_price",
            "trade_price",
            "candle_acc_trade_volume",
            "candle_acc_trade_price",
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "market": market,
                    "candle_date_time_utc": row.get("candle_date_time_utc"),
                    "candle_date_time_kst": row.get("candle_date_time_kst"),
                    "timestamp": row.get("timestamp"),
                    "opening_price": to_float(row.get("opening_price")),
                    "high_price": to_float(row.get("high_price")),
                    "low_price": to_float(row.get("low_price")),
                    "trade_price": to_float(row.get("trade_price")),
                    "candle_acc_trade_volume": to_float(row.get("candle_acc_trade_volume")),
                    "candle_acc_trade_price": to_float(row.get("candle_acc_trade_price")),
                }
            )
    return len(rows)


def write_okx_csv(path: str, upbit_market: str, okx_inst: str, rows: list[list[str]]) -> int:
    ensure_dir(os.path.dirname(path))
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow([
                "upbit_market",
                "okx_inst",
                "candle_date_time_utc",
                "timestamp",
                "opening_price",
                "high_price",
                "low_price",
                "trade_price",
                "candle_acc_trade_volume",
                "candle_acc_trade_price",
            ])
        return 0

    with open(path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "upbit_market",
                "okx_inst",
                "candle_date_time_utc",
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
                    "upbit_market": upbit_market,
                    "okx_inst": okx_inst,
                    "candle_date_time_utc": _to_iso(ts),
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


def _date_from_iso(raw: str):
    if not raw:
        return None
    return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()


def _date_from_ms(ms: str) -> datetime.date:
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).date()


def read_csv_stats(path: str) -> tuple[int, str | None, str | None]:
    if not os.path.isfile(path):
        return 0, None, None
    try:
        with open(path, encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            rows = [r for r in reader if r]
    except Exception:
        return 0, None, None
    if not rows:
        return 0, None, None

    first = rows[0].get("candle_date_time_utc")
    last = rows[-1].get("candle_date_time_utc")
    if not first or not last:
        return len(rows), None, None
    return len(rows), first, last


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect Upbit<->OKX overlap daily candles.")
    p.add_argument("--out-root", type=str, default="/tmp/upbit_okx_overlap", help="Output root dir")
    p.add_argument("--market-prefix-filter", type=str, default="", help="Filter upbit markets by quote prefix (e.g. KRW)")
    p.add_argument("--upbit-quote-picks", type=str, default="KRW,BTC,USDT,ETH", help="Upbit quote preference for each base")
    p.add_argument("--okx-quote-picks", type=str, default="USDT,USDC,BTC,ETH,USD,EUR", help="OKX quote preference")
    p.add_argument("--upbit-candle-count", type=int, default=UPBIT_CANDLE_LIMIT, help="Upbit candles per page")
    p.add_argument("--okx-candle-count", type=int, default=OKX_CANDLE_LIMIT, help="OKX candles per page")
    p.add_argument("--max-pages", type=int, default=0, help="Max Upbit pages per market (0=all)")
    p.add_argument("--max-rows", type=int, default=0, help="Max rows per market (0=no cap)")
    p.add_argument("--summary", type=str, default="", help="Save summary json path (default print)")
    p.add_argument("--notify", action="store_true", help="Enable progress notify-send")
    p.add_argument("--notify-every", type=int, default=10, help="Notify every N processed symbols")
    p.add_argument("--max-symbols", type=int, default=0, help="Process only first N overlap symbols")
    p.add_argument("--top-volume", type=int, default=0, help="Limit mapped symbols by top upbit 24h volume")
    p.add_argument("--start-index", type=int, default=1, help="Resume from this 1-based symbol index")
    p.add_argument("--resume", action="store_true", help="Skip markets that already have output files")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.upbit_candle_count < 1 or args.upbit_candle_count > 200:
        raise SystemExit("--upbit-candle-count must be 1..200")
    if args.okx_candle_count < 1 or args.okx_candle_count > 300:
        raise SystemExit("--okx-candle-count must be 1..300")
    if args.max_pages < 0 or args.max_rows < 0 or args.notify_every < 0:
        raise SystemExit("--max-pages/--max-rows/--notify-every must be >= 0")

    upbit_prefix = args.market_prefix_filter.upper().strip()
    upbit_quote_picks = [q.strip().upper() for q in args.upbit_quote_picks.split(",") if q.strip()]
    okx_quote_picks = [q.strip().upper() for q in args.okx_quote_picks.split(",") if q.strip()]

    out_root = args.out_root
    upbit_dir = os.path.join(out_root, "upbit")
    okx_dir = os.path.join(out_root, "okx")
    ensure_dir(upbit_dir)
    ensure_dir(okx_dir)

    print("Fetching Upbit market list ...")
    upbit_markets = fetch_upbit_markets()
    if upbit_prefix:
        filtered = []
        for row in upbit_markets:
            code = row.get("market", "")
            if code == upbit_prefix or code.startswith(f"{upbit_prefix}-"):
                filtered.append(row)
        upbit_markets = filtered
        if not upbit_markets:
            raise SystemExit(f"No upbit market matched prefix={upbit_prefix}")

    print("Fetching OKX spot instruments ...")
    okx_by_base = fetch_okx_spot_instruments()

    upbit_by_base = build_upbit_by_base(upbit_markets, upbit_quote_picks)

    overlap = sorted(set(upbit_by_base.keys()) & set(okx_by_base.keys()))
    overlap_map = []
    for base in overlap:
        upbit_market = upbit_by_base[base]["market"]
        okx_inst_row = choose_okx_instruments(base, okx_by_base, okx_quote_picks)
        if okx_inst_row is None:
            continue
        overlap_map.append((base, upbit_market, okx_inst_row["instId"]))

    if not overlap_map:
        raise SystemExit("No overlap market found with current quote preferences.")

    # volume-based sorting + limiting for overlap
    if args.top_volume > 0:
        codes = [up for _, up, _ in overlap_map]
        print("Applying top-volume filter from Upbit ...")
        ranked_tickers = []
        for chunk_start in range(0, len(codes), 100):
            chunk = codes[chunk_start : chunk_start + 100]
            tickers = fetch_upbit_tickers(chunk)
            if not isinstance(tickers, list):
                continue
            for row in tickers:
                ranked_tickers.append((row.get("market", ""), to_float(row.get("acc_trade_price_24h")) or 0.0))

        vol_map = {m: v for m, v in ranked_tickers}
        overlap_map.sort(key=lambda item: vol_map.get(item[1], 0.0), reverse=True)
        overlap_map = overlap_map[:args.top_volume]

    if args.max_symbols > 0:
        overlap_map = overlap_map[: args.max_symbols]

    if args.notify:
        send_notification("Upbit/OKX 교집합 수집 시작", f"총 {len(overlap_map)}개 코인")

    start_time = time.time()
    summary: dict = {
        "target_count": len(overlap_map),
        "overlap_count": len(overlap_map),
        "upbit_prefix_filter": upbit_prefix or None,
        "upbit_quote_picks": upbit_quote_picks,
        "okx_quote_picks": okx_quote_picks,
        "rows": [],
    }

    saved_pairs: list[tuple[datetime, datetime]] = []

    start_idx = max(1, args.start_index)
    for idx, (base, upbit_market, okx_inst) in enumerate(overlap_map, start=1):
        if idx < start_idx:
            continue
        print(f"[{idx:03d}/{len(overlap_map)}] {upbit_market} <-> {okx_inst}")

        upbit_path = os.path.join(upbit_dir, f"{upbit_market}_day.csv")
        okx_path = os.path.join(okx_dir, f"{upbit_market}_day.csv")

        if args.resume and os.path.exists(upbit_path) and os.path.exists(okx_path):
            upbit_cnt, upbit_start, upbit_end = read_csv_stats(upbit_path)
            okx_cnt, okx_start, okx_end = read_csv_stats(okx_path)
            if upbit_cnt > 0 and okx_cnt > 0:
                upbit_start_dt = _date_from_iso(upbit_start) if upbit_start else None
                upbit_end_dt = _date_from_iso(upbit_end) if upbit_end else None
                okx_start_dt = _date_from_iso(okx_start) if okx_start else None
                okx_end_dt = _date_from_iso(okx_end) if okx_end else None
                if upbit_start_dt and upbit_end_dt and okx_start_dt and okx_end_dt:
                    intersection_start = max(upbit_start_dt, okx_start_dt)
                    intersection_end = min(upbit_end_dt, okx_end_dt)
                    if intersection_start <= intersection_end:
                        saved_pairs.append((intersection_start, intersection_end))
                    else:
                        intersection_start = None
                        intersection_end = None
                else:
                    intersection_start = None
                    intersection_end = None

                summary["rows"].append(
                    {
                        "base": base,
                        "upbit_market": upbit_market,
                        "okx_inst": okx_inst,
                        "upbit_file": upbit_path,
                        "okx_file": okx_path,
                        "upbit_rows": upbit_cnt,
                        "okx_rows": okx_cnt,
                        "upbit_start": upbit_start,
                        "upbit_end": upbit_end,
                        "okx_start": okx_start,
                        "okx_end": okx_end,
                        "intersection_start": intersection_start.isoformat() if intersection_start else None,
                        "intersection_end": intersection_end.isoformat() if intersection_end else None,
                    }
                )
                print(f"  skipped (resume): Upbit={upbit_cnt} OKX={okx_cnt}")
                continue

        upbit_rows = fetch_upbit_candles_all(
            market=upbit_market,
            count=args.upbit_candle_count,
            max_pages=args.max_pages,
        )
        if args.max_rows > 0:
            upbit_rows = upbit_rows[-args.max_rows :]

        okx_rows = fetch_okx_candles_all(
            inst_id=okx_inst,
            bar="1D",
            count=args.okx_candle_count,
            max_rows=None if args.max_rows <= 0 else args.max_rows,
        )

        upbit_cnt = write_upbit_csv(upbit_path, upbit_market, upbit_rows)
        okx_cnt = write_okx_csv(okx_path, upbit_market, okx_inst, okx_rows)

        upbit_start = upbit_end = None
        okx_start = okx_end = None
        upbit_start_ms = upbit_rows[0].get("timestamp") if upbit_rows else None
        upbit_end_ms = upbit_rows[-1].get("timestamp") if upbit_rows else None
        if upbit_start_ms:
            upbit_start = _date_from_ms(upbit_start_ms)
            upbit_end = _date_from_ms(upbit_end_ms)

        if okx_rows:
            okx_start = _date_from_ms(okx_rows[0][0])
            okx_end = _date_from_ms(okx_rows[-1][0])

        if upbit_start and upbit_end and okx_start and okx_end:
            intersection_start = max(upbit_start, okx_start)
            intersection_end = min(upbit_end, okx_end)
            saved_pairs.append((intersection_start, intersection_end))
        else:
            intersection_start = None
            intersection_end = None

        summary["rows"].append(
            {
                "base": base,
                "upbit_market": upbit_market,
                "okx_inst": okx_inst,
                "upbit_file": upbit_path,
                "okx_file": okx_path,
                "upbit_rows": upbit_cnt,
                "okx_rows": okx_cnt,
                "upbit_start": upbit_start.isoformat() if upbit_start else None,
                "upbit_end": upbit_end.isoformat() if upbit_end else None,
                "okx_start": okx_start.isoformat() if okx_start else None,
                "okx_end": okx_end.isoformat() if okx_end else None,
                "intersection_start": intersection_start.isoformat() if intersection_start else None,
                "intersection_end": intersection_end.isoformat() if intersection_end else None,
            }
        )

        print(f"  saved Upbit={upbit_cnt} OKX={okx_cnt}")
        if args.notify and (args.notify_every == 0 or idx % max(args.notify_every, 1) == 0):
            elapsed = int(time.time() - start_time)
            elapsed_s = f"{elapsed//3600:02d}:{(elapsed%3600)//60:02d}:{elapsed%60:02d}"
            send_notification(
                "Upbit/OKX 수집 진행",
                f"{idx}/{len(overlap_map)} 완료\nUpbit={upbit_cnt} / OKX={okx_cnt}\n경과: {elapsed_s}",
            )

    if saved_pairs:
        union_start = min(s[0] for s in saved_pairs)
        union_end = max(s[1] for s in saved_pairs)
        inter_start = max(s[0] for s in saved_pairs)
        inter_end = min(s[1] for s in saved_pairs)
    else:
        union_start = union_end = inter_start = inter_end = None

    summary["periods"] = {
        "union": {
            "start": union_start.isoformat() if union_start else None,
            "end": union_end.isoformat() if union_end else None,
        },
        "intersection": {
            "start": inter_start.isoformat() if inter_start else None,
            "end": inter_end.isoformat() if inter_end else None,
        },
        "upbit_okx_common_pairs_count": len(saved_pairs),
    }

    if args.summary:
        with open(args.summary, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(summary["periods"], ensure_ascii=False, indent=2))

    if args.notify:
        elapsed = int(time.time() - start_time)
        elapsed_s = f"{elapsed//3600:02d}:{(elapsed%3600)//60:02d}:{elapsed%60:02d}"
        send_notification("Upbit/OKX 교집합 수집 완료", f"총 {len(overlap_map)}개 코인\n소요: {elapsed_s}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
