#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import math
import os
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import upbit_strategy_engine as engine


UPBIT_API_BASE = "https://api.upbit.com"


def to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def format_volume(value: float) -> str:
    text = f"{value:.16f}".rstrip("0").rstrip(".")
    return text or "0"


def format_number_text(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    text = f"{value:.8f}".rstrip("0").rstrip(".")
    return text or "0"


def chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def encode_jwt_hs512(payload: dict, secret_key: str) -> str:
    header = {"alg": "HS512", "typ": "JWT"}
    encoded_header = b64url(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    encoded_payload = b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{encoded_header}.{encoded_payload}".encode("ascii")
    signature = hmac.new(secret_key.encode("utf-8"), signing_input, hashlib.sha512).digest()
    encoded_signature = b64url(signature)
    return f"{encoded_header}.{encoded_payload}.{encoded_signature}"


class UpbitClient:
    def __init__(self, access_key: str | None, secret_key: str | None, timeout: float = 15.0, retries: int = 3):
        self.access_key = access_key or ""
        self.secret_key = secret_key or ""
        self.timeout = timeout
        self.retries = retries

    def _auth_header(self, query_string: str) -> str:
        if not self.access_key or not self.secret_key:
            raise RuntimeError("UPBIT_KEY/UPBIT_SECRET not set")
        payload = {
            "access_key": self.access_key,
            "nonce": str(uuid.uuid4()),
        }
        if query_string:
            payload["query_hash"] = hashlib.sha512(query_string.encode("utf-8")).hexdigest()
            payload["query_hash_alg"] = "SHA512"
        token = encode_jwt_hs512(payload, self.secret_key)
        return f"Bearer {token}"

    def request_json(
        self,
        method: str,
        path: str,
        params: dict[str, str] | None = None,
        *,
        private: bool = False,
    ) -> object:
        query_string = urlencode(params or {}, doseq=True)
        url = f"{UPBIT_API_BASE}{path}"
        data = None
        headers = {"Accept": "application/json"}

        if method.upper() == "GET":
            if query_string:
                url = f"{url}?{query_string}"
        else:
            headers["Content-Type"] = "application/json"
            data = json.dumps(params or {}, separators=(",", ":")).encode("utf-8")

        if private:
            headers["Authorization"] = self._auth_header(query_string)

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            req = Request(url=url, data=data, headers=headers, method=method.upper())
            try:
                with urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read().decode("utf-8")
                    return json.loads(raw) if raw else {}
            except HTTPError as err:
                body = err.read().decode("utf-8", errors="replace")
                retriable = err.code in (429, 500, 502, 503, 504)
                if retriable and attempt < self.retries:
                    time.sleep(0.6 * (2**attempt))
                    continue
                last_error = RuntimeError(f"Upbit API HTTP {err.code} {path}: {body}")
                break
            except URLError as err:
                if attempt < self.retries:
                    time.sleep(0.6 * (2**attempt))
                    continue
                last_error = RuntimeError(f"Upbit API network error {path}: {err}")
                break

        raise last_error or RuntimeError(f"Upbit API request failed: {path}")

    def get(self, path: str, params: dict[str, str] | None = None, *, private: bool = False) -> object:
        return self.request_json("GET", path, params=params, private=private)

    def post(self, path: str, params: dict[str, str] | None = None, *, private: bool = False) -> object:
        return self.request_json("POST", path, params=params, private=private)


def _strip_quotes(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1].strip()
    return text


def _read_assignment_from_text(text: str, key: str) -> str:
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=\s*(.+?)\s*$")
    for line in text.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        raw = match.group(1).strip()
        if " #" in raw:
            raw = raw.split(" #", 1)[0].rstrip()
        value = _strip_quotes(raw)
        if value:
            return value
    return ""


def _candidate_files(primary: str, fallback_names: list[str]) -> list[Path]:
    out: list[Path] = []
    for item in [primary, *fallback_names]:
        if not item:
            continue
        path = Path(item)
        if not path.is_absolute():
            out.append(Path.cwd() / path)
        out.append(path)
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in out:
        key = str(path.resolve()) if path.exists() else str(path.absolute())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def resolve_upbit_keys(secret_file: str) -> tuple[str, str, str]:
    key = os.getenv("UPBIT_KEY", "").strip()
    secret = os.getenv("UPBIT_SECRET", "").strip()
    if key and secret:
        return key, secret, "env"

    files = _candidate_files(
        secret_file,
        [
            "개인정보",
            "개인정보.txt",
            "/home/ubuntu/개인정보.txt",
            "/home/ubuntu/코인매매프로그램/개인정보",
        ],
    )
    for path in files:
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        file_key = _read_assignment_from_text(text, "UPBIT_KEY")
        file_secret = _read_assignment_from_text(text, "UPBIT_SECRET")
        if file_key and file_secret:
            return file_key, file_secret, f"file:{path}"

    raise SystemExit("UPBIT_KEY/UPBIT_SECRET not found in env or secret file")


def _read_webhook_from_file(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    for line in text.splitlines():
        item = line.strip()
        if item.startswith("https://discord.com/api/webhooks/"):
            return item
    value = _read_assignment_from_text(text, "DISCORD_WEBHOOK_URL")
    if value:
        return value
    value = _read_assignment_from_text(text, "WONYODD_DISCORD_WEBHOOK_URL")
    if value:
        return value
    return ""


def resolve_webhook_url(notify_webhook_url: str, notify_file: str) -> tuple[str, str]:
    direct = notify_webhook_url.strip()
    if direct:
        return direct, "arg"

    for env_key in ("DISCORD_WEBHOOK_URL", "WONYODD_DISCORD_WEBHOOK_URL"):
        value = os.getenv(env_key, "").strip()
        if value:
            return value, f"env:{env_key}"

    files = _candidate_files(
        notify_file,
        [
            "개인정보",
            "개인정보.txt",
            "/home/ubuntu/개인정보.txt",
            "/home/ubuntu/코인매매프로그램/개인정보",
        ],
    )
    for path in files:
        url = _read_webhook_from_file(path)
        if url:
            return url, f"file:{path}"
    return "", ""


def build_discord_payload(title: str, lines: list[str], color: int = 0x4B6BB5) -> dict:
    text = "\n".join(lines).strip()
    if len(text) > 3900:
        text = text[:3896] + " ..."
    return {
        "embeds": [
            {
                "title": title,
                "description": text or "-",
                "color": color,
            }
        ]
    }


def send_json_webhook(url: str, payload: dict, timeout: float = 8.0) -> tuple[bool, str]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            if 200 <= resp.status < 300:
                return True, "sent"
            return False, f"http_{resp.status}"
    except Exception as exc:
        return False, str(exc)


def resolve_order_webhook_info(
    info_file: str,
    webhook_url_arg: str,
    webhook_password_arg: str,
) -> tuple[str, str, str]:
    if webhook_url_arg.strip() and webhook_password_arg.strip():
        return webhook_url_arg.strip(), webhook_password_arg.strip(), "arg"

    env_url = os.getenv("AUTO_TRADE_WEBHOOK_URL", "").strip()
    env_pw = os.getenv("AUTO_TRADE_WEBHOOK_PASSWORD", "").strip()
    if env_url and env_pw:
        return env_url, env_pw, "env"

    files = _candidate_files(
        info_file,
        [
            "/home/ubuntu/자동매매정보.txt",
            "자동매매정보.txt",
        ],
    )
    for path in files:
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        url_match = re.search(r"자동매매\s*웹훅\s*주소\s*:\s*(\S+)", text)
        pw_match = re.search(r'"password"\s*:\s*"([^"]+)"', text)
        if url_match and pw_match:
            return url_match.group(1).strip(), pw_match.group(1).strip(), f"file:{path}"

    raise SystemExit("order webhook url/password not found (자동매매정보 또는 인자/환경변수 확인)")


def load_json_file(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        raw = p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return {}
    return {}


def save_json_file(path: str, payload: dict) -> None:
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def utc_today_text() -> str:
    return datetime.utcnow().date().isoformat()


def parse_iso_date_safe(value: str) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def days_since_text(today_text: str, base_text: str) -> int | None:
    today = parse_iso_date_safe(today_text)
    base = parse_iso_date_safe(base_text)
    if today is None or base is None:
        return None
    return (today - base).days


def open_engine_db(path: str) -> sqlite3.Connection | None:
    target = str(path or "").strip()
    if not target:
        return None
    db_path = Path(target)
    if db_path.parent and not db_path.parent.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS autotrade_order_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_ts INTEGER NOT NULL,
            run_date TEXT NOT NULL,
            mode TEXT NOT NULL,
            market TEXT NOT NULL,
            side TEXT NOT NULL,
            reason TEXT NOT NULL,
            qty REAL,
            price REAL,
            notional REAL,
            order_uuid TEXT
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_autotrade_order_log_run_date ON autotrade_order_log(run_date)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_autotrade_order_log_market_side ON autotrade_order_log(market, side, run_date)"
    )
    conn.commit()
    return conn


def db_fetch_last_buy_date_market(conn: sqlite3.Connection, market: str) -> str:
    row = conn.execute(
        "SELECT run_date FROM autotrade_order_log WHERE market=? AND side='buy' ORDER BY run_date DESC, id DESC LIMIT 1",
        (market,),
    ).fetchone()
    if not row:
        return ""
    return str(row["run_date"] or "")


def db_count_trades_on_date(conn: sqlite3.Connection, run_date: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS c FROM autotrade_order_log WHERE run_date=?",
        (run_date,),
    ).fetchone()
    if not row:
        return 0
    try:
        return int(row["c"])
    except Exception:
        return 0


def db_count_trades_since_ts(conn: sqlite3.Connection, start_ts: int) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS c FROM autotrade_order_log WHERE run_ts >= ?",
        (int(start_ts),),
    ).fetchone()
    if not row:
        return 0
    try:
        return int(row["c"])
    except Exception:
        return 0


def db_fetch_last_exit_date_market(conn: sqlite3.Connection, market: str) -> str:
    row = conn.execute(
        """
        SELECT run_date
          FROM autotrade_order_log
         WHERE market=? AND (side='sell' OR side='sell_rebalance' OR side='sell_final')
         ORDER BY run_date DESC, id DESC
         LIMIT 1
        """,
        (market,),
    ).fetchone()
    if not row:
        return ""
    return str(row["run_date"] or "")


def db_insert_order_log(
    conn: sqlite3.Connection,
    *,
    run_ts: int,
    run_date: str,
    mode: str,
    market: str,
    side: str,
    reason: str,
    qty: float,
    price: float,
    notional: float,
    order_uuid: str,
) -> None:
    conn.execute(
        """
        INSERT INTO autotrade_order_log(
            run_ts, run_date, mode, market, side, reason, qty, price, notional, order_uuid
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(run_ts),
            str(run_date),
            str(mode),
            str(market),
            str(side),
            str(reason),
            float(qty),
            float(price),
            float(notional),
            str(order_uuid or ""),
        ),
    )
    conn.commit()


def load_backtest_summary(path: str) -> dict:
    payload = load_json_file(path)
    if not isinstance(payload, dict):
        return {}
    summary = payload.get("summary")
    return summary if isinstance(summary, dict) else {}


@dataclass
class Holding:
    market: str
    qty: float
    avg_buy_price: float


@dataclass
class SignalSnapshot:
    market: str
    signal_date: str
    last_price: float
    enter: bool
    exit: bool
    score: float | None
    rows: int


def fetch_markets(client: UpbitClient, market_prefix: str) -> list[str]:
    rows = client.get("/v1/market/all", {"isDetails": "false"})
    if not isinstance(rows, list):
        raise RuntimeError("invalid response from /v1/market/all")
    prefix = f"{market_prefix.upper()}-"
    out: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        market = str(row.get("market", "")).upper()
        if market.startswith(prefix):
            out.append(market)
    return sorted(set(out))


def fetch_ticker_map(client: UpbitClient, markets: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for group in chunks(markets, 100):
        try:
            rows = client.get("/v1/ticker", {"markets": ",".join(group)})
        except RuntimeError as exc:
            message = str(exc)
            # Upbit returns 404 for the whole request if even one market code is invalid.
            if "HTTP 404" not in message or "Code not found" not in message:
                raise
            rows = []
            for market in group:
                try:
                    item_rows = client.get("/v1/ticker", {"markets": market})
                except RuntimeError as item_exc:
                    item_message = str(item_exc)
                    if "HTTP 404" in item_message and "Code not found" in item_message:
                        continue
                    raise
                if isinstance(item_rows, list):
                    rows.extend(item_rows)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            m = str(row.get("market", "")).upper()
            if m:
                out[m] = row
    return out


def select_universe_by_volume(client: UpbitClient, markets: list[str], top_n: int) -> list[str]:
    if top_n <= 0 or top_n >= len(markets):
        return markets
    ticker_map = fetch_ticker_map(client, markets)
    ranked = sorted(
        markets,
        key=lambda m: to_float(ticker_map.get(m, {}).get("acc_trade_price_24h")),
        reverse=True,
    )
    return ranked[:top_n]


def fetch_daily_candles(client: UpbitClient, market: str, count: int) -> tuple[list[str], list[float], list[float]]:
    rows = client.get("/v1/candles/days", {"market": market, "count": str(count)})
    if not isinstance(rows, list):
        return [], [], []
    rows = list(rows)
    rows.reverse()
    dates: list[str] = []
    closes: list[float] = []
    volumes: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        dt = str(row.get("candle_date_time_utc") or "")
        close = to_float(row.get("trade_price"), default=math.nan)
        vol = to_float(row.get("candle_acc_trade_volume"), default=0.0)
        if not dt or math.isnan(close):
            continue
        dates.append(dt)
        closes.append(close)
        volumes.append(vol)
    return dates, closes, volumes


def strategy_params_from_args(args: argparse.Namespace) -> tuple[str, dict[str, float | int]]:
    if args.strategy == "ma":
        return "ma", {"short_window": args.ma_short, "long_window": args.ma_long}
    if args.strategy == "momentum":
        return "momentum", {
            "momentum_window": args.momentum_window,
            "vol_window": args.momentum_vol_window,
            "enter_threshold": args.momentum_enter,
            "exit_threshold": args.momentum_exit,
        }
    return "rsi", {
        "rsi_window": args.rsi_window,
        "buy_threshold": args.rsi_buy,
        "sell_threshold": args.rsi_sell,
    }


def compute_signals(
    client: UpbitClient,
    markets: list[str],
    strategy: str,
    params: dict[str, float | int],
    *,
    candle_count: int,
    min_data_days: int,
    signal_offset: int,
    pause_sec: float,
) -> dict[str, SignalSnapshot]:
    strategy_fn = engine.make_strategy(strategy)
    out: dict[str, SignalSnapshot] = {}
    for idx, market in enumerate(markets, start=1):
        dates, closes, volumes = fetch_daily_candles(client, market, candle_count)
        if len(closes) < max(10, min_data_days):
            continue
        enters, exits, scores = strategy_fn(closes, volumes, params)
        signal_idx = len(closes) - 1 - signal_offset
        if signal_idx < 1 or signal_idx >= len(closes):
            continue
        out[market] = SignalSnapshot(
            market=market,
            signal_date=dates[signal_idx],
            last_price=closes[-1],
            enter=bool(enters[signal_idx]),
            exit=bool(exits[signal_idx]),
            score=scores[signal_idx],
            rows=len(closes),
        )
        if pause_sec > 0 and idx < len(markets):
            time.sleep(pause_sec)
    return out


def fetch_account_state(client: UpbitClient, market_prefix: str) -> tuple[float, dict[str, Holding]]:
    rows = client.get("/v1/accounts", private=True)
    if not isinstance(rows, list):
        raise RuntimeError("invalid response from /v1/accounts")
    prefix = market_prefix.upper()
    krw_balance = 0.0
    holdings: dict[str, Holding] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        currency = str(row.get("currency", "")).upper()
        unit_currency = str(row.get("unit_currency", "")).upper()
        balance = to_float(row.get("balance"), default=0.0)
        locked = to_float(row.get("locked"), default=0.0)
        total = balance + locked
        if currency == prefix:
            krw_balance = total
            continue
        if unit_currency != prefix:
            continue
        if total <= 0:
            continue
        market = f"{prefix}-{currency}"
        holdings[market] = Holding(
            market=market,
            qty=total,
            avg_buy_price=to_float(row.get("avg_buy_price"), default=0.0),
        )
    return krw_balance, holdings


def rank_candidates(signals: dict[str, SignalSnapshot], max_positions: int) -> list[str]:
    scored = [s for s in signals.values() if s.enter and s.score is not None]
    scored.sort(key=lambda s: float(s.score), reverse=True)
    return [s.market for s in scored[:max_positions]]


def decide_webhook_action(snapshot: SignalSnapshot | None) -> str:
    if snapshot is None:
        return "hold"
    if snapshot.exit:
        return "sell"
    if snapshot.enter:
        return "buy"
    return "hold"


def decide_sells(
    holdings: dict[str, Holding],
    targets: list[str],
    signals: dict[str, SignalSnapshot],
    ticker_map: dict[str, dict],
    *,
    stop_loss: float | None,
    take_profit: float | None,
) -> list[tuple[str, str]]:
    target_set = set(targets)
    has_targets = bool(target_set)
    sells: list[tuple[str, str]] = []
    for market, holding in holdings.items():
        snap = signals.get(market)
        reason = ""
        now_price = to_float(ticker_map.get(market, {}).get("trade_price"), default=0.0)
        if stop_loss and holding.avg_buy_price > 0 and now_price > 0:
            if now_price <= holding.avg_buy_price * (1.0 - stop_loss):
                reason = "stop_loss"
        if not reason and take_profit and holding.avg_buy_price > 0 and now_price > 0:
            if now_price >= holding.avg_buy_price * (1.0 + take_profit):
                reason = "take_profit"
        if not reason and snap and snap.exit:
            reason = "exit_signal"
        if not reason and has_targets and market not in target_set:
            reason = "not_in_target"
        if reason:
            sells.append((market, reason))
    return sells


def place_market_sell(client: UpbitClient, market: str, qty: float) -> dict:
    payload = {
        "market": market,
        "side": "ask",
        "ord_type": "market",
        "volume": format_volume(qty),
    }
    resp = client.post("/v1/orders", payload, private=True)
    if not isinstance(resp, dict):
        raise RuntimeError(f"invalid sell response for {market}")
    return resp


def place_market_buy(client: UpbitClient, market: str, krw_amount: float) -> dict:
    price = int(math.floor(krw_amount))
    if price <= 0:
        raise RuntimeError(f"invalid buy amount for {market}: {krw_amount}")
    payload = {
        "market": market,
        "side": "bid",
        "ord_type": "price",
        "price": str(price),
    }
    resp = client.post("/v1/orders", payload, private=True)
    if not isinstance(resp, dict):
        raise RuntimeError(f"invalid buy response for {market}")
    return resp


def main() -> int:
    parser = argparse.ArgumentParser(description="Run strategy signals on Upbit and execute rebalance orders.")
    parser.add_argument(
        "--execution",
        type=str,
        default="webhook",
        choices=["webhook", "upbit"],
        help="Execution backend: webhook (default) or direct upbit",
    )
    parser.add_argument("--market", type=str, default="KRW", help="Quote market prefix (default: KRW)")
    parser.add_argument("--universe-top", type=int, default=30, help="Top-N liquid markets by 24h volume (0=all)")
    parser.add_argument("--candle-count", type=int, default=200, help="Daily candle count per market (1..200)")
    parser.add_argument("--min-data-days", type=int, default=120, help="Skip markets with fewer rows than this")
    parser.add_argument("--signal-offset", type=int, default=1, help="Use n-th candle from latest for signal (default=1)")
    parser.add_argument("--max-positions", type=int, default=3, help="Maximum concurrent positions")
    parser.add_argument("--min-order-krw", type=float, default=5500, help="Minimum notional per order in KRW")
    parser.add_argument("--cash-buffer-krw", type=float, default=10000, help="Keep this KRW unallocated")
    parser.add_argument("--stop-loss", type=float, default=0.0, help="Optional stop loss ratio (e.g. 0.08)")
    parser.add_argument("--take-profit", type=float, default=0.0, help="Optional take profit ratio (e.g. 0.20)")
    parser.add_argument("--request-pause", type=float, default=0.05, help="Pause seconds between per-market candle requests")
    parser.add_argument("--engine-db", type=str, default="", help="SQLite DB path for order log/risk controls")
    parser.add_argument(
        "--entry-cooldown-days",
        "--buy-cooldown-days",
        dest="entry_cooldown_days",
        type=int,
        default=3,
        help="Cooldown days after any sell before re-entry",
    )
    parser.add_argument(
        "--min-holding-days",
        "--min-hold-days",
        dest="min_holding_days",
        type=int,
        default=3,
        help="Minimum holding days before signal/rebalance exits",
    )
    parser.add_argument("--max-trades-per-day", type=int, default=0, help="Maximum live orders per UTC day (0=unlimited)")
    parser.add_argument("--max-trades-per-year", type=float, default=90.0, help="Maximum live orders per 365 days (0=unlimited)")
    parser.add_argument("--backtest-summary-json", type=str, default="", help="Backtest summary JSON path for live guard")
    parser.add_argument("--min-backtest-return", type=float, default=-0.20, help="Block live if backtest total return is lower")
    parser.add_argument("--max-backtest-dd-usage", type=float, default=0.60, help="Block live if abs(total_return)/MDD is higher")
    parser.add_argument("--require-backtest-summary", action="store_true", help="Require backtest summary before live run")

    parser.add_argument("--strategy", type=str, default="ma", choices=["ma", "momentum", "rsi"])
    parser.add_argument("--ma-short", type=int, default=5)
    parser.add_argument("--ma-long", type=int, default=20)
    parser.add_argument("--momentum-window", type=int, default=10)
    parser.add_argument("--momentum-vol-window", type=int, default=20)
    parser.add_argument("--momentum-enter", type=float, default=0.02)
    parser.add_argument("--momentum-exit", type=float, default=-0.03)
    parser.add_argument("--rsi-window", type=int, default=14)
    parser.add_argument("--rsi-buy", type=float, default=30.0)
    parser.add_argument("--rsi-sell", type=float, default=60.0)

    parser.add_argument("--dry-run", action="store_true", help="Simulate only, do not place orders")
    parser.add_argument("--live", action="store_true", help="Place real orders (requires API keys)")
    parser.add_argument("--show-top", type=int, default=10, help="Print top candidate list")
    parser.add_argument("--secret-file", type=str, default="개인정보", help="Path to secret file containing UPBIT_KEY/UPBIT_SECRET")
    parser.add_argument("--notify-json", action="store_true", help="Send execution summary as JSON webhook message")
    parser.add_argument("--notify-file", type=str, default="개인정보", help="Path to file containing webhook URL")
    parser.add_argument("--notify-webhook-url", type=str, default="", help="Webhook URL override")
    parser.add_argument("--notify-timeout", type=float, default=8.0, help="Webhook request timeout in seconds")
    parser.add_argument("--order-info-file", type=str, default="/home/ubuntu/자동매매정보.txt", help="Path to auto-trade webhook info file")
    parser.add_argument("--order-webhook-url", type=str, default="", help="Order webhook URL override")
    parser.add_argument("--order-webhook-password", type=str, default="", help="Order webhook password override")
    parser.add_argument("--webhook-market", type=str, default="KRW-BTC", help="Market used for webhook action decision")
    parser.add_argument("--webhook-exchange", type=str, default="UPBIT", help="exchange field in order webhook payload")
    parser.add_argument("--webhook-buy-percent", type=float, default=95.0, help="percent for buy order payload")
    parser.add_argument("--webhook-sell-percent", type=float, default=100.0, help="percent for sell order payload")
    parser.add_argument("--webhook-state-file", type=str, default="/tmp/upbit_live_webhook_state.json", help="State file to prevent duplicate sends")
    parser.add_argument("--force-send", action="store_true", help="Send webhook even if already sent for same signal date/action")
    args = parser.parse_args()

    if args.candle_count < 1 or args.candle_count > 200:
        raise SystemExit("--candle-count must be 1..200")
    if args.max_positions < 1:
        raise SystemExit("--max-positions must be >= 1")
    if args.signal_offset < 1:
        raise SystemExit("--signal-offset must be >= 1")
    if args.live and args.dry_run:
        raise SystemExit("use either --dry-run or --live")
    if not args.live and not args.dry_run:
        args.dry_run = True
    if args.entry_cooldown_days < 0:
        raise SystemExit("--entry-cooldown-days must be >= 0")
    if args.min_holding_days < 0:
        raise SystemExit("--min-holding-days must be >= 0")
    if args.max_trades_per_day < 0:
        raise SystemExit("--max-trades-per-day must be >= 0")
    if args.max_trades_per_year < 0:
        raise SystemExit("--max-trades-per-year must be >= 0")

    if args.execution == "upbit":
        access_key, secret_key, key_source = resolve_upbit_keys(args.secret_file)
        print(f"credentials loaded from {key_source}")
    else:
        access_key, secret_key = "", ""
        print("execution=webhook: upbit private key not required")

    order_webhook_url = ""
    order_webhook_password = ""
    if args.execution == "webhook":
        order_webhook_url, order_webhook_password, order_source = resolve_order_webhook_info(
            args.order_info_file,
            args.order_webhook_url,
            args.order_webhook_password,
        )
        print(f"order webhook loaded from {order_source}")

    webhook_url = ""
    webhook_source = ""
    if args.notify_json:
        webhook_url, webhook_source = resolve_webhook_url(args.notify_webhook_url, args.notify_file)
        if webhook_url:
            print(f"notify webhook loaded from {webhook_source}")
        else:
            print("notify webhook not found; continuing without notifications")

    def notify(title: str, lines: list[str], *, color: int = 0x4B6BB5) -> None:
        if not args.notify_json or not webhook_url:
            return
        payload = build_discord_payload(title=title, lines=lines, color=color)
        ok, detail = send_json_webhook(webhook_url, payload, timeout=max(1.0, args.notify_timeout))
        if ok:
            print("  notify sent")
        else:
            print(f"  notify failed: {detail}")

    run_ts = int(time.time())
    run_date = utc_today_text()
    engine_db = open_engine_db(args.engine_db)
    if engine_db:
        print(f"engine db connected: {args.engine_db}")

    backtest_summary = load_backtest_summary(args.backtest_summary_json)
    bt_total_return = to_float(backtest_summary.get("total_return"), default=0.0) if backtest_summary else 0.0
    bt_dd_usage = to_float(backtest_summary.get("total_return_vs_mdd_downside"), default=0.0) if backtest_summary else 0.0
    if backtest_summary:
        print(
            "backtest guard loaded: "
            f"total_return={bt_total_return*100:.2f}% dd_usage={bt_dd_usage:.3f}"
        )
    elif args.backtest_summary_json:
        print(f"backtest guard unavailable: {args.backtest_summary_json}")

    if args.live:
        if args.require_backtest_summary and not backtest_summary:
            raise SystemExit("backtest summary required for live mode")
        if backtest_summary:
            if bt_total_return < args.min_backtest_return:
                raise SystemExit(
                    f"live blocked: backtest total return {bt_total_return:.4f} < min {args.min_backtest_return:.4f}"
                )
            if args.max_backtest_dd_usage > 0 and bt_dd_usage > args.max_backtest_dd_usage:
                raise SystemExit(
                    f"live blocked: backtest dd usage {bt_dd_usage:.4f} > max {args.max_backtest_dd_usage:.4f}"
                )

    client = UpbitClient(access_key=access_key, secret_key=secret_key)
    strategy, params = strategy_params_from_args(args)
    notify(
        "업비트 자동매매 시작",
        [
            f"mode={'LIVE' if args.live else 'DRY-RUN'}",
            f"execution={args.execution}",
            f"market={args.market}",
            f"strategy={strategy}",
            f"universe_top={args.universe_top}",
            f"max_positions={args.max_positions}",
        ],
        color=0x4B6BB5,
    )

    print(f"[1/6] market list ({args.market})")
    markets = fetch_markets(client, args.market)
    if not markets:
        raise SystemExit("no markets found")

    print(f"[2/6] volume ranking top={args.universe_top}")
    universe = select_universe_by_volume(client, markets, args.universe_top)
    if not universe:
        raise SystemExit("no market in universe")
    print(f"  universe size={len(universe)}")

    print(f"[3/6] compute signals strategy={strategy} params={params}")
    signals = compute_signals(
        client,
        universe,
        strategy,
        params,
        candle_count=args.candle_count,
        min_data_days=args.min_data_days,
        signal_offset=args.signal_offset,
        pause_sec=max(0.0, args.request_pause),
    )
    if not signals:
        raise SystemExit("no valid signals computed")

    targets = rank_candidates(signals, args.max_positions)
    if not targets:
        print("  no entry signals -> no buy target")

    if args.show_top > 0:
        print("[signal-top]")
        top_list = sorted(
            [s for s in signals.values() if s.score is not None],
            key=lambda s: float(s.score),
            reverse=True,
        )
        for item in top_list[: args.show_top]:
            print(
                f"  {item.market} score={float(item.score):.6f} "
                f"enter={int(item.enter)} exit={int(item.exit)} "
                f"date={item.signal_date} price={item.last_price:.4f}"
            )

    if args.execution == "webhook":
        webhook_market = args.webhook_market.upper()
        snap = signals.get(webhook_market)
        if snap is None:
            dates, closes, volumes = fetch_daily_candles(client, webhook_market, args.candle_count)
            if len(closes) >= max(10, args.min_data_days):
                strategy_fn = engine.make_strategy(strategy)
                enters, exits, scores = strategy_fn(closes, volumes, params)
                signal_idx = len(closes) - 1 - args.signal_offset
                if 1 <= signal_idx < len(closes):
                    snap = SignalSnapshot(
                        market=webhook_market,
                        signal_date=dates[signal_idx],
                        last_price=closes[-1],
                        enter=bool(enters[signal_idx]),
                        exit=bool(exits[signal_idx]),
                        score=scores[signal_idx],
                        rows=len(closes),
                    )

        action = decide_webhook_action(snap)
        signal_date = snap.signal_date if snap else ""
        state_key = f"{webhook_market}|{signal_date}|{action}"

        print("[4/6] webhook decision")
        if snap is None:
            print(f"  market={webhook_market} signal unavailable -> hold")
            action = "hold"
        else:
            score_text = f"{float(snap.score):.6f}" if snap.score is not None else "-"
            print(
                f"  market={webhook_market} date={snap.signal_date} "
                f"enter={int(snap.enter)} exit={int(snap.exit)} score={score_text}"
            )
            print(f"  action={action}")

        if action not in ("buy", "sell"):
            notify(
                "업비트 웹훅 실행 종료",
                [
                    f"mode={'LIVE' if args.live else 'DRY-RUN'}",
                    f"market={webhook_market}",
                    "action=hold",
                ],
                color=0x26A69A,
            )
            print("done")
            return 0

        state = load_json_file(args.webhook_state_file)
        last_sent_key = str(state.get("last_sent_key", ""))
        if (not args.force_send) and state_key == last_sent_key:
            print(f"  duplicate signal skipped: {state_key}")
            notify(
                "업비트 웹훅 실행 건너뜀",
                [
                    f"market={webhook_market}",
                    f"action={action}",
                    "reason=duplicate_signal",
                ],
                color=0xF3C969,
            )
            print("done")
            return 0

        base = webhook_market.split("-", 1)[1] if "-" in webhook_market else webhook_market
        quote = webhook_market.split("-", 1)[0] if "-" in webhook_market else args.market.upper()
        payload = {
            "password": order_webhook_password,
            "exchange": args.webhook_exchange,
            "base": base,
            "quote": quote,
            "side": action,
            "type": "market",
            "amount": "NaN",
            "percent": format_number_text(args.webhook_buy_percent if action == "buy" else args.webhook_sell_percent),
            "order_name": "업비트 풀매수" if action == "buy" else "업비트 풀매도",
        }
        masked_payload = dict(payload)
        masked_payload["password"] = "***"

        print("[5/6] webhook payload")
        print(f"  url={order_webhook_url}")
        print(f"  payload={json.dumps(masked_payload, ensure_ascii=False)}")

        print("[6/6] webhook send")
        if args.live:
            ok, detail = send_json_webhook(order_webhook_url, payload, timeout=max(1.0, args.notify_timeout))
            if not ok:
                notify(
                    "업비트 웹훅 전송 실패",
                    [
                        f"market={webhook_market}",
                        f"action={action}",
                        f"error={detail}",
                    ],
                    color=0xEF5350,
                )
                raise SystemExit(f"webhook send failed: {detail}")
            save_json_file(
                args.webhook_state_file,
                {
                    "market": webhook_market,
                    "last_signal_date": signal_date,
                    "last_action": action,
                    "last_sent_key": state_key,
                    "updated_at": int(time.time()),
                },
            )
            print("  webhook sent")
        else:
            print("  dry-run mode: webhook not sent")

        notify(
            "업비트 웹훅 실행 완료",
            [
                f"mode={'LIVE' if args.live else 'DRY-RUN'}",
                f"market={webhook_market}",
                f"action={action}",
                f"signal_date={signal_date or '-'}",
            ],
            color=0x26A69A,
        )
        print("done")
        return 0

    print("[4/6] account state")
    try:
        krw_balance, holdings = fetch_account_state(client, args.market)
        print(f"  KRW={krw_balance:.0f} holdings={len(holdings)}")
    except Exception as exc:
        if args.live:
            notify(
                "업비트 자동매매 실패",
                [
                    "stage=account_state",
                    f"mode={'LIVE' if args.live else 'DRY-RUN'}",
                    f"error={exc}",
                ],
                color=0xEF5350,
            )
            raise
        print(f"  account fetch failed (dry-run fallback): {exc}")
        krw_balance, holdings = 0.0, {}

    watch_markets = sorted(set(list(holdings.keys()) + targets))
    ticker_map = fetch_ticker_map(client, watch_markets) if watch_markets else {}

    raw_sells = decide_sells(
        holdings,
        targets,
        signals,
        ticker_map,
        stop_loss=(args.stop_loss if args.stop_loss > 0 else None),
        take_profit=(args.take_profit if args.take_profit > 0 else None),
    )
    sells: list[tuple[str, str]] = []
    for market, reason in raw_sells:
        if reason in ("not_in_target", "exit_signal") and engine_db and args.min_holding_days > 0:
            last_buy_date = db_fetch_last_buy_date_market(engine_db, market)
            if last_buy_date:
                age = days_since_text(run_date, last_buy_date)
                if age is not None and age < args.min_holding_days:
                    print(f"  HOLD {market}: min-holding lock ({age}d/{args.min_holding_days}d)")
                    continue
        sells.append((market, reason))

    keep_after_sell = set(holdings.keys()) - {m for m, _ in sells}
    buy_candidates = [m for m in targets if m not in keep_after_sell]

    if engine_db and args.entry_cooldown_days > 0 and buy_candidates:
        cooled: list[str] = []
        for market in buy_candidates:
            last_exit_date = db_fetch_last_exit_date_market(engine_db, market)
            if last_exit_date:
                gap = days_since_text(run_date, last_exit_date)
                if gap is not None and gap < args.entry_cooldown_days:
                    print(f"  SKIP BUY {market}: entry-cooldown ({gap}d/{args.entry_cooldown_days}d)")
                    continue
            cooled.append(market)
        buy_candidates = cooled

    print("[5/6] execution plan")
    print(f"  targets={targets}")
    print(f"  sells={len(sells)} buys={len(buy_candidates)} mode={'LIVE' if args.live else 'DRY-RUN'}")
    for market, reason in sells:
        qty = holdings[market].qty
        est_price = to_float(ticker_map.get(market, {}).get("trade_price"), default=0.0)
        est_notional = qty * est_price
        print(f"  SELL {market} qty={qty:.10f} est={est_notional:.0f} reason={reason}")
    for market in buy_candidates:
        snap = signals.get(market)
        sc = float(snap.score) if snap and snap.score is not None else float("nan")
        print(f"  BUY  {market} score={sc:.6f}")
    notify(
        "업비트 자동매매 계획",
        [
            f"mode={'LIVE' if args.live else 'DRY-RUN'}",
            f"targets={','.join(targets) if targets else '-'}",
            f"sell_count={len(sells)}",
            f"buy_count={len(buy_candidates)}",
            f"krw_balance={krw_balance:.0f}",
        ],
        color=0xF3C969,
    )

    print("[6/6] apply")
    simulated_krw = krw_balance
    executed_sells = 0
    executed_buys = 0
    trade_limit_left: int | None = None
    if args.live and engine_db:
        caps: list[int] = []
        if args.max_trades_per_day > 0:
            used_today = db_count_trades_on_date(engine_db, run_date)
            day_left = max(0, int(args.max_trades_per_day) - used_today)
            caps.append(day_left)
            print(f"  live trade/day cap: used_today={used_today} left={day_left} limit={int(args.max_trades_per_day)}")
        if args.max_trades_per_year > 0:
            year_limit = int(math.floor(args.max_trades_per_year))
            used_year = db_count_trades_since_ts(engine_db, run_ts - (365 * 24 * 60 * 60))
            year_left = max(0, year_limit - used_year)
            caps.append(year_left)
            print(f"  live trade/year cap: used_365d={used_year} left={year_left} limit={year_limit}")
        if caps:
            trade_limit_left = min(caps)

    for market, reason in sells:
        holding = holdings.get(market)
        if not holding:
            continue
        price = to_float(ticker_map.get(market, {}).get("trade_price"), default=0.0)
        est_notional = holding.qty * price
        if est_notional < args.min_order_krw:
            print(f"  SKIP SELL {market}: below min order ({est_notional:.0f} KRW)")
            continue
        if args.live:
            if trade_limit_left is not None and trade_limit_left <= 0:
                print("  trade cap reached: stopping further live orders")
                break
            resp = place_market_sell(client, market, holding.qty)
            print(f"  LIVE SELL {market} uuid={resp.get('uuid', '')}")
            executed_sells += 1
            if trade_limit_left is not None:
                trade_limit_left -= 1
            if engine_db:
                db_insert_order_log(
                    engine_db,
                    run_ts=run_ts,
                    run_date=run_date,
                    mode="LIVE",
                    market=market,
                    side="sell",
                    reason=reason,
                    qty=holding.qty,
                    price=price,
                    notional=est_notional,
                    order_uuid=str(resp.get("uuid", "")),
                )
        else:
            print(f"  DRY SELL  {market} qty={holding.qty:.10f} est={est_notional:.0f}")
            simulated_krw += est_notional
            executed_sells += 1

    if args.live:
        # Refresh balances after sell fills.
        time.sleep(0.3)
        krw_balance, holdings = fetch_account_state(client, args.market)
        simulated_krw = krw_balance
        keep_after_sell = set(holdings.keys())
        buy_candidates = [m for m in targets if m not in keep_after_sell]
        if engine_db and args.entry_cooldown_days > 0 and buy_candidates:
            cooled_after_sell: list[str] = []
            for market in buy_candidates:
                last_exit_date = db_fetch_last_exit_date_market(engine_db, market)
                if last_exit_date:
                    gap = days_since_text(run_date, last_exit_date)
                    if gap is not None and gap < args.entry_cooldown_days:
                        continue
                cooled_after_sell.append(market)
            buy_candidates = cooled_after_sell

    allocatable = simulated_krw - args.cash_buffer_krw
    if allocatable <= 0:
        print("  no allocatable KRW after buffer")
        notify(
            "업비트 자동매매 종료",
            [
                f"mode={'LIVE' if args.live else 'DRY-RUN'}",
                f"executed_sells={executed_sells}",
                f"executed_buys={executed_buys}",
                "result=no_allocatable_cash",
            ],
            color=0x26A69A,
        )
        return 0

    remaining = list(buy_candidates)
    for idx, market in enumerate(remaining):
        if args.live and trade_limit_left is not None and trade_limit_left <= 0:
            print("  trade cap reached: skipping remaining buy orders")
            break
        slots_left = len(remaining) - idx
        if slots_left <= 0:
            break
        budget = allocatable / slots_left
        if budget < args.min_order_krw:
            print(f"  SKIP BUY {market}: budget below min order ({budget:.0f} KRW)")
            continue
        budget = float(int(math.floor(budget)))
        if budget < args.min_order_krw:
            print(f"  SKIP BUY {market}: budget below min order after rounding ({budget:.0f} KRW)")
            continue
        if args.live:
            resp = place_market_buy(client, market, budget)
            print(f"  LIVE BUY  {market} krw={budget:.0f} uuid={resp.get('uuid', '')}")
            executed_buys += 1
            if trade_limit_left is not None:
                trade_limit_left -= 1
            if engine_db:
                est_price = to_float(ticker_map.get(market, {}).get("trade_price"), default=0.0)
                qty = (budget / est_price) if est_price > 0 else 0.0
                db_insert_order_log(
                    engine_db,
                    run_ts=run_ts,
                    run_date=run_date,
                    mode="LIVE",
                    market=market,
                    side="buy",
                    reason="target_entry",
                    qty=qty,
                    price=est_price,
                    notional=budget,
                    order_uuid=str(resp.get("uuid", "")),
                )
        else:
            print(f"  DRY BUY   {market} krw={budget:.0f}")
            executed_buys += 1
        allocatable -= budget

    notify(
        "업비트 자동매매 종료",
        [
            f"mode={'LIVE' if args.live else 'DRY-RUN'}",
            f"executed_sells={executed_sells}",
            f"executed_buys={executed_buys}",
            f"remaining_allocatable={allocatable:.0f}",
        ],
        color=0x26A69A,
    )
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
