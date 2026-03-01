#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from statistics import mean, pstdev
from typing import Callable, Dict, Iterable, List


StrategyFn = Callable[[List[float], List[float], Dict[str, float]], tuple[list[bool], list[bool], list[float | None]]]


@dataclass
class MarketSeries:
    market: str
    dates: list[str]
    closes: list[float]
    volumes: list[float]
    enters: list[bool]
    exits: list[bool]
    scores: list[float | None]
    date_to_index: dict[str, int]


@dataclass
class Position:
    market: str
    qty: float
    entry_price: float
    entry_day: int
    entry_capital: float


@dataclass
class BacktestResult:
    config: dict
    final_capital: float
    total_return: float
    annualized_return: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    win_rate: float
    trade_count: int
    total_trades: int
    equity_curve: list[float]
    equity_dates: list[str]
    trades: list[dict]


def parse_date(value: str) -> str:
    raw = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(raw)
    if dt.time().tzinfo is None:
        dt = dt.replace(tzinfo=None)
    return dt.date().isoformat()


def to_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_list(values: str, cast=float) -> list:
    parts = [item.strip() for item in values.split(",") if item.strip()]
    return [cast(item) for item in parts]


def ranking_key(result: "BacktestResult", objective: str) -> tuple[float, float]:
    if objective == "return":
        return (result.total_return, result.sharpe)
    return (result.sharpe, result.total_return)


def iter_csv_files(path: str) -> Iterable[str]:
    if not os.path.isdir(path):
        return iter(())
    for entry in os.listdir(path):
        lower = entry.lower()
        if lower.endswith("_day.csv") and lower.endswith(".csv"):
            yield os.path.join(path, entry)


def load_market_series(data_dir: str, market_prefix: str) -> dict[str, MarketSeries]:
    out: dict[str, MarketSeries] = {}
    for file_path in iter_csv_files(data_dir):
        filename = os.path.basename(file_path)
        market = filename[:-8]  # remove _day.csv
        if market_prefix and not market.startswith(f"{market_prefix}-") and market != market_prefix:
            continue

        rows = []
        with open(file_path, encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                date_raw = row.get("candle_date_time_utc") or row.get("candle_date_time_kst")
                date = parse_date(date_raw) if date_raw else None
                if not date:
                    continue
                close = to_float(row.get("trade_price"))
                volume = to_float(row.get("candle_acc_trade_volume")) or to_float(row.get("acc_trade_volume"))
                if close is None:
                    continue
                rows.append((date, close, volume or 0.0))

        if not rows:
            continue

        rows.sort(key=lambda item: item[0])
        deduped: list[tuple[str, float, float]] = []
        seen = set()
        for item in rows:
            if item[0] not in seen:
                seen.add(item[0])
                deduped.append(item)
        if len(deduped) < 15:
            continue

        dates = [r[0] for r in deduped]
        closes = [r[1] for r in deduped]
        volumes = [r[2] for r in deduped]
        date_to_index = {date: i for i, date in enumerate(dates)}
        out[market] = MarketSeries(
            market=market,
            dates=dates,
            closes=closes,
            volumes=volumes,
            enters=[False] * len(deduped),
            exits=[False] * len(deduped),
            scores=[None] * len(deduped),
            date_to_index=date_to_index,
        )
    return out


def fill_signals_ma(closes: list[float], volumes: list[float], params: dict[str, float | int]) -> tuple[list[bool], list[bool], list[float | None]]:
    del volumes  # unused
    short_w = int(params["short_window"])
    long_w = int(params["long_window"])
    n = len(closes)
    if short_w < 1 or long_w <= short_w:
        return [False] * n, [False] * n, [None] * n

    short_sma = [None] * n
    long_sma = [None] * n
    s_sum = 0.0
    l_sum = 0.0
    for i, value in enumerate(closes):
        s_sum += value
        l_sum += value
        if i >= short_w:
            s_sum -= closes[i - short_w]
        if i >= long_w:
            l_sum -= closes[i - long_w]

        if i >= short_w - 1:
            short_sma[i] = s_sum / short_w
        if i >= long_w - 1:
            long_sma[i] = l_sum / long_w

    enters = [False] * n
    exits = [False] * n
    scores = [None] * n
    for i in range(1, n):
        if short_sma[i] is None or long_sma[i] is None or short_sma[i - 1] is None or long_sma[i - 1] is None:
            continue
        scores[i] = ((short_sma[i] - long_sma[i]) / long_sma[i]) * 100.0
        if scores[i] is None:
            continue
        enters[i] = short_sma[i] > long_sma[i] and short_sma[i - 1] <= long_sma[i - 1]
        exits[i] = short_sma[i] < long_sma[i] and short_sma[i - 1] >= long_sma[i - 1]
    return enters, exits, scores


def fill_signals_momentum(closes: list[float], volumes: list[float], params: dict[str, float | int]) -> tuple[list[bool], list[bool], list[float | None]]:
    del volumes
    momentum_window = int(params["momentum_window"])
    vol_window = int(params["vol_window"])
    enter_thr = float(params["enter_threshold"])
    exit_thr = float(params["exit_threshold"])
    n = len(closes)
    if n <= max(momentum_window, vol_window):
        return [False] * n, [False] * n, [None] * n

    returns = [0.0] * n
    for i in range(1, n):
        if closes[i - 1] == 0.0:
            returns[i] = 0.0
        else:
            returns[i] = (closes[i] / closes[i - 1]) - 1.0

    enters = [False] * n
    exits = [False] * n
    scores = [None] * n
    for i in range(1, n):
        if i < momentum_window:
            continue
        momentum = (closes[i] / closes[i - momentum_window]) - 1.0
        vol = 0.0
        if i >= vol_window:
            window = returns[i - vol_window + 1 : i + 1]
            m = mean(window)
            var = sum((x - m) ** 2 for x in window) / len(window)
            vol = math.sqrt(var)
        score = momentum / (vol + 1e-12)
        scores[i] = score
        if momentum >= enter_thr:
            enters[i] = True
        if momentum <= exit_thr:
            exits[i] = True
    return enters, exits, scores


def fill_signals_rsi(closes: list[float], volumes: list[float], params: dict[str, float | int]) -> tuple[list[bool], list[bool], list[float | None]]:
    del volumes
    window = int(params["rsi_window"])
    buy_thr = float(params["buy_threshold"])
    sell_thr = float(params["sell_threshold"])
    n = len(closes)
    if n < window + 2:
        return [False] * n, [False] * n, [None] * n

    gains = [0.0] * n
    losses = [0.0] * n
    for i in range(1, n):
        diff = closes[i] - closes[i - 1]
        if diff > 0:
            gains[i] = diff
        else:
            losses[i] = -diff

    enters = [False] * n
    exits = [False] * n
    scores = [None] * n
    for i in range(1, n):
        if i < window:
            continue
        avg_gain = mean(gains[i - window + 1 : i + 1])
        avg_loss = mean(losses[i - window + 1 : i + 1])
        if avg_loss <= 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
        scores[i] = 100.0 - rsi
        if rsi <= buy_thr:
            enters[i] = True
        if rsi >= sell_thr:
            exits[i] = True
    return enters, exits, scores


def make_strategy(name: str) -> StrategyFn:
    if name == "ma":
        return fill_signals_ma
    if name == "momentum":
        return fill_signals_momentum
    if name == "rsi":
        return fill_signals_rsi
    raise ValueError(f"unsupported strategy: {name}")


def prepare_signals(series_by_market: dict[str, MarketSeries], strategy: str, params: dict[str, float | int]) -> None:
    fn = make_strategy(strategy)
    for ms in series_by_market.values():
        enters, exits, scores = fn(ms.closes, ms.volumes, params)
        ms.enters = enters
        ms.exits = exits
        ms.scores = scores


def backtest(
    series_by_market: dict[str, MarketSeries],
    strategy: str,
    params: dict,
    *,
    initial_capital: float,
    max_positions: int,
    rebalance_every: int,
    fee_rate: float,
    slippage_rate: float,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    max_holding_days: int | None = None,
    entry_cooldown_days: int | None = None,
    min_holding_days: int | None = None,
) -> BacktestResult:
    # Collect union dates across markets
    all_dates_set = set()
    for ms in series_by_market.values():
        all_dates_set.update(ms.dates)
    all_dates = sorted(all_dates_set)
    if len(all_dates) < 30:
        raise RuntimeError("not enough unified date range")

    cash = float(initial_capital)
    positions: dict[str, Position] = {}
    last_exit_day: dict[str, int] = {}
    trades: list[dict] = []
    equity_curve: list[float] = []
    equity_dates: list[str] = []

    for day_idx, date in enumerate(all_dates):
        # mark-to-market and collect today's actionable price index per market
        today_price: dict[str, tuple[int, float, bool, bool, float | None]] = {}
        for market, ms in series_by_market.items():
            idx = ms.date_to_index.get(date)
            if idx is None:
                continue
            today_price[market] = (
                idx,
                ms.closes[idx],
                ms.enters[idx - 1] if idx > 0 else False,
                ms.exits[idx - 1] if idx > 0 else False,
                ms.scores[idx - 1] if idx > 0 else None,
            )

        # 1) forced exits: signal exits, stop loss, take profit, max hold
        for market, pos in list(positions.items()):
            today = today_price.get(market)
            if today is None:
                continue
            idx, close, _, prev_exit, _ = today
            hold_days = day_idx - pos.entry_day

            should_exit = False
            if prev_exit and (min_holding_days is None or hold_days >= min_holding_days):
                should_exit = True
            if not should_exit and stop_loss is not None and close <= pos.entry_price * (1.0 - stop_loss):
                should_exit = True
            elif not should_exit and take_profit is not None and close >= pos.entry_price * (1.0 + take_profit):
                should_exit = True
            elif not should_exit and max_holding_days is not None and hold_days >= max_holding_days:
                should_exit = True

            if should_exit:
                proceeds = pos.qty * close * (1.0 - fee_rate) * (1.0 - slippage_rate)
                cash += proceeds
                pnl = proceeds - pos.entry_capital
                trades.append(
                    {
                        "date": date,
                        "market": market,
                        "side": "sell",
                        "price": close,
                        "qty": pos.qty,
                        "notional": proceeds,
                        "pnl": pnl,
                    }
                )
                del positions[market]
                last_exit_day[market] = day_idx

        # 2) rebalance based on ranked candidates
        if day_idx % max(1, rebalance_every) == 0:
            candidates: list[tuple[float, str]] = []
            for market, (_, close, prev_enter, _, score) in today_price.items():
                if not prev_enter:
                    continue
                if market in positions:
                    continue
                if score is None or close <= 0.0:
                    continue
                candidates.append((float(score), market))

            candidates.sort(reverse=True, key=lambda item: item[0])
            target_markets = set(item[1] for item in candidates[:max_positions])

                # close underperforming positions to keep target exposure
            if max_positions > 0:
                for mkt in list(positions.keys()):
                    if mkt not in target_markets:
                        today = today_price.get(mkt)
                        if today is None:
                            continue
                        idx, close, _, _, _ = today
                        pos = positions[mkt]
                        hold_days = day_idx - pos.entry_day
                        if min_holding_days is not None and hold_days < min_holding_days:
                            continue
                        proceeds = pos.qty * close * (1.0 - fee_rate) * (1.0 - slippage_rate)
                        cash += proceeds
                        pnl = proceeds - pos.entry_capital
                        trades.append(
                            {
                                "date": date,
                                "market": mkt,
                                "side": "sell_rebalance",
                                "price": close,
                                "qty": pos.qty,
                                "notional": proceeds,
                                "pnl": pnl,
                            }
                        )
                        del positions[mkt]
                        last_exit_day[mkt] = day_idx

                # open targets
                for score, market in candidates:
                    if len(positions) >= max_positions:
                        break
                    if market in positions:
                        continue
                    if entry_cooldown_days is not None:
                        prev_exit_day = last_exit_day.get(market)
                        if prev_exit_day is not None and (day_idx - prev_exit_day) < entry_cooldown_days:
                            continue
                    idx, close, _, _, _ = today_price[market]
                    if close <= 0.0:
                        continue
                    alloc_capital = cash / (max_positions - len(positions))
                    if alloc_capital <= 0:
                        break
                    fee = alloc_capital * fee_rate
                    net_capital = alloc_capital - fee
                    if net_capital <= 0:
                        continue
                    qty = (net_capital * (1.0 - slippage_rate)) / close
                    if qty <= 0:
                        continue
                    cash -= alloc_capital
                    positions[market] = Position(
                        market=market,
                        qty=qty,
                        entry_price=close,
                        entry_day=day_idx,
                        entry_capital=alloc_capital,
                    )
                    trades.append(
                        {
                            "date": date,
                            "market": market,
                            "side": "buy",
                            "price": close,
                            "qty": qty,
                            "notional": alloc_capital,
                            "score": score,
                        }
                    )

        # 3) mark-to-market
        value = cash
        for market, pos in positions.items():
            today = today_price.get(market)
            if today is None:
                continue
            _, close, _, _, _ = today
            value += pos.qty * close
        equity_curve.append(value)
        equity_dates.append(date)

    # close everything at the final day
    if all_dates:
        final_date = all_dates[-1]
        for market, pos in list(positions.items()):
            ms = series_by_market[market]
            idx = ms.date_to_index.get(final_date)
            if idx is None:
                idx = len(ms.dates) - 1
            close = ms.closes[idx]
            proceeds = pos.qty * close * (1.0 - fee_rate) * (1.0 - slippage_rate)
            cash += proceeds
            pnl = proceeds - pos.entry_capital
            trades.append(
                {
                    "date": final_date,
                    "market": market,
                    "side": "sell_final",
                    "price": close,
                    "qty": pos.qty,
                    "notional": proceeds,
                    "pnl": pnl,
                }
            )
            del positions[market]
        if equity_curve:
            equity_curve[-1] = cash

    returns = []
    for i in range(1, len(equity_curve)):
        prev = equity_curve[i - 1]
        cur = equity_curve[i]
        if prev > 0:
            returns.append((cur / prev) - 1.0)
        else:
            returns.append(0.0)

    total_ret = (equity_curve[-1] / initial_capital - 1.0) if equity_curve else 0.0
    days = max(1, len(equity_curve))
    annualized = (equity_curve[-1] / initial_capital) ** (365.0 / days) - 1.0 if len(equity_curve) > 1 else 0.0

    if returns:
        mean_ret = mean(returns)
        std_ret = pstdev(returns) if len(returns) > 1 else 0.0
        sharpe = (mean_ret / std_ret) * math.sqrt(252.0) if std_ret > 0 else 0.0
        downside = [r for r in returns if r < 0]
        downside_std = pstdev(downside) if len(downside) > 1 else 0.0
        sortino = (mean_ret / downside_std) * math.sqrt(252.0) if downside_std > 0 else 0.0
    else:
        sharpe = 0.0
        sortino = 0.0

    running_peak = equity_curve[0] if equity_curve else initial_capital
    max_dd = 0.0
    for value in equity_curve:
        if value > running_peak:
            running_peak = value
        if running_peak > 0:
            max_dd = max(max_dd, (running_peak - value) / running_peak)
    calmar = total_ret / abs(max_dd) if max_dd > 0 else 0.0

    sells = [t for t in trades if t["side"].startswith("sell")]
    win_rate = 0.0
    if sells:
        wins = sum(1 for t in sells if t.get("pnl", 0.0) > 0)
        win_rate = wins / len(sells)

    return BacktestResult(
        config={
            "strategy": strategy,
            "params": params,
            "initial_capital": initial_capital,
            "max_positions": max_positions,
            "rebalance_every": rebalance_every,
            "fee_rate": fee_rate,
            "slippage_rate": slippage_rate,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "max_holding_days": max_holding_days,
            "entry_cooldown_days": entry_cooldown_days,
            "min_holding_days": min_holding_days,
        },
        final_capital=equity_curve[-1] if equity_curve else initial_capital,
        total_return=total_ret,
        annualized_return=annualized,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        calmar=calmar,
        win_rate=win_rate,
        trade_count=len(sells),
        total_trades=len(trades),
        equity_curve=equity_curve,
        equity_dates=equity_dates,
        trades=trades,
    )


def summarize_results(results: list[BacktestResult], top_n: int = 10, objective: str = "sharpe") -> None:
    if not results:
        print("No results.")
        return
    ranked = sorted(results, key=lambda r: ranking_key(r, objective), reverse=True)
    print("Top results:")
    for idx, r in enumerate(ranked[:top_n], start=1):
        cfg = r.config
        print(
            f"{idx:2d} | {cfg['strategy']} {cfg['params']} | "
            f"ret={r.total_return*100:.2f}% calmar={r.calmar:.3f} sharpe={r.sharpe:.3f} "
            f"maxDD={r.max_drawdown*100:.2f}% trades={r.trade_count} win={r.win_rate*100:.1f}%"
        )


def build_grid(args: argparse.Namespace, mode: str, strategy: str) -> list[dict]:
    if mode == "single":
        if strategy == "ma":
            return [
                {
                    "strategy": "ma",
                    "short_window": args.ma_short,
                    "long_window": args.ma_long,
                }
            ]
        if strategy == "momentum":
            return [
                {
                    "strategy": "momentum",
                    "momentum_window": args.momentum_window,
                    "vol_window": args.momentum_vol_window,
                    "enter_threshold": args.momentum_enter,
                    "exit_threshold": args.momentum_exit,
                }
            ]
        return [
            {
                "strategy": "rsi",
                "rsi_window": args.rsi_window,
                "buy_threshold": args.rsi_buy,
                "sell_threshold": args.rsi_sell,
            }
        ]

    # optimize mode
    out = []
    if strategy in ("ma", "all"):
        for short in args.ma_short_grid:
            for long in args.ma_long_grid:
                if short >= long:
                    continue
                out.append({"strategy": "ma", "short_window": short, "long_window": long})
    if strategy in ("momentum", "all"):
        for momentum in args.momentum_window_grid:
            for vol in args.momentum_vol_window_grid:
                for enter in args.momentum_enter_grid:
                    for exit in args.momentum_exit_grid:
                        out.append(
                            {
                                "strategy": "momentum",
                                "momentum_window": momentum,
                                "vol_window": vol,
                                "enter_threshold": enter,
                                "exit_threshold": exit,
                            }
                        )
    if strategy in ("rsi", "all"):
        for rsi_w in args.rsi_window_grid:
            for buy in args.rsi_buy_grid:
                for sell in args.rsi_sell_grid:
                    if buy >= sell:
                        continue
                    out.append(
                        {
                            "strategy": "rsi",
                            "rsi_window": rsi_w,
                            "buy_threshold": buy,
                            "sell_threshold": sell,
                        }
                    )
    return out


def strategy_params(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if k != "strategy"}


def write_csv_summary(path: str, results: list[BacktestResult], objective: str = "sharpe") -> None:
    ranked = sorted(results, key=lambda r: ranking_key(r, objective), reverse=True)
    with open(path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "rank",
                "strategy",
                "params",
                "total_return",
                "annualized_return",
                "sharpe",
                "sortino",
                "max_drawdown",
                "calmar",
                "win_rate",
                "trade_count",
                "final_capital",
            ]
        )
        for idx, r in enumerate(ranked, start=1):
            writer.writerow(
                [
                    idx,
                    r.config["strategy"],
                    json.dumps(r.config["params"], ensure_ascii=False),
                    r.total_return,
                    r.annualized_return,
                    r.sharpe,
                    r.sortino,
                    r.max_drawdown,
                    r.calmar,
                    r.win_rate,
                    r.trade_count,
                    r.final_capital,
                ]
            )


def write_json_result(path: str, result: BacktestResult) -> None:
    payload = {
        "config": result.config,
        "final_capital": result.final_capital,
        "total_return": result.total_return,
        "annualized_return": result.annualized_return,
        "sharpe": result.sharpe,
        "sortino": result.sortino,
        "max_drawdown": result.max_drawdown,
        "calmar": result.calmar,
        "win_rate": result.win_rate,
        "trade_count": result.trade_count,
        "total_trades": result.total_trades,
        "equity_dates": result.equity_dates,
        "equity_curve": result.equity_curve,
        "trades": result.trades,
    }
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def main() -> int:
    p = argparse.ArgumentParser(description="Backtest coin selection and simple auto-trading rules.")
    p.add_argument("--data-dir", type=str, default="/tmp/upbit_hist", help="Directory with *_day.csv files")
    p.add_argument("--market", type=str, default="KRW", help="Market prefix filter, e.g. KRW")
    p.add_argument("--strategy", type=str, default="ma", choices=["ma", "momentum", "rsi", "all"])
    p.add_argument("--initial-capital", type=float, default=1_000_000, help="Starting KRW capital")
    p.add_argument("--max-positions", type=int, default=3, help="Maximum concurrent positions")
    p.add_argument("--rebalance-every", type=int, default=5, help="Rebalance frequency in days (1 = every day)")
    p.add_argument("--fee-bps", type=float, default=1.0, help="Fee per transaction in basis points")
    p.add_argument("--slippage-bps", type=float, default=0.0, help="Slippage in basis points")
    p.add_argument("--stop-loss", type=float, default=0.08, help="Optional stop loss ratio (e.g. 0.08 = 8%%)")
    p.add_argument("--take-profit", type=float, default=0.20, help="Optional take profit ratio (e.g. 0.20 = 20%%)")
    p.add_argument("--max-holding-days", type=int, default=90, help="Maximum holding period in days")
    p.add_argument("--entry-cooldown-days", type=int, default=0, help="Cooldown days after exit before re-entry")
    p.add_argument("--min-holding-days", type=int, default=0, help="Minimum holding days before signal/rebalance exits")
    p.add_argument("--min-data-days", type=int, default=120, help="Ignore coins with fewer rows than this")
    p.add_argument("--optimize", action="store_true", help="Run parameter grid search instead of single config")
    p.add_argument("--top", type=int, default=10, help="Show top N configs after optimization")
    p.add_argument("--out-csv", type=str, default="", help="Optional path to save optimization summary CSV")
    p.add_argument("--out-json", type=str, default="", help="Optional path to save best run detail JSON")
    p.add_argument("--objective", type=str, default="sharpe", choices=["sharpe", "return"], help="Optimization/ranking objective")

    p.add_argument("--ma-short", type=int, default=5, help="MA short window")
    p.add_argument("--ma-long", type=int, default=20, help="MA long window")
    p.add_argument("--ma-short-grid", type=str, default="5,8,13", help="MA short window candidates for optimization")
    p.add_argument("--ma-long-grid", type=str, default="20,40,60", help="MA long window candidates for optimization")

    p.add_argument("--momentum-window", type=int, default=10, help="Momentum lookback")
    p.add_argument("--momentum-vol-window", type=int, default=20, help="Volatility window used in momentum score")
    p.add_argument("--momentum-enter", type=float, default=0.02, help="Momentum enter threshold")
    p.add_argument("--momentum-exit", type=float, default=-0.03, help="Momentum exit threshold")
    p.add_argument("--momentum-window-grid", type=str, default="5,10,20", help="Momentum lookback candidates for optimization")
    p.add_argument("--momentum-vol-window-grid", type=str, default="10,20,30", help="Vol-window candidates for optimization")
    p.add_argument("--momentum-enter-grid", type=str, default="0.01,0.02,0.03", help="Momentum enter threshold candidates for optimization")
    p.add_argument("--momentum-exit-grid", type=str, default="-0.01,-0.02,-0.03", help="Momentum exit threshold candidates for optimization")

    p.add_argument("--rsi-window", type=int, default=14, help="RSI window")
    p.add_argument("--rsi-buy", type=float, default=30, help="RSI buy threshold")
    p.add_argument("--rsi-sell", type=float, default=60, help="RSI sell threshold")
    p.add_argument("--rsi-window-grid", type=str, default="10,14,20", help="RSI window candidates for optimization")
    p.add_argument("--rsi-buy-grid", type=str, default="20,25,30", help="RSI buy threshold candidates for optimization")
    p.add_argument("--rsi-sell-grid", type=str, default="55,60,65", help="RSI sell threshold candidates for optimization")

    args = p.parse_args()

    if args.max_positions < 1:
        raise SystemExit("Error: --max-positions must be >= 1")
    if args.rebalance_every < 1:
        raise SystemExit("Error: --rebalance-every must be >= 1")

    args.ma_short_grid = parse_list(args.ma_short_grid, float)
    args.ma_long_grid = parse_list(args.ma_long_grid, float)
    args.momentum_window_grid = parse_list(args.momentum_window_grid, float)
    args.momentum_vol_window_grid = parse_list(args.momentum_vol_window_grid, float)
    args.momentum_enter_grid = parse_list(args.momentum_enter_grid, float)
    args.momentum_exit_grid = parse_list(args.momentum_exit_grid, float)
    args.rsi_window_grid = parse_list(args.rsi_window_grid, float)
    args.rsi_buy_grid = parse_list(args.rsi_buy_grid, float)
    args.rsi_sell_grid = parse_list(args.rsi_sell_grid, float)

    series_by_market = load_market_series(args.data_dir, args.market)
    if not series_by_market:
        raise SystemExit(f"No history files found in {args.data_dir}.")

    # basic filtering for minimum available samples
    if args.min_data_days > 0:
        filtered = {
            market: ms
            for market, ms in series_by_market.items()
            if len(ms.dates) >= args.min_data_days
        }
        if not filtered:
            raise SystemExit("No market passed min-data-days filter.")
        series_by_market = filtered

    fee_rate = args.fee_bps / 10_000.0
    slip_rate = args.slippage_bps / 10_000.0
    stop_loss = args.stop_loss if args.stop_loss > 0 else None
    take_profit = args.take_profit if args.take_profit > 0 else None
    max_holding_days = args.max_holding_days if args.max_holding_days > 0 else None
    entry_cooldown_days = args.entry_cooldown_days if args.entry_cooldown_days > 0 else None
    min_holding_days = args.min_holding_days if args.min_holding_days > 0 else None

    # prepare mode
    if args.optimize:
        strategy_list = args.strategy
        cfgs = build_grid(args, "opt", strategy_list)
        print(f"Optimization start: {len(cfgs)} configs, {len(series_by_market)} markets")
        results: list[BacktestResult] = []
        for idx, cfg in enumerate(cfgs, start=1):
            strategy = cfg["strategy"]
            params = strategy_params(cfg)
            prepare_signals(series_by_market, strategy, params)
            result = backtest(
                series_by_market,
                strategy,
                params,
                initial_capital=args.initial_capital,
                max_positions=args.max_positions,
                rebalance_every=args.rebalance_every,
                fee_rate=fee_rate,
                slippage_rate=slip_rate,
                stop_loss=stop_loss,
                take_profit=take_profit,
                max_holding_days=max_holding_days,
                entry_cooldown_days=entry_cooldown_days,
                min_holding_days=min_holding_days,
            )
            results.append(result)
            if idx % 25 == 0:
                print(f"  tested {idx}/{len(cfgs)}")
        ranked = sorted(results, key=lambda r: ranking_key(r, args.objective), reverse=True)
        summarize_results(ranked, top_n=args.top, objective=args.objective)
        if args.out_csv:
            write_csv_summary(args.out_csv, ranked, objective=args.objective)
            print(f"Saved CSV summary: {args.out_csv}")
        if args.out_json:
            best = ranked[0]
            write_json_result(args.out_json, best)
            print(f"Saved best run detail: {args.out_json}")
        return 0

    strategy = args.strategy
    cfgs = build_grid(args, "single", strategy)
    cfg = cfgs[0]
    strategy_name = cfg["strategy"]
    params = strategy_params(cfg)
    prepare_signals(series_by_market, strategy_name, params)
    result = backtest(
        series_by_market,
        strategy_name,
        params,
        initial_capital=args.initial_capital,
        max_positions=args.max_positions,
        rebalance_every=args.rebalance_every,
        fee_rate=fee_rate,
        slippage_rate=slip_rate,
        stop_loss=stop_loss,
        take_profit=take_profit,
        max_holding_days=max_holding_days,
        entry_cooldown_days=entry_cooldown_days,
        min_holding_days=min_holding_days,
    )

    print(
        f"Result [{strategy_name}] return={result.total_return*100:.2f}% "
        f"annualized={result.annualized_return*100:.2f}% Sharpe={result.sharpe:.3f} "
        f"Sortino={result.sortino:.3f} MaxDD={result.max_drawdown*100:.2f}% "
        f"Calmar={result.calmar:.3f} Trades={result.trade_count}"
    )
    if args.out_json:
        write_json_result(args.out_json, result)
        print(f"Saved result detail: {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
