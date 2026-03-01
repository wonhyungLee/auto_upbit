#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, List

import csv

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import upbit_strategy_engine as engine


def parse_list(raw: str, cast=float) -> list:
    if not raw:
        return []
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def _to_ms_float(v: str | float | int | None) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def load_summary_rows(summary_path: str) -> list[dict]:
    d = json.load(open(summary_path))
    return d["rows"]


def load_allowed_markets(summary_rows: list[dict], min_rows: int) -> set[str]:
    allowed = set()
    for row in summary_rows:
        if row["upbit_rows"] >= min_rows and row["okx_rows"] >= min_rows:
            allowed.add(row["upbit_market"])
    return allowed


def build_returns(prices: List[float]) -> list[float]:
    returns = []
    for i in range(1, len(prices)):
        prev = prices[i - 1]
        cur = prices[i]
        if prev == 0:
            returns.append(0.0)
        else:
            returns.append(cur / prev - 1.0)
    return returns


def corr_pearson(x: List[float], y: List[float]) -> float | None:
    n = len(x)
    if n < 10:
        return None
    mean_x = mean(x)
    mean_y = mean(y)
    var_x = sum((v - mean_x) ** 2 for v in x)
    var_y = sum((v - mean_y) ** 2 for v in y)
    if var_x <= 0 or var_y <= 0:
        return None
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    return cov / (var_x * var_y) ** 0.5


def pick_lag_rules(
    series_by_market: Dict[str, engine.MarketSeries],
    btc_market: str,
    max_lag: int,
) -> dict[str, dict]:
    if btc_market not in series_by_market:
        raise RuntimeError(f"{btc_market} not found in data_dir")

    btc = series_by_market[btc_market]
    btc_returns = build_returns(btc.closes)
    btc_dates = btc.dates[1:]
    btc_idx = {d: i for i, d in enumerate(btc_dates)}

    out: dict[str, dict] = {}
    for market, ms in series_by_market.items():
        if market == btc_market:
            continue
        coin_returns = build_returns(ms.closes)
        coin_dates = ms.dates[1:]
        if len(coin_returns) < 20:
            continue

        best_abs_corr = 0.0
        best = {"lag": 0, "corr": 0.0, "n": 0, "mean_abs": 0.0}

        for lag in range(max_lag + 1):
            x = []
            y = []
            for i, date in enumerate(coin_dates):
                j = btc_idx.get(date)
                if j is None:
                    continue
                if j - lag < 0:
                    continue
                x.append(coin_returns[i])
                y.append(btc_returns[j - lag])
            if len(x) < 20:
                continue
            c = corr_pearson(x, y)
            if c is None:
                continue
            if abs(c) > best_abs_corr:
                best_abs_corr = abs(c)
                best = {
                    "lag": lag,
                    "corr": c,
                    "n": len(x),
                    "mean_abs": mean(abs(v) for v in x),
                }
        if best["n"] > 0:
            out[market] = best
    return out


def parse_targets(summary_path: str, min_rows: int, include_btc: bool = False) -> set[str]:
    rows = load_summary_rows(summary_path)
    markets = load_allowed_markets(rows, min_rows)
    if not include_btc:
        markets.discard("KRW-BTC")
    return markets


def copy_series_map(series_by_market: Dict[str, engine.MarketSeries]) -> Dict[str, engine.MarketSeries]:
    copied = {}
    for market, ms in series_by_market.items():
        copied[market] = engine.MarketSeries(
            market=ms.market,
            dates=list(ms.dates),
            closes=list(ms.closes),
            volumes=list(ms.volumes),
            enters=[False] * len(ms.enters),
            exits=[False] * len(ms.exits),
            scores=[None] * len(ms.scores),
            date_to_index=dict(ms.date_to_index),
        )
    return copied


def build_btc_lag_signals(
    market_series: Dict[str, engine.MarketSeries],
    btc_market: str,
    btc_lag_rules: dict[str, dict],
    lag_mode: str,
    fixed_lag: int,
    lookback: int,
    enter_threshold: float,
    exit_threshold: float,
    min_corr: float,
    require_lag_min: int | None,
    require_lag_max: int | None,
) -> None:
    btc = market_series[btc_market]
    btc_returns = build_returns(btc.closes)
    btc_dates = btc.dates[1:]
    btc_idx = {d: i for i, d in enumerate(btc_dates)}
    btc_close = btc.closes

    for market, ms in market_series.items():
        if market == btc_market:
            ms.enters = [False] * len(ms.dates)
            ms.exits = [False] * len(ms.dates)
            ms.scores = [None] * len(ms.dates)
            continue

        rule = btc_lag_rules.get(market)
        if rule is None:
            ms.enters = [False] * len(ms.dates)
            ms.exits = [False] * len(ms.dates)
            ms.scores = [None] * len(ms.dates)
            continue

        lag = fixed_lag if lag_mode == "global" else rule["lag"]
        if require_lag_min is not None and lag < require_lag_min:
            ms.enters = [False] * len(ms.dates)
            ms.exits = [False] * len(ms.dates)
            ms.scores = [None] * len(ms.dates)
            continue
        if require_lag_max is not None and lag > require_lag_max:
            ms.enters = [False] * len(ms.dates)
            ms.exits = [False] * len(ms.dates)
            ms.scores = [None] * len(ms.dates)
            continue
        corr = rule["corr"]
        if abs(corr) < min_corr:
            ms.enters = [False] * len(ms.dates)
            ms.exits = [False] * len(ms.dates)
            ms.scores = [None] * len(ms.dates)
            continue

        enters = [False] * len(ms.dates)
        exits = [False] * len(ms.dates)
        scores = [None] * len(ms.dates)

        for i, date in enumerate(ms.dates):
            j = btc_idx.get(date)
            if j is None:
                continue
            src = j - lag
            if src < 0:
                continue

            # align to BTC return index
            if src - lookback + 1 < 0:
                continue

            # avoid using future BTC values when lag=0
            src = min(src, len(btc_dates) - 1)
            window = btc_returns[src - lookback + 1 : src + 1]
            if len(window) != lookback:
                continue
            btc_feat = mean(window)
            score = corr * btc_feat
            scores[i] = score
            if score > enter_threshold:
                enters[i] = True
            if score < exit_threshold:
                exits[i] = True

        ms.enters = enters
        ms.exits = exits
        ms.scores = scores


def backtest_candidate(
    base_series: Dict[str, engine.MarketSeries],
    btc_market: str,
    btc_lag_rules: dict[str, dict],
    args: argparse.Namespace,
    lag_mode: str,
    fixed_lag: int,
    lookback: int,
    enter_threshold: float,
    exit_threshold: float,
    min_corr: float,
    require_lag_min: int | None,
    require_lag_max: int | None,
) -> engine.BacktestResult:
    series = copy_series_map(base_series)
    build_btc_lag_signals(
        series,
        btc_market=btc_market,
        btc_lag_rules=btc_lag_rules,
        lag_mode=lag_mode,
        fixed_lag=fixed_lag,
        lookback=lookback,
        enter_threshold=enter_threshold,
        exit_threshold=exit_threshold,
        min_corr=min_corr,
        require_lag_min=require_lag_min,
        require_lag_max=require_lag_max,
    )

    filtered = {
        m: ms
        for m, ms in series.items()
        if m != btc_market and len(ms.closes) >= args.min_data_days
    }

    # extra guard: if no candidates, skip
    if not filtered:
        return None

    return engine.backtest(
        filtered,
        strategy="btc_lag",
        params={
            "lag_mode": lag_mode,
            "fixed_lag": fixed_lag,
            "lookback": lookback,
            "enter_threshold": enter_threshold,
            "exit_threshold": exit_threshold,
            "min_corr": min_corr,
        },
        initial_capital=args.initial_capital,
        max_positions=args.max_positions,
        rebalance_every=args.rebalance_every,
        fee_rate=args.fee_bps / 10_000.0,
        slippage_rate=args.slippage_bps / 10_000.0,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
        max_holding_days=args.max_holding_days,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BTC lag-rule backtest on upbit top overlap set")
    p.add_argument("--summary", type=str, default="/tmp/upbit_okx_overlap_all_v2_summary.json")
    p.add_argument("--data-dir", type=str, default="/tmp/upbit_okx_overlap_all_v2/upbit")
    p.add_argument("--market-prefix", type=str, default="", help="empty: all")
    p.add_argument("--btc-market", type=str, default="KRW-BTC")
    p.add_argument("--min-data-days", type=int, default=120)
    p.add_argument("--lag-modes", type=str, default="global,market")
    p.add_argument("--global-lags", type=str, default="0,1,2,3,4,5,7,10,14")
    p.add_argument("--max-lag-est", type=int, default=30)
    p.add_argument("--lookback-grid", type=str, default="3,5,10")
    p.add_argument("--enter-grid", type=str, default="0.0005,0.001,0.002")
    p.add_argument("--exit-grid", type=str, default="-0.0005,-0.001,-0.002")
    p.add_argument("--min-corr", type=float, default=0.15)
    p.add_argument("--require-lag-min", type=int, default=None)
    p.add_argument("--require-lag-max", type=int, default=None)
    p.add_argument("--initial-capital", type=float, default=1_000_000)
    p.add_argument("--max-positions", type=int, default=3)
    p.add_argument("--rebalance-every", type=int, default=5)
    p.add_argument("--fee-bps", type=float, default=1.0)
    p.add_argument("--slippage-bps", type=float, default=0.0)
    p.add_argument("--stop-loss", type=float, default=0.08)
    p.add_argument("--take-profit", type=float, default=0.20)
    p.add_argument("--max-holding-days", type=int, default=90)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--out-csv", type=str, default="")
    p.add_argument("--out-json", type=str, default="")
    p.add_argument("--out-lag-csv", type=str, default="/tmp/btc_lag_rules.csv")
    return p.parse_args()


def write_json(path: str, result: engine.BacktestResult) -> None:
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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_results(path: str, results: list[engine.BacktestResult]) -> None:
    ranked = sorted(results, key=lambda r: (r.sharpe, r.total_return), reverse=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "rank",
                "lag_mode",
                "fixed_lag",
                "lookback",
                "enter_threshold",
                "exit_threshold",
                "min_corr",
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
            cfg = r.config["params"]
            w.writerow(
                [
                    idx,
                    cfg["lag_mode"],
                    cfg["fixed_lag"],
                    cfg["lookback"],
                    cfg["enter_threshold"],
                    cfg["exit_threshold"],
                    cfg["min_corr"],
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


def main() -> int:
    args = parse_args()
    if args.min_data_days < 1:
        raise SystemExit("--min-data-days must be >= 1")
    if args.initial_capital <= 0:
        raise SystemExit("--initial-capital must be > 0")

    all_markets = parse_targets(args.summary, args.min_data_days, include_btc=True)
    all_markets.add(args.btc_market)

    series = engine.load_market_series(args.data_dir, args.market_prefix)
    series = {m: s for m, s in series.items() if m in all_markets}
    if args.btc_market not in series:
        raise SystemExit(f"BTC market not in data: {args.btc_market}")

    lag_modes = [m.strip() for m in args.lag_modes.split(",") if m.strip()]
    global_lags = parse_list(args.global_lags, int)
    lookbacks = parse_list(args.lookback_grid, int)
    enters = parse_list(args.enter_grid, float)
    exits = parse_list(args.exit_grid, float)

    lag_rules = pick_lag_rules(series, args.btc_market, args.max_lag_est)
    # save lag map
    with open(args.out_lag_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["market", "lag", "corr", "sample_n", "mean_abs_return"])
        for market in sorted(lag_rules):
            r = lag_rules[market]
            w.writerow([market, r["lag"], r["corr"], r["n"], r["mean_abs"]])

    print(f"estimated lag rules: {len(lag_rules)} markets")

    results: list[engine.BacktestResult] = []
    for lag_mode in lag_modes:
        for lag in (global_lags if lag_mode == "global" else [0]):
            for lb in lookbacks:
                for et in enters:
                    for xt in exits:
                        fixed = lag if lag_mode == "global" else 0
                        r = backtest_candidate(
                            series,
                            args.btc_market,
                            lag_rules,
                            args,
                            lag_mode=lag_mode,
                            fixed_lag=fixed,
                            lookback=lb,
                            enter_threshold=et,
                            exit_threshold=xt,
                            min_corr=args.min_corr,
                            require_lag_min=args.require_lag_min,
                            require_lag_max=args.require_lag_max,
                        )
                        if r is None:
                            continue
                        r.config["strategy"] = "btc_lag"
                        # record readable strategy context
                        r.config["params"]["lag_mode"] = lag_mode
                        r.config["params"]["fixed_lag"] = fixed
                        r.config["params"]["lookback"] = lb
                        r.config["params"]["enter_threshold"] = et
                        r.config["params"]["exit_threshold"] = xt
                        r.config["params"]["min_corr"] = args.min_corr
                        r.config["params"]["require_lag_min"] = args.require_lag_min
                        r.config["params"]["require_lag_max"] = args.require_lag_max
                        r.config["params"]["initial_capital"] = args.initial_capital
                        r.config["params"]["max_positions"] = args.max_positions
                        r.config["params"]["rebalance_every"] = args.rebalance_every
                        r.config["params"]["fee_rate"] = args.fee_bps / 10_000.0
                        r.config["params"]["slippage_rate"] = args.slippage_bps / 10_000.0
                        r.config["params"]["stop_loss"] = args.stop_loss
                        r.config["params"]["take_profit"] = args.take_profit
                        r.config["params"]["max_holding_days"] = args.max_holding_days
                        results.append(r)

    if not results:
        raise SystemExit("No valid backtest result. Consider lowering min-corr and thresholds.")

    results = sorted(results, key=lambda r: (r.sharpe, r.total_return), reverse=True)
    print("Top results:")
    for rank, r in enumerate(results[: args.top], start=1):
        cfg = r.config["params"]
        print(
            f"{rank:2d} | mode={cfg['lag_mode']} lag={cfg['fixed_lag']} lb={cfg['lookback']} "
            f"enter={cfg['enter_threshold']} exit={cfg['exit_threshold']} "
            f"ret={r.total_return*100:.2f}% sharpe={r.sharpe:.3f} calmar={r.calmar:.3f} "
            f"maxDD={r.max_drawdown*100:.2f}% trades={r.trade_count} win={r.win_rate*100:.1f}%"
        )

    if args.out_csv:
        write_results(args.out_csv, results)
        print(f"saved result csv: {args.out_csv}")
    if args.out_json:
        write_json(args.out_json, results[0])
        print(f"saved best json: {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
