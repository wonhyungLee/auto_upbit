#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from statistics import mean
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import btc_lag_backtest as lagbt
import upbit_strategy_engine as engine


@dataclass
class CycleModel:
    market: str
    period: int
    coef_sin: float
    coef_cos: float
    corr: float
    sample_n: int


@dataclass
class FoldOutcome:
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    model_count: int
    avg_period: float
    dominant_period: int
    chosen_params: dict
    train_result: engine.BacktestResult
    test_result: engine.BacktestResult


def parse_list(raw: str, cast=float) -> list:
    if not raw:
        return []
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def normalize_date(raw: str) -> str:
    return raw.split("T", 1)[0]


def date_to_ordinal(d: str) -> int:
    return datetime.strptime(d, "%Y-%m-%d").date().toordinal()


def load_overlap_rows(summary_path: str, min_overlap_rows: int) -> dict[str, dict]:
    payload = json.load(open(summary_path, encoding="utf-8"))
    rows = {}
    for row in payload["rows"]:
        if row["upbit_rows"] < min_overlap_rows or row["okx_rows"] < min_overlap_rows:
            continue
        rows[row["upbit_market"]] = {
            "start": normalize_date(row["intersection_start"]),
            "end": normalize_date(row["intersection_end"]),
            "upbit_rows": row["upbit_rows"],
            "okx_rows": row["okx_rows"],
        }
    return rows


def load_ranked_markets(path: str, top_n: int) -> list[str]:
    out = []
    with open(path, encoding="utf-8", newline="") as fp:
        reader = csv.reader(fp)
        for row in reader:
            if len(row) < 3:
                continue
            market = row[2].strip()
            if market:
                out.append(market)
            if top_n > 0 and len(out) >= top_n:
                break
    return out


def clip_series(ms: engine.MarketSeries, start_date: str, end_date: str) -> engine.MarketSeries | None:
    idxs = [i for i, d in enumerate(ms.dates) if start_date <= d <= end_date]
    if not idxs:
        return None
    s = idxs[0]
    e = idxs[-1] + 1
    dates = ms.dates[s:e]
    closes = ms.closes[s:e]
    volumes = ms.volumes[s:e]
    if len(dates) < 2:
        return None
    return engine.MarketSeries(
        market=ms.market,
        dates=list(dates),
        closes=list(closes),
        volumes=list(volumes),
        enters=[False] * len(dates),
        exits=[False] * len(dates),
        scores=[None] * len(dates),
        date_to_index={d: i for i, d in enumerate(dates)},
    )


def build_overlap_series(
    all_series: Dict[str, engine.MarketSeries],
    overlap_rows: dict[str, dict],
    selected_markets: set[str],
) -> Dict[str, engine.MarketSeries]:
    out: Dict[str, engine.MarketSeries] = {}
    for market, info in overlap_rows.items():
        if market not in selected_markets:
            continue
        ms = all_series.get(market)
        if ms is None:
            continue
        clipped = clip_series(ms, info["start"], info["end"])
        if clipped is None:
            continue
        out[market] = clipped
    return out


def slice_universe(series: Dict[str, engine.MarketSeries], start_date: str, end_date: str) -> Dict[str, engine.MarketSeries]:
    out: Dict[str, engine.MarketSeries] = {}
    for market, ms in series.items():
        clipped = clip_series(ms, start_date, end_date)
        if clipped is not None:
            out[market] = clipped
    return out


def pearson_corr(x: List[float], y: List[float]) -> float | None:
    n = len(x)
    if n < 10:
        return None
    mx = sum(x) / n
    my = sum(y) / n
    vx = sum((v - mx) ** 2 for v in x)
    vy = sum((v - my) ** 2 for v in y)
    if vx <= 0 or vy <= 0:
        return None
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return cov / math.sqrt(vx * vy)


def build_return_points(ms: engine.MarketSeries) -> tuple[list[int], list[float]]:
    xs = []
    ys = []
    for i in range(1, len(ms.closes)):
        prev = ms.closes[i - 1]
        cur = ms.closes[i]
        if prev <= 0:
            continue
        xs.append(date_to_ordinal(ms.dates[i]))
        ys.append(cur / prev - 1.0)
    return xs, ys


def fit_cycle_model_for_market(ms: engine.MarketSeries, min_cycle: int, max_cycle: int) -> CycleModel | None:
    x, y = build_return_points(ms)
    if len(y) < max(40, min_cycle * 3):
        return None

    best = None
    for period in range(min_cycle, max_cycle + 1):
        w = 2.0 * math.pi / float(period)
        sin_v = [math.sin(w * xi) for xi in x]
        cos_v = [math.cos(w * xi) for xi in x]

        ss = sum(v * v for v in sin_v)
        cc = sum(v * v for v in cos_v)
        sc = sum(a * b for a, b in zip(sin_v, cos_v))
        sy = sum(a * b for a, b in zip(sin_v, y))
        cy = sum(a * b for a, b in zip(cos_v, y))

        det = ss * cc - sc * sc
        if abs(det) < 1e-12:
            continue

        a = (sy * cc - cy * sc) / det
        b = (cy * ss - sy * sc) / det
        pred = [a * s + b * c for s, c in zip(sin_v, cos_v)]
        corr = pearson_corr(y, pred)
        if corr is None:
            continue

        key = abs(corr)
        if best is None or key > best[0]:
            best = (key, period, a, b, corr, len(y))

    if best is None:
        return None
    _, period, a, b, corr, n = best
    return CycleModel(
        market=ms.market,
        period=period,
        coef_sin=a,
        coef_cos=b,
        corr=corr,
        sample_n=n,
    )


def fit_cycle_models(series: Dict[str, engine.MarketSeries], min_cycle: int, max_cycle: int) -> dict[str, CycleModel]:
    out: dict[str, CycleModel] = {}
    for market, ms in series.items():
        model = fit_cycle_model_for_market(ms, min_cycle=min_cycle, max_cycle=max_cycle)
        if model is not None:
            out[market] = model
    return out


def apply_cycle_signals(
    series: Dict[str, engine.MarketSeries],
    models: dict[str, CycleModel],
    *,
    min_fit_corr: float,
    enter_threshold: float,
    exit_threshold: float,
) -> None:
    for market, ms in series.items():
        model = models.get(market)
        if model is None or abs(model.corr) < min_fit_corr:
            ms.enters = [False] * len(ms.dates)
            ms.exits = [False] * len(ms.dates)
            ms.scores = [None] * len(ms.dates)
            continue

        enters = [False] * len(ms.dates)
        exits = [False] * len(ms.dates)
        scores = [None] * len(ms.dates)
        w = 2.0 * math.pi / float(model.period)
        for i, d in enumerate(ms.dates):
            x = date_to_ordinal(d)
            score = model.coef_sin * math.sin(w * x) + model.coef_cos * math.cos(w * x)
            scores[i] = score
            enters[i] = score > enter_threshold
            exits[i] = score < exit_threshold

        ms.enters = enters
        ms.exits = exits
        ms.scores = scores


def run_cycle_cfg(
    series: Dict[str, engine.MarketSeries],
    models: dict[str, CycleModel],
    cfg: dict,
    *,
    min_market_days: int,
    initial_capital: float,
    max_positions: int,
    rebalance_every: int,
    fee_bps: float,
    slippage_bps: float,
    stop_loss: float,
    take_profit: float,
    max_holding_days: int,
) -> engine.BacktestResult | None:
    copied = lagbt.copy_series_map(series)
    apply_cycle_signals(
        copied,
        models,
        min_fit_corr=cfg["min_fit_corr"],
        enter_threshold=cfg["enter_threshold"],
        exit_threshold=cfg["exit_threshold"],
    )

    filtered = {m: ms for m, ms in copied.items() if len(ms.closes) >= min_market_days}
    if not filtered:
        return None

    return engine.backtest(
        filtered,
        strategy="cycle_period",
        params=cfg,
        initial_capital=initial_capital,
        max_positions=max_positions,
        rebalance_every=rebalance_every,
        fee_rate=fee_bps / 10_000.0,
        slippage_rate=slippage_bps / 10_000.0,
        stop_loss=stop_loss,
        take_profit=take_profit,
        max_holding_days=max_holding_days,
    )


def summarize_models(models: dict[str, CycleModel], min_fit_corr: float) -> tuple[int, float, int]:
    used = [m for m in models.values() if abs(m.corr) >= min_fit_corr]
    if not used:
        return 0, 0.0, 0
    periods = [m.period for m in used]
    dominant = Counter(periods).most_common(1)[0][0]
    return len(used), mean(periods), dominant


def build_cfg_grid(args: argparse.Namespace) -> list[dict]:
    enters = parse_list(args.enter_grid, float)
    exits = parse_list(args.exit_grid, float)
    min_fit_corrs = parse_list(args.min_fit_corr_grid, float)
    out = []
    for et in enters:
        for xt in exits:
            for mc in min_fit_corrs:
                out.append(
                    {
                        "enter_threshold": et,
                        "exit_threshold": xt,
                        "min_fit_corr": mc,
                    }
                )
    return out


def choose_best_cfg(
    series_eval: Dict[str, engine.MarketSeries],
    models_eval: dict[str, CycleModel],
    cfg_grid: list[dict],
    *,
    min_market_days: int,
    min_trades: int,
    initial_capital: float,
    max_positions: int,
    rebalance_every: int,
    fee_bps: float,
    slippage_bps: float,
    stop_loss: float,
    take_profit: float,
    max_holding_days: int,
) -> tuple[dict | None, engine.BacktestResult | None]:
    best_cfg = None
    best_result = None
    best_key = None

    for cfg in cfg_grid:
        result = run_cycle_cfg(
            series_eval,
            models_eval,
            cfg,
            min_market_days=min_market_days,
            initial_capital=initial_capital,
            max_positions=max_positions,
            rebalance_every=rebalance_every,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            stop_loss=stop_loss,
            take_profit=take_profit,
            max_holding_days=max_holding_days,
        )
        if result is None:
            continue
        if result.trade_count < min_trades:
            continue
        key = (result.sharpe, result.total_return, result.calmar)
        if best_key is None or key > best_key:
            best_key = key
            best_cfg = cfg
            best_result = result

    if best_result is not None:
        return best_cfg, best_result

    # fallback without minimum trade count
    for cfg in cfg_grid:
        result = run_cycle_cfg(
            series_eval,
            models_eval,
            cfg,
            min_market_days=min_market_days,
            initial_capital=initial_capital,
            max_positions=max_positions,
            rebalance_every=rebalance_every,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            stop_loss=stop_loss,
            take_profit=take_profit,
            max_holding_days=max_holding_days,
        )
        if result is None:
            continue
        key = (result.sharpe, result.total_return, result.calmar)
        if best_key is None or key > best_key:
            best_key = key
            best_cfg = cfg
            best_result = result
    return best_cfg, best_result


def pick_best_train_result(
    series_train: Dict[str, engine.MarketSeries],
    args: argparse.Namespace,
    cfg_grid: list[dict],
) -> tuple[dict, engine.BacktestResult, dict[str, CycleModel]] | None:
    if args.select_mode == "train":
        models = fit_cycle_models(series_train, args.min_cycle, args.max_cycle)
        if not models:
            return None
        best_cfg, train_result = choose_best_cfg(
            series_train,
            models,
            cfg_grid,
            min_market_days=args.min_market_days,
            min_trades=args.min_train_trades,
            initial_capital=args.initial_capital,
            max_positions=args.max_positions,
            rebalance_every=args.rebalance_every,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            stop_loss=args.stop_loss,
            take_profit=args.take_profit,
            max_holding_days=args.max_holding_days,
        )
        if best_cfg is None or train_result is None:
            return None
        return best_cfg, train_result, models

    # validation split inside train window
    any_market = next(iter(series_train.values()))
    if len(any_market.dates) <= args.val_days + 60:
        models = fit_cycle_models(series_train, args.min_cycle, args.max_cycle)
        if not models:
            return None
        best_cfg, train_result = choose_best_cfg(
            series_train,
            models,
            cfg_grid,
            min_market_days=args.min_market_days,
            min_trades=args.min_train_trades,
            initial_capital=args.initial_capital,
            max_positions=args.max_positions,
            rebalance_every=args.rebalance_every,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            stop_loss=args.stop_loss,
            take_profit=args.take_profit,
            max_holding_days=args.max_holding_days,
        )
        if best_cfg is None or train_result is None:
            return None
        return best_cfg, train_result, models

    train_dates = any_market.dates
    pre_start = train_dates[0]
    pre_end = train_dates[-args.val_days - 1]
    val_start = train_dates[-args.val_days]
    val_end = train_dates[-1]

    series_pre = slice_universe(series_train, pre_start, pre_end)
    series_val = slice_universe(series_train, val_start, val_end)

    models_pre = fit_cycle_models(series_pre, args.min_cycle, args.max_cycle)
    if not models_pre:
        return None

    best_cfg, _ = choose_best_cfg(
        series_val,
        models_pre,
        cfg_grid,
        min_market_days=max(5, min(args.min_market_days, args.val_days // 2)),
        min_trades=args.min_val_trades,
        initial_capital=args.initial_capital,
        max_positions=args.max_positions,
        rebalance_every=args.rebalance_every,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
        max_holding_days=args.max_holding_days,
    )
    if best_cfg is None:
        return None

    models_full = fit_cycle_models(series_train, args.min_cycle, args.max_cycle)
    if not models_full:
        return None

    train_result = run_cycle_cfg(
        series_train,
        models_full,
        best_cfg,
        min_market_days=args.min_market_days,
        initial_capital=args.initial_capital,
        max_positions=args.max_positions,
        rebalance_every=args.rebalance_every,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
        max_holding_days=args.max_holding_days,
    )
    if train_result is None:
        return None

    return best_cfg, train_result, models_full


def write_fold_csv(path: str, rows: list[FoldOutcome]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(
            [
                "fold",
                "train_start",
                "train_end",
                "test_start",
                "test_end",
                "model_count",
                "avg_period",
                "dominant_period",
                "enter_threshold",
                "exit_threshold",
                "min_fit_corr",
                "train_return",
                "train_sharpe",
                "train_calmar",
                "train_max_drawdown",
                "train_trade_count",
                "test_return",
                "test_sharpe",
                "test_calmar",
                "test_max_drawdown",
                "test_trade_count",
            ]
        )
        for item in rows:
            p = item.chosen_params
            w.writerow(
                [
                    item.fold,
                    item.train_start,
                    item.train_end,
                    item.test_start,
                    item.test_end,
                    item.model_count,
                    item.avg_period,
                    item.dominant_period,
                    p["enter_threshold"],
                    p["exit_threshold"],
                    p["min_fit_corr"],
                    item.train_result.total_return,
                    item.train_result.sharpe,
                    item.train_result.calmar,
                    item.train_result.max_drawdown,
                    item.train_result.trade_count,
                    item.test_result.total_return,
                    item.test_result.sharpe,
                    item.test_result.calmar,
                    item.test_result.max_drawdown,
                    item.test_result.trade_count,
                ]
            )


def write_summary_json(path: str, rows: list[FoldOutcome], meta: dict) -> None:
    test_compound = 1.0
    train_compound = 1.0
    for item in rows:
        test_compound *= 1.0 + item.test_result.total_return
        train_compound *= 1.0 + item.train_result.total_return

    payload = {
        "meta": meta,
        "fold_count": len(rows),
        "train_compound_return": train_compound - 1.0,
        "test_compound_return": test_compound - 1.0,
        "avg_test_sharpe": (sum(r.test_result.sharpe for r in rows) / len(rows)) if rows else 0.0,
        "avg_test_calmar": (sum(r.test_result.calmar for r in rows) / len(rows)) if rows else 0.0,
        "avg_test_max_drawdown": (sum(r.test_result.max_drawdown for r in rows) / len(rows)) if rows else 0.0,
        "total_test_trades": sum(r.test_result.trade_count for r in rows),
        "folds": [
            {
                "fold": r.fold,
                "train_start": r.train_start,
                "train_end": r.train_end,
                "test_start": r.test_start,
                "test_end": r.test_end,
                "model_count": r.model_count,
                "avg_period": r.avg_period,
                "dominant_period": r.dominant_period,
                "chosen_params": r.chosen_params,
                "train_return": r.train_result.total_return,
                "train_sharpe": r.train_result.sharpe,
                "train_calmar": r.train_result.calmar,
                "train_max_drawdown": r.train_result.max_drawdown,
                "train_trade_count": r.train_result.trade_count,
                "test_return": r.test_result.total_return,
                "test_sharpe": r.test_result.sharpe,
                "test_calmar": r.test_result.calmar,
                "test_max_drawdown": r.test_result.max_drawdown,
                "test_trade_count": r.test_result.trade_count,
            }
            for r in rows
        ],
    }
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward validation for coin cycle-period strategy")
    p.add_argument("--summary", type=str, default="/tmp/upbit_okx_overlap_all_v2_summary.json")
    p.add_argument("--data-dir", type=str, default="/tmp/upbit_okx_overlap_all_v2/upbit")
    p.add_argument("--rank-csv", type=str, default="/tmp/top50_overlap_by_upbit_volume.csv")
    p.add_argument("--top-n", type=int, default=50)
    p.add_argument("--min-overlap-rows", type=int, default=120)
    p.add_argument("--anchor-market", type=str, default="KRW-BTC")

    p.add_argument("--train-days", type=int, default=540)
    p.add_argument("--test-days", type=int, default=120)
    p.add_argument("--step-days", type=int, default=120)
    p.add_argument("--min-market-days", type=int, default=30)
    p.add_argument("--min-train-trades", type=int, default=20)
    p.add_argument("--select-mode", type=str, default="validation", choices=["train", "validation"])
    p.add_argument("--val-days", type=int, default=120)
    p.add_argument("--min-val-trades", type=int, default=8)

    p.add_argument("--min-cycle", type=int, default=7)
    p.add_argument("--max-cycle", type=int, default=60)
    p.add_argument("--enter-grid", type=str, default="0.0,0.0002,0.0005")
    p.add_argument("--exit-grid", type=str, default="0.0,-0.0002,-0.0005")
    p.add_argument("--min-fit-corr-grid", type=str, default="0.03,0.05,0.08")

    p.add_argument("--initial-capital", type=float, default=1_000_000)
    p.add_argument("--max-positions", type=int, default=3)
    p.add_argument("--rebalance-every", type=int, default=5)
    p.add_argument("--fee-bps", type=float, default=1.0)
    p.add_argument("--slippage-bps", type=float, default=0.0)
    p.add_argument("--stop-loss", type=float, default=0.08)
    p.add_argument("--take-profit", type=float, default=0.20)
    p.add_argument("--max-holding-days", type=int, default=90)

    p.add_argument("--notify-bell", action="store_true")
    p.add_argument("--out-csv", type=str, default="/tmp/cycle_period_walkforward_top50.csv")
    p.add_argument("--out-json", type=str, default="/tmp/cycle_period_walkforward_top50.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.train_days < 60:
        raise SystemExit("--train-days must be >= 60")
    if args.test_days < 30:
        raise SystemExit("--test-days must be >= 30")
    if args.step_days < 1:
        raise SystemExit("--step-days must be >= 1")
    if args.max_positions < 1:
        raise SystemExit("--max-positions must be >= 1")
    if args.min_cycle < 2 or args.max_cycle <= args.min_cycle:
        raise SystemExit("invalid cycle range")

    overlap_rows = load_overlap_rows(args.summary, args.min_overlap_rows)
    if not overlap_rows:
        raise SystemExit("No overlap rows after filtering")

    if args.rank_csv:
        ranked = load_ranked_markets(args.rank_csv, args.top_n)
        selected = set(ranked)
    else:
        selected = set(overlap_rows.keys())
    selected.add(args.anchor_market)

    all_series = engine.load_market_series(args.data_dir, market_prefix="")
    overlap_series = build_overlap_series(all_series, overlap_rows, selected)
    if args.anchor_market not in overlap_series:
        raise SystemExit(f"{args.anchor_market} not available in overlap-clipped series")

    cfg_grid = build_cfg_grid(args)
    if not cfg_grid:
        raise SystemExit("empty config grid")

    anchor_dates = overlap_series[args.anchor_market].dates
    if len(anchor_dates) < args.train_days + args.test_days:
        raise SystemExit("not enough anchor dates for requested train/test windows")

    fold_starts = []
    i = args.train_days
    last = len(anchor_dates) - args.test_days
    while i <= last:
        fold_starts.append(i)
        i += args.step_days

    print(
        f"cycle walk-forward start: folds={len(fold_starts)} cfgs={len(cfg_grid)} "
        f"markets={len(overlap_series)} anchor_dates={len(anchor_dates)}"
    )
    if args.notify_bell:
        print("\a", end="", flush=True)

    outcomes: list[FoldOutcome] = []
    for idx, anchor in enumerate(fold_starts, start=1):
        train_start = anchor_dates[anchor - args.train_days]
        train_end = anchor_dates[anchor - 1]
        test_start = anchor_dates[anchor]
        test_end = anchor_dates[min(len(anchor_dates) - 1, anchor + args.test_days - 1)]

        print(f"[fold {idx}/{len(fold_starts)}] train={train_start}..{train_end} test={test_start}..{test_end}")

        series_train = slice_universe(overlap_series, train_start, train_end)
        series_test = slice_universe(overlap_series, test_start, test_end)
        if args.anchor_market not in series_train or args.anchor_market not in series_test:
            print("  skip: anchor missing in one of fold windows")
            continue

        # strategy trades all eligible markets except anchor market
        series_train = {m: s for m, s in series_train.items() if m != args.anchor_market}
        series_test = {m: s for m, s in series_test.items() if m != args.anchor_market}
        if not series_train or not series_test:
            print("  skip: empty train/test market universe")
            continue

        picked = pick_best_train_result(series_train, args, cfg_grid)
        if picked is None:
            print("  skip: could not choose config")
            continue
        best_cfg, train_result, models = picked

        test_result = run_cycle_cfg(
            series_test,
            models,
            best_cfg,
            min_market_days=max(5, min(args.min_market_days, args.test_days // 2)),
            initial_capital=args.initial_capital,
            max_positions=args.max_positions,
            rebalance_every=args.rebalance_every,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            stop_loss=args.stop_loss,
            take_profit=args.take_profit,
            max_holding_days=args.max_holding_days,
        )
        if test_result is None:
            print("  skip: selected config invalid on test")
            continue

        model_count, avg_period, dominant_period = summarize_models(models, best_cfg["min_fit_corr"])
        outcomes.append(
            FoldOutcome(
                fold=idx,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                model_count=model_count,
                avg_period=avg_period,
                dominant_period=dominant_period,
                chosen_params=best_cfg,
                train_result=train_result,
                test_result=test_result,
            )
        )

        print(
            f"  picked enter={best_cfg['enter_threshold']} exit={best_cfg['exit_threshold']} "
            f"min_fit_corr={best_cfg['min_fit_corr']} models={model_count} avg_period={avg_period:.1f} "
            f"| train ret={train_result.total_return*100:.2f}% sharpe={train_result.sharpe:.3f} "
            f"test ret={test_result.total_return*100:.2f}% sharpe={test_result.sharpe:.3f}"
        )
        if args.notify_bell:
            print("\a", end="", flush=True)

    if not outcomes:
        raise SystemExit("no successful fold outcome")

    write_fold_csv(args.out_csv, outcomes)
    meta = {
        "summary": args.summary,
        "data_dir": args.data_dir,
        "rank_csv": args.rank_csv,
        "top_n": args.top_n,
        "min_overlap_rows": args.min_overlap_rows,
        "anchor_market": args.anchor_market,
        "train_days": args.train_days,
        "test_days": args.test_days,
        "step_days": args.step_days,
        "cfg_count": len(cfg_grid),
        "min_cycle": args.min_cycle,
        "max_cycle": args.max_cycle,
    }
    write_summary_json(args.out_json, outcomes, meta)

    compounded = 1.0
    for r in outcomes:
        compounded *= 1.0 + r.test_result.total_return
    print(
        f"cycle walk-forward done: folds={len(outcomes)} compounded_test_return={(compounded - 1.0)*100:.2f}% "
        f"out_csv={args.out_csv} out_json={args.out_json}"
    )
    if args.notify_bell:
        print("\a", end="", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
