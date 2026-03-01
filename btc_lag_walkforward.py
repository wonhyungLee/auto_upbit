#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import btc_lag_backtest as lagbt
import upbit_strategy_engine as engine


@dataclass
class FoldOutcome:
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_markets: int
    test_markets: int
    lag_rule_count: int
    chosen_params: dict
    train_result: engine.BacktestResult
    test_result: engine.BacktestResult


@dataclass
class CycleModel:
    market: str
    period: int
    coef_sin: float
    coef_cos: float
    corr: float
    sample_n: int


def parse_list(raw: str, cast=float) -> list:
    if not raw:
        return []
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def parse_bool_list(raw: str) -> list[bool]:
    if not raw:
        return []
    out = []
    for token in raw.split(","):
        t = token.strip().lower()
        if not t:
            continue
        if t in ("1", "true", "t", "yes", "y", "on"):
            out.append(True)
        elif t in ("0", "false", "f", "no", "n", "off"):
            out.append(False)
        else:
            raise ValueError(f"invalid bool token: {token}")
    return out


def selection_key(result: engine.BacktestResult, objective: str) -> tuple[float, float, float]:
    if objective == "return":
        return (result.total_return, result.calmar, result.sharpe)
    return (result.sharpe, result.total_return, result.calmar)


def annualized_trade_count(result: engine.BacktestResult) -> float:
    days = max(1, len(result.equity_curve))
    return result.trade_count * (365.0 / days)


def top_markets_by_trade_count(trades: list[dict], *, side_prefix: str, top_n: int = 10) -> list[dict]:
    counts: dict[str, int] = {}
    for t in trades:
        side = str(t.get("side", ""))
        if side_prefix and not side.startswith(side_prefix):
            continue
        market = t.get("market")
        if not market:
            continue
        counts[str(market)] = counts.get(str(market), 0) + 1
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return [{"market": m, "count": c} for m, c in ranked[:top_n]]


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


def rolling_sma(values: List[float], window: int) -> list[float | None]:
    n = len(values)
    out: list[float | None] = [None] * n
    if window <= 0:
        return out
    acc = 0.0
    for i, v in enumerate(values):
        acc += v
        if i >= window:
            acc -= values[i - window]
        if i >= window - 1:
            out[i] = acc / window
    return out


def rolling_ret_std(closes: List[float], window: int) -> list[float | None]:
    n = len(closes)
    out: list[float | None] = [None] * n
    if window <= 1 or n < 2:
        return out
    rets = [0.0] * n
    for i in range(1, n):
        prev = closes[i - 1]
        if prev <= 0:
            rets[i] = 0.0
        else:
            rets[i] = closes[i] / prev - 1.0
    for i in range(window, n):
        seg = rets[i - window + 1 : i + 1]
        mu = sum(seg) / len(seg)
        var = sum((x - mu) ** 2 for x in seg) / len(seg)
        out[i] = math.sqrt(var)
    return out


def apply_btc_regime_filter(series_map: Dict[str, engine.MarketSeries], btc_market: str, cfg: dict) -> None:
    mode = cfg.get("regime_mode", "off")
    if mode == "off":
        return

    btc = series_map[btc_market]
    btc_idx = btc.date_to_index
    regime_short = int(cfg.get("regime_short", 0) or 0)
    regime_long = int(cfg.get("regime_long", 0) or 0)
    regime_vol_window = int(cfg.get("regime_vol_window", 0) or 0)
    regime_vol_max = cfg.get("regime_vol_max", None)
    regime_force_exit = bool(cfg.get("regime_force_exit", False))

    sma_short = rolling_sma(btc.closes, regime_short) if regime_short > 0 else [None] * len(btc.closes)
    sma_long = rolling_sma(btc.closes, regime_long) if regime_long > 0 else [None] * len(btc.closes)
    vol_std = rolling_ret_std(btc.closes, regime_vol_window) if regime_vol_window > 0 else [None] * len(btc.closes)

    for market, ms in series_map.items():
        if market == btc_market:
            continue
        for i, d in enumerate(ms.dates):
            j = btc_idx.get(d)
            if j is None:
                continue
            allow = True
            if mode in ("ma", "ma_vol"):
                s = sma_short[j]
                l = sma_long[j]
                allow = bool(s is not None and l is not None and s > l)
            if allow and mode in ("vol", "ma_vol"):
                v = vol_std[j]
                allow = bool(v is not None and regime_vol_max is not None and v <= float(regime_vol_max))

            if not allow:
                ms.enters[i] = False
                if regime_force_exit:
                    ms.exits[i] = True


def apply_cycle_overlay(series_map: Dict[str, engine.MarketSeries], cycle_models: dict[str, CycleModel] | None, cfg: dict) -> None:
    mode = cfg.get("cycle_mode", "off")
    if mode == "off" or not cycle_models:
        return
    min_cycle_corr = float(cfg.get("cycle_min_corr", 0.0))
    enter_min = float(cfg.get("cycle_enter_min", 0.0))
    exit_max = float(cfg.get("cycle_exit_max", 0.0))
    score_weight = float(cfg.get("cycle_score_weight", 0.0))

    for market, ms in series_map.items():
        model = cycle_models.get(market)
        if model is None or abs(model.corr) < min_cycle_corr:
            if mode == "filter":
                ms.enters = [False] * len(ms.dates)
            continue

        w = 2.0 * math.pi / float(model.period)
        for i, d in enumerate(ms.dates):
            lag_score = ms.scores[i]
            if lag_score is None:
                continue

            x = date_to_ordinal(d)
            cycle_score = model.coef_sin * math.sin(w * x) + model.coef_cos * math.cos(w * x)
            combined = lag_score + score_weight * cycle_score
            ms.scores[i] = combined
            ms.enters[i] = combined > float(cfg["enter_threshold"]) and cycle_score >= enter_min
            ms.exits[i] = ms.exits[i] or (combined < float(cfg["exit_threshold"])) or (cycle_score <= exit_max)


def run_btc_lag_cfg(
    series: Dict[str, engine.MarketSeries],
    btc_market: str,
    lag_rules: dict[str, dict],
    cycle_models: dict[str, CycleModel] | None,
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
    entry_cooldown_days: int | None,
    min_holding_days: int | None,
) -> engine.BacktestResult | None:
    copied = lagbt.copy_series_map(series)
    lagbt.build_btc_lag_signals(
        copied,
        btc_market=btc_market,
        btc_lag_rules=lag_rules,
        lag_mode=cfg["lag_mode"],
        fixed_lag=cfg["fixed_lag"],
        lookback=cfg["lookback"],
        enter_threshold=cfg["enter_threshold"],
        exit_threshold=cfg["exit_threshold"],
        min_corr=cfg["min_corr"],
        require_lag_min=cfg["require_lag_min"],
        require_lag_max=cfg["require_lag_max"],
    )
    apply_btc_regime_filter(copied, btc_market, cfg)
    apply_cycle_overlay(copied, cycle_models, cfg)
    filtered = {
        m: ms
        for m, ms in copied.items()
        if m != btc_market and len(ms.closes) >= min_market_days
    }
    if not filtered:
        return None
    result = engine.backtest(
        filtered,
        strategy="btc_lag",
        params=cfg,
        initial_capital=initial_capital,
        max_positions=max_positions,
        rebalance_every=rebalance_every,
        fee_rate=fee_bps / 10_000.0,
        slippage_rate=slippage_bps / 10_000.0,
        stop_loss=stop_loss,
        take_profit=take_profit,
        max_holding_days=max_holding_days,
        entry_cooldown_days=entry_cooldown_days,
        min_holding_days=min_holding_days,
    )
    return result


def build_cfg_grid(args: argparse.Namespace) -> list[dict]:
    lag_modes = [m.strip() for m in args.lag_modes.split(",") if m.strip()]
    global_lags = parse_list(args.global_lags, int)
    lookbacks = parse_list(args.lookback_grid, int)
    enters = parse_list(args.enter_grid, float)
    exits = parse_list(args.exit_grid, float)
    min_corrs = parse_list(args.min_corr_grid, float)
    regime_modes = [m.strip() for m in args.regime_modes.split(",") if m.strip()]
    regime_shorts = parse_list(args.regime_short_grid, int)
    regime_longs = parse_list(args.regime_long_grid, int)
    regime_vol_windows = parse_list(args.regime_vol_window_grid, int)
    regime_vol_maxs = parse_list(args.regime_vol_max_grid, float)
    regime_force_exits = parse_bool_list(args.regime_force_exit_grid)
    cycle_modes = [m.strip() for m in args.cycle_modes.split(",") if m.strip()]
    cycle_min_corrs = parse_list(args.cycle_min_corr_grid, float)
    cycle_enter_mins = parse_list(args.cycle_enter_min_grid, float)
    cycle_exit_maxs = parse_list(args.cycle_exit_max_grid, float)
    cycle_score_weights = parse_list(args.cycle_score_weight_grid, float)

    if not regime_modes:
        regime_modes = ["off"]
    if not regime_force_exits:
        regime_force_exits = [True]
    if not cycle_modes:
        cycle_modes = ["off"]

    def add_cycle_variants(base_cfg: dict, out_list: list[dict]) -> None:
        for cycle_mode in cycle_modes:
            if cycle_mode == "off":
                c = dict(base_cfg)
                c.update(
                    {
                        "cycle_mode": "off",
                        "cycle_min_corr": 0.0,
                        "cycle_enter_min": 0.0,
                        "cycle_exit_max": 0.0,
                        "cycle_score_weight": 0.0,
                    }
                )
                out_list.append(c)
                continue
            for cycle_min_corr in cycle_min_corrs:
                for cycle_enter_min in cycle_enter_mins:
                    for cycle_exit_max in cycle_exit_maxs:
                        for cycle_score_weight in cycle_score_weights:
                            c = dict(base_cfg)
                            c.update(
                                {
                                    "cycle_mode": cycle_mode,
                                    "cycle_min_corr": cycle_min_corr,
                                    "cycle_enter_min": cycle_enter_min,
                                    "cycle_exit_max": cycle_exit_max,
                                    "cycle_score_weight": cycle_score_weight,
                                }
                            )
                            out_list.append(c)

    out = []
    for lag_mode in lag_modes:
        lag_values = global_lags if lag_mode == "global" else [0]
        for fixed_lag in lag_values:
            for lookback in lookbacks:
                for enter in enters:
                    for exit_ in exits:
                        for min_corr in min_corrs:
                            base = {
                                "lag_mode": lag_mode,
                                "fixed_lag": fixed_lag,
                                "lookback": lookback,
                                "enter_threshold": enter,
                                "exit_threshold": exit_,
                                "min_corr": min_corr,
                                "require_lag_min": args.require_lag_min,
                                "require_lag_max": args.require_lag_max,
                            }
                            for regime_mode in regime_modes:
                                if regime_mode == "off":
                                    cfg = dict(base)
                                    cfg.update(
                                        {
                                            "regime_mode": "off",
                                            "regime_short": None,
                                            "regime_long": None,
                                            "regime_vol_window": None,
                                            "regime_vol_max": None,
                                            "regime_force_exit": False,
                                        }
                                    )
                                    add_cycle_variants(cfg, out)
                                    continue

                                if regime_mode == "ma":
                                    for rs in regime_shorts:
                                        for rl in regime_longs:
                                            if rs >= rl:
                                                continue
                                            for force_exit in regime_force_exits:
                                                cfg = dict(base)
                                                cfg.update(
                                                    {
                                                        "regime_mode": "ma",
                                                        "regime_short": rs,
                                                        "regime_long": rl,
                                                        "regime_vol_window": None,
                                                        "regime_vol_max": None,
                                                        "regime_force_exit": force_exit,
                                                    }
                                                )
                                                add_cycle_variants(cfg, out)
                                    continue

                                if regime_mode == "vol":
                                    for vw in regime_vol_windows:
                                        for vmax in regime_vol_maxs:
                                            for force_exit in regime_force_exits:
                                                cfg = dict(base)
                                                cfg.update(
                                                    {
                                                        "regime_mode": "vol",
                                                        "regime_short": None,
                                                        "regime_long": None,
                                                        "regime_vol_window": vw,
                                                        "regime_vol_max": vmax,
                                                        "regime_force_exit": force_exit,
                                                    }
                                                )
                                                add_cycle_variants(cfg, out)
                                    continue

                                if regime_mode == "ma_vol":
                                    for rs in regime_shorts:
                                        for rl in regime_longs:
                                            if rs >= rl:
                                                continue
                                            for vw in regime_vol_windows:
                                                for vmax in regime_vol_maxs:
                                                    for force_exit in regime_force_exits:
                                                        cfg = dict(base)
                                                        cfg.update(
                                                            {
                                                                "regime_mode": "ma_vol",
                                                                "regime_short": rs,
                                                                "regime_long": rl,
                                                                "regime_vol_window": vw,
                                                                "regime_vol_max": vmax,
                                                                "regime_force_exit": force_exit,
                                                            }
                                                        )
                                                        add_cycle_variants(cfg, out)
    return out


def pick_best_train_result(
    series_train: Dict[str, engine.MarketSeries],
    btc_market: str,
    args: argparse.Namespace,
    cfg_grid: list[dict],
) -> tuple[dict, engine.BacktestResult, dict[str, dict]] | None:
    def choose_best_cfg(
        eval_series: Dict[str, engine.MarketSeries],
        eval_lag_rules: dict[str, dict],
        eval_cycle_models: dict[str, CycleModel] | None,
        min_days: int,
        min_trades: int,
    ) -> tuple[dict | None, engine.BacktestResult | None]:
        local_best_cfg = None
        local_best_result = None
        local_best_key = None

        for cfg in cfg_grid:
            result = run_btc_lag_cfg(
                eval_series,
                btc_market,
                eval_lag_rules,
                eval_cycle_models,
                cfg,
                min_market_days=min_days,
                initial_capital=args.initial_capital,
                max_positions=args.max_positions,
                rebalance_every=args.rebalance_every,
                fee_bps=args.fee_bps,
                slippage_bps=args.slippage_bps,
                stop_loss=args.stop_loss,
                take_profit=args.take_profit,
                max_holding_days=args.max_holding_days,
                entry_cooldown_days=args.entry_cooldown_days,
                min_holding_days=args.min_holding_days,
            )
            if result is None:
                continue
            if result.trade_count < min_trades:
                continue
            if args.max_trades_per_year > 0 and annualized_trade_count(result) > args.max_trades_per_year:
                continue
            key = selection_key(result, args.objective)
            if local_best_key is None or key > local_best_key:
                local_best_key = key
                local_best_cfg = cfg
                local_best_result = result

        if local_best_result is not None:
            return local_best_cfg, local_best_result

        # fallback without minimum trade constraint
        for cfg in cfg_grid:
            result = run_btc_lag_cfg(
                eval_series,
                btc_market,
                eval_lag_rules,
                eval_cycle_models,
                cfg,
                min_market_days=min_days,
                initial_capital=args.initial_capital,
                max_positions=args.max_positions,
                rebalance_every=args.rebalance_every,
                fee_bps=args.fee_bps,
                slippage_bps=args.slippage_bps,
                stop_loss=args.stop_loss,
                take_profit=args.take_profit,
                max_holding_days=args.max_holding_days,
                entry_cooldown_days=args.entry_cooldown_days,
                min_holding_days=args.min_holding_days,
            )
            if result is None:
                continue
            if args.max_trades_per_year > 0 and annualized_trade_count(result) > args.max_trades_per_year:
                continue
            key = selection_key(result, args.objective)
            if local_best_key is None or key > local_best_key:
                local_best_key = key
                local_best_cfg = cfg
                local_best_result = result
        return local_best_cfg, local_best_result

    # mode: choose by full-train
    if args.select_mode == "train":
        lag_rules = lagbt.pick_lag_rules(series_train, btc_market, args.max_lag_est)
        if not lag_rules:
            return None
        cycle_models = fit_cycle_models(series_train, args.cycle_min_period, args.cycle_max_period)
        best_cfg, best_result = choose_best_cfg(
            series_train,
            lag_rules,
            cycle_models,
            min_days=args.min_market_days,
            min_trades=args.min_train_trades,
        )
        if best_result is None or best_cfg is None:
            return None
        return best_cfg, best_result, lag_rules

    # mode: choose by validation split inside training window
    btc = series_train.get(btc_market)
    if btc is None:
        return None

    if len(btc.dates) <= args.val_days + 60:
        # not enough for validation split; fallback to train mode
        lag_rules = lagbt.pick_lag_rules(series_train, btc_market, args.max_lag_est)
        if not lag_rules:
            return None
        cycle_models = fit_cycle_models(series_train, args.cycle_min_period, args.cycle_max_period)
        best_cfg, best_result = choose_best_cfg(
            series_train,
            lag_rules,
            cycle_models,
            min_days=args.min_market_days,
            min_trades=args.min_train_trades,
        )
        if best_result is None or best_cfg is None:
            return None
        return best_cfg, best_result, lag_rules

    pre_start = btc.dates[0]
    pre_end = btc.dates[-args.val_days - 1]
    val_start = btc.dates[-args.val_days]
    val_end = btc.dates[-1]

    series_pre = slice_universe(series_train, pre_start, pre_end)
    series_val = slice_universe(series_train, val_start, val_end)
    if btc_market not in series_pre or btc_market not in series_val:
        return None

    lag_rules_pre = lagbt.pick_lag_rules(series_pre, btc_market, args.max_lag_est)
    if not lag_rules_pre:
        return None
    cycle_models_pre = fit_cycle_models(series_pre, args.cycle_min_period, args.cycle_max_period)

    best_cfg, _ = choose_best_cfg(
        series_val,
        lag_rules_pre,
        cycle_models_pre,
        min_days=max(5, min(args.min_market_days, args.val_days // 2)),
        min_trades=args.min_val_trades,
    )
    if best_cfg is None:
        return None

    # fit train report and test-time rules on full train
    lag_rules_full = lagbt.pick_lag_rules(series_train, btc_market, args.max_lag_est)
    if not lag_rules_full:
        return None
    cycle_models_full = fit_cycle_models(series_train, args.cycle_min_period, args.cycle_max_period)
    train_result = run_btc_lag_cfg(
        series_train,
        btc_market,
        lag_rules_full,
        cycle_models_full,
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
        entry_cooldown_days=args.entry_cooldown_days,
        min_holding_days=args.min_holding_days,
    )
    if train_result is None:
        return None
    return best_cfg, train_result, lag_rules_full


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
                "train_markets",
                "test_markets",
                "lag_rule_count",
                "lag_mode",
                "fixed_lag",
                "lookback",
                "enter_threshold",
                "exit_threshold",
                "min_corr",
                "require_lag_min",
                "require_lag_max",
                "regime_mode",
                "regime_short",
                "regime_long",
                "regime_vol_window",
                "regime_vol_max",
                "regime_force_exit",
                "cycle_mode",
                "cycle_min_corr",
                "cycle_enter_min",
                "cycle_exit_max",
                "cycle_score_weight",
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
                    item.train_markets,
                    item.test_markets,
                    item.lag_rule_count,
                    p["lag_mode"],
                    p["fixed_lag"],
                    p["lookback"],
                    p["enter_threshold"],
                    p["exit_threshold"],
                    p["min_corr"],
                    p["require_lag_min"],
                    p["require_lag_max"],
                    p["regime_mode"],
                    p["regime_short"],
                    p["regime_long"],
                    p["regime_vol_window"],
                    p["regime_vol_max"],
                    p["regime_force_exit"],
                    p["cycle_mode"],
                    p["cycle_min_corr"],
                    p["cycle_enter_min"],
                    p["cycle_exit_max"],
                    p["cycle_score_weight"],
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

    oos_dates = []
    oos_curve = []
    stitched_capital = meta.get("initial_capital", 1_000_000)
    for item in rows:
        dates = item.test_result.equity_dates
        curve = item.test_result.equity_curve
        if not dates or not curve:
            continue
        if curve[0] == 0:
            continue
        scale = stitched_capital / curve[0]
        scaled = [v * scale for v in curve]
        if oos_dates and dates[0] == oos_dates[-1]:
            oos_dates.extend(dates[1:])
            oos_curve.extend(scaled[1:])
        else:
            oos_dates.extend(dates)
            oos_curve.extend(scaled)
        stitched_capital = scaled[-1]

    oos_total_return = 0.0
    oos_cagr = 0.0
    oos_mdd = 0.0
    if oos_curve:
        initial_capital = float(meta.get("initial_capital", 1_000_000))
        oos_total_return = oos_curve[-1] / initial_capital - 1.0
        peak = oos_curve[0]
        for v in oos_curve:
            if v > peak:
                peak = v
            if peak > 0:
                oos_mdd = max(oos_mdd, (peak - v) / peak)
        if len(oos_dates) >= 2:
            d0 = datetime.strptime(oos_dates[0], "%Y-%m-%d").date()
            d1 = datetime.strptime(oos_dates[-1], "%Y-%m-%d").date()
            days = max(1, (d1 - d0).days + 1)
            oos_cagr = (oos_curve[-1] / initial_capital) ** (365.0 / days) - 1.0

    all_test_trades: list[dict] = []
    for item in rows:
        all_test_trades.extend(item.test_result.trades)

    payload = {
        "meta": meta,
        "fold_count": len(rows),
        "train_compound_return": train_compound - 1.0,
        "test_compound_return": test_compound - 1.0,
        "avg_test_sharpe": (sum(r.test_result.sharpe for r in rows) / len(rows)) if rows else 0.0,
        "avg_test_calmar": (sum(r.test_result.calmar for r in rows) / len(rows)) if rows else 0.0,
        "avg_test_max_drawdown": (sum(r.test_result.max_drawdown for r in rows) / len(rows)) if rows else 0.0,
        "total_test_trades": sum(r.test_result.trade_count for r in rows),
        "avg_test_annualized_trades": (
            sum(annualized_trade_count(r.test_result) for r in rows) / len(rows)
        )
        if rows
        else 0.0,
        "top_test_markets_by_buy_count": top_markets_by_trade_count(all_test_trades, side_prefix="buy", top_n=15),
        "top_test_markets_by_sell_count": top_markets_by_trade_count(all_test_trades, side_prefix="sell", top_n=15),
        "oos_total_return": oos_total_return,
        "oos_cagr": oos_cagr,
        "oos_mdd": oos_mdd,
        "folds": [
            {
                "fold": r.fold,
                "train_start": r.train_start,
                "train_end": r.train_end,
                "test_start": r.test_start,
                "test_end": r.test_end,
                "train_markets": r.train_markets,
                "test_markets": r.test_markets,
                "lag_rule_count": r.lag_rule_count,
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
                "test_top_markets_by_buy_count": top_markets_by_trade_count(
                    r.test_result.trades, side_prefix="buy", top_n=8
                ),
                "test_top_markets_by_sell_count": top_markets_by_trade_count(
                    r.test_result.trades, side_prefix="sell", top_n=8
                ),
            }
            for r in rows
        ],
    }
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward validation for BTC-lag strategy")
    p.add_argument("--summary", type=str, default="/tmp/upbit_okx_overlap_all_v2_summary.json")
    p.add_argument("--data-dir", type=str, default="/tmp/upbit_okx_overlap_all_v2/upbit")
    p.add_argument("--rank-csv", type=str, default="/tmp/top50_overlap_by_upbit_volume.csv")
    p.add_argument("--top-n", type=int, default=50)
    p.add_argument("--min-overlap-rows", type=int, default=120)
    p.add_argument("--btc-market", type=str, default="KRW-BTC")
    p.add_argument("--train-days", type=int, default=540)
    p.add_argument("--test-days", type=int, default=120)
    p.add_argument("--step-days", type=int, default=120)
    p.add_argument("--min-market-days", type=int, default=30)
    p.add_argument("--min-train-trades", type=int, default=20)
    p.add_argument("--select-mode", type=str, default="validation", choices=["train", "validation"])
    p.add_argument("--objective", type=str, default="return", choices=["sharpe", "return"])
    p.add_argument("--val-days", type=int, default=120)
    p.add_argument("--min-val-trades", type=int, default=8)
    p.add_argument("--max-trades-per-year", type=float, default=90.0)

    p.add_argument("--lag-modes", type=str, default="global,market")
    p.add_argument("--global-lags", type=str, default="0,1,2,3,5,7,10")
    p.add_argument("--max-lag-est", type=int, default=30)
    p.add_argument("--lookback-grid", type=str, default="5,10,20")
    p.add_argument("--enter-grid", type=str, default="0.001,0.002,0.004")
    p.add_argument("--exit-grid", type=str, default="-0.001,-0.002,-0.004")
    p.add_argument("--min-corr-grid", type=str, default="0.0,0.1")
    p.add_argument("--require-lag-min", type=int, default=None)
    p.add_argument("--require-lag-max", type=int, default=None)
    p.add_argument("--regime-modes", type=str, default="off,ma")
    p.add_argument("--regime-short-grid", type=str, default="20")
    p.add_argument("--regime-long-grid", type=str, default="120")
    p.add_argument("--regime-vol-window-grid", type=str, default="20")
    p.add_argument("--regime-vol-max-grid", type=str, default="0.06")
    p.add_argument("--regime-force-exit-grid", type=str, default="1")
    p.add_argument("--cycle-min-period", type=int, default=7)
    p.add_argument("--cycle-max-period", type=int, default=60)
    p.add_argument("--cycle-modes", type=str, default="off")
    p.add_argument("--cycle-min-corr-grid", type=str, default="0.03")
    p.add_argument("--cycle-enter-min-grid", type=str, default="0.0")
    p.add_argument("--cycle-exit-max-grid", type=str, default="-0.0002")
    p.add_argument("--cycle-score-weight-grid", type=str, default="0.0")

    p.add_argument("--initial-capital", type=float, default=1_000_000)
    p.add_argument("--max-positions", type=int, default=2)
    p.add_argument("--rebalance-every", type=int, default=7)
    p.add_argument("--fee-bps", type=float, default=1.0)
    p.add_argument("--slippage-bps", type=float, default=0.0)
    p.add_argument("--stop-loss", type=float, default=0.08)
    p.add_argument("--take-profit", type=float, default=0.20)
    p.add_argument("--max-holding-days", type=int, default=120)
    p.add_argument("--entry-cooldown-days", type=int, default=3)
    p.add_argument("--min-holding-days", type=int, default=3)

    p.add_argument("--notify-bell", action="store_true")
    p.add_argument("--out-csv", type=str, default="/tmp/btc_lag_walkforward_top50.csv")
    p.add_argument("--out-json", type=str, default="/tmp/btc_lag_walkforward_top50.json")
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
    if args.entry_cooldown_days < 0:
        raise SystemExit("--entry-cooldown-days must be >= 0")
    if args.min_holding_days < 0:
        raise SystemExit("--min-holding-days must be >= 0")
    if args.max_trades_per_year < 0:
        raise SystemExit("--max-trades-per-year must be >= 0")

    overlap_rows = load_overlap_rows(args.summary, args.min_overlap_rows)
    if not overlap_rows:
        raise SystemExit("No overlap rows after filtering")

    if args.rank_csv:
        ranked = load_ranked_markets(args.rank_csv, args.top_n)
        selected = set(ranked)
    else:
        selected = set(overlap_rows.keys())

    selected.add(args.btc_market)

    all_series = engine.load_market_series(args.data_dir, market_prefix="")
    overlap_series = build_overlap_series(all_series, overlap_rows, selected)
    if args.btc_market not in overlap_series:
        raise SystemExit(f"{args.btc_market} not available in overlap-clipped series")

    cfg_grid = build_cfg_grid(args)
    if not cfg_grid:
        raise SystemExit("empty config grid")

    btc_dates = overlap_series[args.btc_market].dates
    if len(btc_dates) < args.train_days + args.test_days:
        raise SystemExit("not enough BTC overlap dates for requested train/test windows")

    fold_starts = []
    i = args.train_days
    last = len(btc_dates) - args.test_days
    while i <= last:
        fold_starts.append(i)
        i += args.step_days

    print(
        f"walk-forward start: folds={len(fold_starts)} cfgs={len(cfg_grid)} "
        f"markets={len(overlap_series)} btc_dates={len(btc_dates)} "
        f"fee_bps={args.fee_bps} cooldown={args.entry_cooldown_days} "
        f"min_hold={args.min_holding_days} max_trades_per_year={args.max_trades_per_year}"
    )
    if args.notify_bell:
        print("\a", end="", flush=True)

    outcomes: list[FoldOutcome] = []
    for idx, anchor in enumerate(fold_starts, start=1):
        train_start = btc_dates[anchor - args.train_days]
        train_end = btc_dates[anchor - 1]
        test_start = btc_dates[anchor]
        test_end = btc_dates[min(len(btc_dates) - 1, anchor + args.test_days - 1)]

        print(
            f"[fold {idx}/{len(fold_starts)}] "
            f"train={train_start}..{train_end} test={test_start}..{test_end}"
        )

        series_train = slice_universe(overlap_series, train_start, train_end)
        series_test = slice_universe(overlap_series, test_start, test_end)
        if args.btc_market not in series_train or args.btc_market not in series_test:
            print("  skip: BTC not available in one of fold windows")
            continue

        picked = pick_best_train_result(series_train, args.btc_market, args, cfg_grid)
        if picked is None:
            print("  skip: could not select train best config")
            continue
        best_cfg, train_result, lag_rules = picked

        test_result = run_btc_lag_cfg(
            series_test,
            args.btc_market,
            lag_rules,
            fit_cycle_models(series_train, args.cycle_min_period, args.cycle_max_period),
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
            entry_cooldown_days=args.entry_cooldown_days,
            min_holding_days=args.min_holding_days,
        )
        if test_result is None:
            print("  skip: selected config invalid on test window")
            continue

        test_markets = sum(
            1
            for m, ms in series_test.items()
            if m != args.btc_market and len(ms.closes) >= max(5, min(args.min_market_days, args.test_days // 2))
        )
        train_markets = sum(
            1 for m, ms in series_train.items() if m != args.btc_market and len(ms.closes) >= args.min_market_days
        )

        outcomes.append(
            FoldOutcome(
                fold=idx,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_markets=train_markets,
                test_markets=test_markets,
                lag_rule_count=len(lag_rules),
                chosen_params=best_cfg,
                train_result=train_result,
                test_result=test_result,
            )
        )

        print(
            f"  picked mode={best_cfg['lag_mode']} lag={best_cfg['fixed_lag']} lb={best_cfg['lookback']} "
            f"enter={best_cfg['enter_threshold']} exit={best_cfg['exit_threshold']} mcorr={best_cfg['min_corr']} "
            f"regime={best_cfg['regime_mode']} cycle={best_cfg['cycle_mode']} | "
            f"train ret={train_result.total_return*100:.2f}% sharpe={train_result.sharpe:.3f} "
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
        "btc_market": args.btc_market,
        "train_days": args.train_days,
        "test_days": args.test_days,
        "step_days": args.step_days,
        "cfg_count": len(cfg_grid),
        "market_count": len(overlap_series),
        "initial_capital": args.initial_capital,
        "objective": args.objective,
        "fee_bps": args.fee_bps,
        "slippage_bps": args.slippage_bps,
        "entry_cooldown_days": args.entry_cooldown_days,
        "min_holding_days": args.min_holding_days,
        "max_trades_per_year": args.max_trades_per_year,
        "rebalance_every": args.rebalance_every,
        "max_positions": args.max_positions,
    }
    write_summary_json(args.out_json, outcomes, meta)

    compounded = 1.0
    for r in outcomes:
        compounded *= 1.0 + r.test_result.total_return
    print(
        f"walk-forward done: folds={len(outcomes)} compounded_test_return={(compounded - 1.0)*100:.2f}% "
        f"out_csv={args.out_csv} out_json={args.out_json}"
    )
    if args.notify_bell:
        print("\a", end="", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
