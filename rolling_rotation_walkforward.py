#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import upbit_strategy_engine as eng


@dataclass
class PlainConfig:
    rebalance_every: int
    momentum_window: int
    vol_window: int
    enter_threshold: float
    exit_threshold: float


@dataclass
class RegimeConfig:
    rebalance_every: int
    regime_short: int
    regime_long: int
    momentum_window: int
    vol_window: int
    enter_threshold: float
    exit_threshold: float


def parse_date_only(raw: str) -> str:
    return raw.split("T", 1)[0]


def parse_list(raw: str, cast=float) -> list:
    if not raw:
        return []
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


def parse_int_pairs(raw: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if not raw:
        return out
    for token in raw.split(","):
        t = token.strip()
        if not t:
            continue
        if ":" not in t:
            raise ValueError(f"invalid pair token (need short:long): {t}")
        a, b = t.split(":", 1)
        out.append((int(a), int(b)))
    return out


def to_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def add_months(d: date, months: int) -> date:
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    last_day = 31
    while True:
        try:
            return date(y, m, min(d.day, last_day))
        except ValueError:
            last_day -= 1


def month_floor(d: date) -> date:
    return date(d.year, d.month, 1)


def to_iso(d: date) -> str:
    return d.isoformat()


def clip_series(ms: eng.MarketSeries, start_d: str, end_d: str, min_rows: int) -> eng.MarketSeries | None:
    idxs = [i for i, d in enumerate(ms.dates) if start_d <= d <= end_d]
    if len(idxs) < min_rows:
        return None
    s = idxs[0]
    e = idxs[-1] + 1
    dates = ms.dates[s:e]
    closes = ms.closes[s:e]
    vols = ms.volumes[s:e]
    return eng.MarketSeries(
        market=ms.market,
        dates=list(dates),
        closes=list(closes),
        volumes=list(vols),
        enters=[False] * len(dates),
        exits=[False] * len(dates),
        scores=[None] * len(dates),
        date_to_index={d: i for i, d in enumerate(dates)},
    )


def copy_series_map(series_map: dict[str, eng.MarketSeries]) -> dict[str, eng.MarketSeries]:
    out: dict[str, eng.MarketSeries] = {}
    for m, ms in series_map.items():
        out[m] = eng.MarketSeries(
            market=ms.market,
            dates=list(ms.dates),
            closes=list(ms.closes),
            volumes=list(ms.volumes),
            enters=list(ms.enters),
            exits=list(ms.exits),
            scores=list(ms.scores),
            date_to_index=dict(ms.date_to_index),
        )
    return out


def sma(values: list[float], window: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
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


def build_regime_allow(btc: eng.MarketSeries, short_w: int, long_w: int) -> dict[str, bool]:
    s = sma(btc.closes, short_w)
    l = sma(btc.closes, long_w)
    allow = {}
    for i, d in enumerate(btc.dates):
        allow[d] = bool(s[i] is not None and l[i] is not None and s[i] > l[i])
    return allow


def regime_on_ratio(allow: dict[str, bool]) -> float:
    if not allow:
        return 0.0
    return sum(1 for v in allow.values() if v) / len(allow)


def apply_regime_gate(series_map: dict[str, eng.MarketSeries], allow: dict[str, bool], force_exit: bool) -> None:
    for ms in series_map.values():
        for i, d in enumerate(ms.dates):
            if not allow.get(d, False):
                ms.enters[i] = False
                if force_exit:
                    ms.exits[i] = True


def load_overlap_filtered_markets(summary_path: str, min_overlap_rows: int) -> set[str]:
    payload = json.load(open(summary_path, encoding="utf-8"))
    out: set[str] = set()
    for row in payload["rows"]:
        if row["upbit_rows"] < min_overlap_rows or row["okx_rows"] < min_overlap_rows:
            continue
        out.add(row["upbit_market"])
    return out


def load_top_markets(rank_csv: str, top_n: int) -> list[str]:
    out = []
    with open(rank_csv, encoding="utf-8", newline="") as fp:
        rd = csv.reader(fp)
        for row in rd:
            if len(row) < 3:
                continue
            market = row[2].strip()
            if market:
                out.append(market)
            if top_n > 0 and len(out) >= top_n:
                break
    return out


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "win_rate": 0.0,
        }
    xs = sorted(values)
    n = len(xs)
    mid = n // 2
    if n % 2 == 1:
        median = xs[mid]
    else:
        median = (xs[mid - 1] + xs[mid]) / 2.0
    wins = sum(1 for x in xs if x > 0.0)
    return {
        "count": n,
        "mean": sum(xs) / n,
        "median": median,
        "min": xs[0],
        "max": xs[-1],
        "win_rate": wins / n,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Monthly rolling 1y train -> 1y test rotation walk-forward.")
    p.add_argument("--summary", type=str, default="/tmp/upbit_okx_overlap_all_v2_summary.json")
    p.add_argument("--data-dir", type=str, default="/tmp/upbit_okx_overlap_all_v2/upbit")
    p.add_argument("--rank-csv", type=str, default="/tmp/top50_overlap_by_upbit_volume.csv")
    p.add_argument("--top-n", type=int, default=50)
    p.add_argument("--min-overlap-rows", type=int, default=120)
    p.add_argument("--min-window-rows", type=int, default=120)
    p.add_argument("--btc-market", type=str, default="KRW-BTC")
    p.add_argument("--train-days", type=int, default=365)
    p.add_argument("--test-days", type=int, default=365)
    p.add_argument("--step-months", type=int, default=1)
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument("--initial-capital", type=float, default=1_000_000)
    p.add_argument("--max-positions", type=int, default=1)
    p.add_argument("--fee-bps", type=float, default=1.0)
    p.add_argument("--slippage-bps", type=float, default=5.0)

    p.add_argument("--rb-grid", type=str, default="3,5")
    p.add_argument("--mw-grid", type=str, default="5,10,20")
    p.add_argument("--vw-grid", type=str, default="10,20")
    p.add_argument("--enter-grid", type=str, default="0.0,0.01")
    p.add_argument("--exit-grid", type=str, default="-0.03,-0.01,0.0")
    p.add_argument("--regime-grid", type=str, default="20:120,30:150,50:200")
    p.add_argument("--regime-min-on-ratio", type=float, default=0.2)
    p.add_argument("--regime-force-exit", action="store_true")

    p.add_argument("--out-json", type=str, default="/tmp/rolling_rotation_walkforward.json")
    p.add_argument("--out-csv", type=str, default="/tmp/rolling_rotation_walkforward.csv")

    args = p.parse_args()

    rb_grid = parse_list(args.rb_grid, int)
    mw_grid = parse_list(args.mw_grid, int)
    vw_grid = parse_list(args.vw_grid, int)
    enter_grid = parse_list(args.enter_grid, float)
    exit_grid = parse_list(args.exit_grid, float)
    regime_grid = parse_int_pairs(args.regime_grid)

    overlap_markets = load_overlap_filtered_markets(args.summary, args.min_overlap_rows)
    ranked = load_top_markets(args.rank_csv, args.top_n)
    selected = [m for m in ranked if m in overlap_markets]
    all_series = eng.load_market_series(args.data_dir, "KRW")
    all_series = {m: ms for m, ms in all_series.items() if m in selected}
    if args.btc_market not in all_series:
        raise SystemExit(f"{args.btc_market} not found in selected universe.")

    btc = all_series[args.btc_market]
    start_btc = to_date(btc.dates[0])
    end_btc = to_date(btc.dates[-1])
    first_test = start_btc + timedelta(days=args.train_days)
    latest_test = end_btc - timedelta(days=args.test_days - 1)

    cur = month_floor(first_test)
    if cur < first_test:
        cur = add_months(cur, 1)

    windows: list[tuple[str, str, str, str]] = []
    while cur <= latest_test:
        test_start = cur
        train_end = test_start - timedelta(days=1)
        train_start = train_end - timedelta(days=args.train_days - 1)
        test_end = test_start + timedelta(days=args.test_days - 1)
        if train_start >= start_btc and test_end <= end_btc:
            windows.append((to_iso(train_start), to_iso(train_end), to_iso(test_start), to_iso(test_end)))
        if args.max_windows > 0 and len(windows) >= args.max_windows:
            break
        cur = add_months(cur, args.step_months)

    if not windows:
        raise SystemExit("No feasible rolling windows.")

    fee_rate = args.fee_bps / 10_000.0
    slip_rate = args.slippage_bps / 10_000.0

    rows = []
    total = len(windows)
    print(
        f"rolling start: windows={total} markets={len(all_series)} "
        f"cfg_plain={len(rb_grid)*len(mw_grid)*len(vw_grid)*len(enter_grid)*len(exit_grid)} "
        f"cfg_regime={len(rb_grid)*len(regime_grid)*len(mw_grid)*len(vw_grid)*len(enter_grid)*len(exit_grid)}"
    )

    for i, (train_s, train_e, test_s, test_e) in enumerate(windows, start=1):
        train_map: dict[str, eng.MarketSeries] = {}
        test_map: dict[str, eng.MarketSeries] = {}
        for m, ms in all_series.items():
            tr = clip_series(ms, train_s, train_e, args.min_window_rows)
            te = clip_series(ms, test_s, test_e, args.min_window_rows)
            if tr is None or te is None:
                continue
            train_map[m] = tr
            test_map[m] = te

        if args.btc_market not in train_map or args.btc_market not in test_map:
            continue
        if len(test_map) < 10:
            continue

        # 1) plain select on train
        plain_best_key = None
        plain_best_cfg = None
        plain_best_train = None
        for rb in rb_grid:
            for mw in mw_grid:
                for vw in vw_grid:
                    for en in enter_grid:
                        for ex in exit_grid:
                            params = {
                                "momentum_window": mw,
                                "vol_window": vw,
                                "enter_threshold": en,
                                "exit_threshold": ex,
                            }
                            sim = copy_series_map(train_map)
                            eng.prepare_signals(sim, "momentum", params)
                            r = eng.backtest(
                                sim,
                                "momentum",
                                params,
                                initial_capital=args.initial_capital,
                                max_positions=args.max_positions,
                                rebalance_every=rb,
                                fee_rate=fee_rate,
                                slippage_rate=slip_rate,
                                stop_loss=None,
                                take_profit=None,
                                max_holding_days=None,
                            )
                            key = (r.total_return, r.sharpe)
                            if plain_best_key is None or key > plain_best_key:
                                plain_best_key = key
                                plain_best_cfg = PlainConfig(rb, mw, vw, en, ex)
                                plain_best_train = r

        # 2) regime select on train
        regime_best_key = None
        regime_best_cfg = None
        regime_best_train = None
        regime_best_on_ratio = 0.0

        def scan_regime(allow_min_ratio: float) -> None:
            nonlocal regime_best_key, regime_best_cfg, regime_best_train, regime_best_on_ratio
            for rb in rb_grid:
                for rs, rl in regime_grid:
                    allow_train = build_regime_allow(train_map[args.btc_market], rs, rl)
                    on_ratio = regime_on_ratio(allow_train)
                    if on_ratio < allow_min_ratio:
                        continue
                    for mw in mw_grid:
                        for vw in vw_grid:
                            for en in enter_grid:
                                for ex in exit_grid:
                                    params = {
                                        "momentum_window": mw,
                                        "vol_window": vw,
                                        "enter_threshold": en,
                                        "exit_threshold": ex,
                                    }
                                    sim = copy_series_map(train_map)
                                    eng.prepare_signals(sim, "momentum", params)
                                    apply_regime_gate(sim, allow_train, force_exit=args.regime_force_exit)
                                    r = eng.backtest(
                                        sim,
                                        "momentum_regime",
                                        {
                                            **params,
                                            "regime_short": rs,
                                            "regime_long": rl,
                                        },
                                        initial_capital=args.initial_capital,
                                        max_positions=args.max_positions,
                                        rebalance_every=rb,
                                        fee_rate=fee_rate,
                                        slippage_rate=slip_rate,
                                        stop_loss=None,
                                        take_profit=None,
                                        max_holding_days=None,
                                    )
                                    key = (r.total_return, r.sharpe)
                                    if regime_best_key is None or key > regime_best_key:
                                        regime_best_key = key
                                        regime_best_cfg = RegimeConfig(rb, rs, rl, mw, vw, en, ex)
                                        regime_best_train = r
                                        regime_best_on_ratio = on_ratio

        scan_regime(args.regime_min_on_ratio)
        if regime_best_cfg is None:
            # Fallback for long bearish windows where strict regime-on ratio filters out all candidates.
            scan_regime(0.0)

        if plain_best_cfg is None or regime_best_cfg is None:
            continue

        # 3) evaluate OOS
        plain_params = {
            "momentum_window": plain_best_cfg.momentum_window,
            "vol_window": plain_best_cfg.vol_window,
            "enter_threshold": plain_best_cfg.enter_threshold,
            "exit_threshold": plain_best_cfg.exit_threshold,
        }
        plain_sim_test = copy_series_map(test_map)
        eng.prepare_signals(plain_sim_test, "momentum", plain_params)
        plain_test = eng.backtest(
            plain_sim_test,
            "momentum",
            plain_params,
            initial_capital=args.initial_capital,
            max_positions=args.max_positions,
            rebalance_every=plain_best_cfg.rebalance_every,
            fee_rate=fee_rate,
            slippage_rate=slip_rate,
            stop_loss=None,
            take_profit=None,
            max_holding_days=None,
        )

        regime_params = {
            "momentum_window": regime_best_cfg.momentum_window,
            "vol_window": regime_best_cfg.vol_window,
            "enter_threshold": regime_best_cfg.enter_threshold,
            "exit_threshold": regime_best_cfg.exit_threshold,
        }
        regime_allow_test = build_regime_allow(
            test_map[args.btc_market], regime_best_cfg.regime_short, regime_best_cfg.regime_long
        )
        regime_sim_test = copy_series_map(test_map)
        eng.prepare_signals(regime_sim_test, "momentum", regime_params)
        apply_regime_gate(regime_sim_test, regime_allow_test, force_exit=args.regime_force_exit)
        regime_test = eng.backtest(
            regime_sim_test,
            "momentum_regime",
            {
                **regime_params,
                "regime_short": regime_best_cfg.regime_short,
                "regime_long": regime_best_cfg.regime_long,
            },
            initial_capital=args.initial_capital,
            max_positions=args.max_positions,
            rebalance_every=regime_best_cfg.rebalance_every,
            fee_rate=fee_rate,
            slippage_rate=slip_rate,
            stop_loss=None,
            take_profit=None,
            max_holding_days=None,
        )

        row = {
            "window_index": i,
            "train_start": train_s,
            "train_end": train_e,
            "test_start": test_s,
            "test_end": test_e,
            "market_count": len(test_map),
            "plain_rebalance_every": plain_best_cfg.rebalance_every,
            "plain_params": plain_params,
            "plain_train_return": plain_best_train.total_return,
            "plain_test_return": plain_test.total_return,
            "plain_test_mdd": plain_test.max_drawdown,
            "plain_test_sharpe": plain_test.sharpe,
            "plain_test_trades": plain_test.trade_count,
            "regime_rebalance_every": regime_best_cfg.rebalance_every,
            "regime_short": regime_best_cfg.regime_short,
            "regime_long": regime_best_cfg.regime_long,
            "regime_params": regime_params,
            "regime_train_return": regime_best_train.total_return,
            "regime_train_on_ratio": regime_best_on_ratio,
            "regime_test_return": regime_test.total_return,
            "regime_test_mdd": regime_test.max_drawdown,
            "regime_test_sharpe": regime_test.sharpe,
            "regime_test_trades": regime_test.trade_count,
            "regime_test_on_ratio": regime_on_ratio(regime_allow_test),
            "plain_minus_regime_return": plain_test.total_return - regime_test.total_return,
        }
        rows.append(row)
        print(
            f"[{i}/{total}] test={test_s}..{test_e} markets={len(test_map)} "
            f"plain={plain_test.total_return*100:.2f}% regime={regime_test.total_return*100:.2f}%"
        )

    if not rows:
        raise SystemExit("No valid window results.")

    plain_returns = [r["plain_test_return"] for r in rows]
    regime_returns = [r["regime_test_return"] for r in rows]
    plain_mdds = [r["plain_test_mdd"] for r in rows]
    regime_mdds = [r["regime_test_mdd"] for r in rows]
    plain_sharpes = [r["plain_test_sharpe"] for r in rows]
    regime_sharpes = [r["regime_test_sharpe"] for r in rows]
    deltas = [r["plain_minus_regime_return"] for r in rows]

    summary = {
        "meta": {
            "summary": args.summary,
            "data_dir": args.data_dir,
            "rank_csv": args.rank_csv,
            "top_n": args.top_n,
            "min_overlap_rows": args.min_overlap_rows,
            "min_window_rows": args.min_window_rows,
            "btc_market": args.btc_market,
            "train_days": args.train_days,
            "test_days": args.test_days,
            "step_months": args.step_months,
            "fee_bps": args.fee_bps,
            "slippage_bps": args.slippage_bps,
            "initial_capital": args.initial_capital,
            "max_positions": args.max_positions,
            "regime_force_exit": args.regime_force_exit,
        },
        "window_count": len(rows),
        "plain_return_summary": summarize(plain_returns),
        "regime_return_summary": summarize(regime_returns),
        "plain_mdd_summary": summarize(plain_mdds),
        "regime_mdd_summary": summarize(regime_mdds),
        "plain_sharpe_summary": summarize(plain_sharpes),
        "regime_sharpe_summary": summarize(regime_sharpes),
        "plain_minus_regime_return_summary": summarize(deltas),
        "rows": rows,
    }

    # CSV
    with open(args.out_csv, "w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(
            [
                "window_index",
                "train_start",
                "train_end",
                "test_start",
                "test_end",
                "market_count",
                "plain_rebalance_every",
                "plain_momentum_window",
                "plain_vol_window",
                "plain_enter",
                "plain_exit",
                "plain_train_return",
                "plain_test_return",
                "plain_test_mdd",
                "plain_test_sharpe",
                "plain_test_trades",
                "regime_rebalance_every",
                "regime_short",
                "regime_long",
                "regime_momentum_window",
                "regime_vol_window",
                "regime_enter",
                "regime_exit",
                "regime_train_return",
                "regime_train_on_ratio",
                "regime_test_return",
                "regime_test_mdd",
                "regime_test_sharpe",
                "regime_test_trades",
                "regime_test_on_ratio",
                "plain_minus_regime_return",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r["window_index"],
                    r["train_start"],
                    r["train_end"],
                    r["test_start"],
                    r["test_end"],
                    r["market_count"],
                    r["plain_rebalance_every"],
                    r["plain_params"]["momentum_window"],
                    r["plain_params"]["vol_window"],
                    r["plain_params"]["enter_threshold"],
                    r["plain_params"]["exit_threshold"],
                    r["plain_train_return"],
                    r["plain_test_return"],
                    r["plain_test_mdd"],
                    r["plain_test_sharpe"],
                    r["plain_test_trades"],
                    r["regime_rebalance_every"],
                    r["regime_short"],
                    r["regime_long"],
                    r["regime_params"]["momentum_window"],
                    r["regime_params"]["vol_window"],
                    r["regime_params"]["enter_threshold"],
                    r["regime_params"]["exit_threshold"],
                    r["regime_train_return"],
                    r["regime_train_on_ratio"],
                    r["regime_test_return"],
                    r["regime_test_mdd"],
                    r["regime_test_sharpe"],
                    r["regime_test_trades"],
                    r["regime_test_on_ratio"],
                    r["plain_minus_regime_return"],
                ]
            )

    with open(args.out_json, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    print(
        "rolling done:",
        f"windows={len(rows)}",
        f"plain_mean={summary['plain_return_summary']['mean']*100:.2f}%",
        f"regime_mean={summary['regime_return_summary']['mean']*100:.2f}%",
        f"out_json={args.out_json}",
        f"out_csv={args.out_csv}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
