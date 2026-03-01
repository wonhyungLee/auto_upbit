# Fee 0.01% + Overtrading Guard Patch

This patch updates the coin selection/trading backtest workflow with:
- Fee default set to `0.01%` per side (`--fee-bps 1.0`)
- Overtrading controls in engine:
  - `entry_cooldown_days`
  - `min_holding_days`
- Walk-forward selection guard:
  - `max_trades_per_year`
- Test-market selection summary in JSON output.
- Reproducible profitable rule profile script (`aggressive`/`balanced`).

## Files changed
- `upbit_strategy_engine.py`
- `btc_lag_walkforward.py`
- `btc_lag_backtest.py`
- `cycle_period_walkforward.py`
- `rolling_rotation_walkforward.py`
- `run_fee001_overtrade_backtest.sh` (new)

## Apply patch on another server
Run in the project root:

```bash
patch -p1 < fee001_overtrade_patch.patch
```

## Run backtest after patch

```bash
./run_fee001_overtrade_backtest.sh
```

Run with reduced trade frequency profile:

```bash
PROFILE=balanced ./run_fee001_overtrade_backtest.sh
```

Optional env overrides:

```bash
SUMMARY=/tmp/upbit_okx_overlap_all_v2_summary.json \
DATA_DIR=/tmp/upbit_okx_overlap_all_v2/upbit \
RANK_CSV=/tmp/top50_overlap_by_upbit_volume.csv \
OUT_CSV=/tmp/btc_lag_walkforward_fee001_best.csv \
OUT_JSON=/tmp/btc_lag_walkforward_fee001_best.json \
PROFILE=aggressive \
./run_fee001_overtrade_backtest.sh
```

## Expected outputs
- CSV: `/tmp/btc_lag_walkforward_fee001_best.csv`
- JSON: `/tmp/btc_lag_walkforward_fee001_best.json`

The JSON includes:
- OOS performance (`oos_total_return`, `oos_cagr`, `oos_mdd`)
- Trade activity (`total_test_trades`, `avg_test_annualized_trades`)
- Selected market frequency (`top_test_markets_by_buy_count`, `top_test_markets_by_sell_count`)

## Verified profiles (fee 0.01%, slippage 0%)
- `PROFILE=aggressive`: higher return / higher turnover
  - OOS total return: about `+202.42%`
  - OOS CAGR: about `+61.75%`
  - OOS MDD: about `31.74%`
  - Total test trades: `207` (annualized ~`89.95`)
- `PROFILE=balanced`: lower turnover / lower return
  - OOS total return: about `+53.83%`
  - OOS CAGR: about `+20.58%`
  - OOS MDD: about `24.94%`
  - Total test trades: `105` (annualized ~`45.62`)
