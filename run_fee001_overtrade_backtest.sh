#!/usr/bin/env bash
set -euo pipefail

SUMMARY="${SUMMARY:-/tmp/upbit_okx_overlap_all_v2_summary.json}"
DATA_DIR="${DATA_DIR:-/tmp/upbit_okx_overlap_all_v2/upbit}"
RANK_CSV="${RANK_CSV:-/tmp/top50_overlap_by_upbit_volume.csv}"
OUT_CSV="${OUT_CSV:-/tmp/btc_lag_walkforward_fee001_best.csv}"
OUT_JSON="${OUT_JSON:-/tmp/btc_lag_walkforward_fee001_best.json}"
PROFILE="${PROFILE:-aggressive}"

case "$PROFILE" in
  aggressive)
    REBALANCE_EVERY=5
    ENTRY_COOLDOWN_DAYS=2
    MIN_HOLDING_DAYS=2
    MAX_TRADES_PER_YEAR=110
    ;;
  balanced)
    REBALANCE_EVERY=10
    ENTRY_COOLDOWN_DAYS=3
    MIN_HOLDING_DAYS=3
    MAX_TRADES_PER_YEAR=70
    ;;
  *)
    echo "invalid PROFILE=$PROFILE (allowed: aggressive|balanced)" >&2
    exit 1
    ;;
esac

python3 btc_lag_walkforward.py \
  --summary "$SUMMARY" \
  --data-dir "$DATA_DIR" \
  --rank-csv "$RANK_CSV" \
  --top-n 50 \
  --train-days 540 \
  --test-days 120 \
  --step-days 120 \
  --min-overlap-rows 120 \
  --min-market-days 30 \
  --min-train-trades 0 \
  --select-mode train \
  --objective return \
  --lag-modes global \
  --global-lags 1 \
  --max-lag-est 30 \
  --lookback-grid 5 \
  --enter-grid 0.001 \
  --exit-grid=-0.001 \
  --min-corr-grid 0.0 \
  --regime-modes off \
  --cycle-modes filter \
  --cycle-min-corr-grid 0.03 \
  --cycle-enter-min-grid 0.0 \
  --cycle-exit-max-grid -0.0002 \
  --cycle-score-weight-grid 0.0 \
  --initial-capital 1000000 \
  --max-positions 3 \
  --rebalance-every "$REBALANCE_EVERY" \
  --fee-bps 1.0 \
  --slippage-bps 0.0 \
  --stop-loss 0.08 \
  --take-profit 0.20 \
  --max-holding-days 90 \
  --entry-cooldown-days "$ENTRY_COOLDOWN_DAYS" \
  --min-holding-days "$MIN_HOLDING_DAYS" \
  --max-trades-per-year "$MAX_TRADES_PER_YEAR" \
  --out-csv "$OUT_CSV" \
  --out-json "$OUT_JSON"

python3 - <<'PY' "$OUT_JSON"
import json
import sys

path = sys.argv[1]
with open(path, encoding='utf-8') as f:
    j = json.load(f)

print('summary_file=', path)
print('oos_total_return=', round(j.get('oos_total_return', 0.0) * 100, 2), '%')
print('oos_cagr=', round(j.get('oos_cagr', 0.0) * 100, 2), '%')
print('oos_mdd=', round(j.get('oos_mdd', 0.0) * 100, 2), '%')
print('total_test_trades=', j.get('total_test_trades', 0))
print('avg_test_annualized_trades=', round(j.get('avg_test_annualized_trades', 0.0), 2))
print('top_test_markets_by_buy_count=', j.get('top_test_markets_by_buy_count', [])[:10])
PY

echo "profile=$PROFILE"
echo "saved_csv=$OUT_CSV"
echo "saved_json=$OUT_JSON"
