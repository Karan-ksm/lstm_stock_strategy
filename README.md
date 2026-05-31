# LSTM Stock Strategy

A research pipeline for training an LSTM on historical OHLCV data,
backtesting the resulting strategy under realistic assumptions, and
running it against Alpaca paper trading on live market data.

This project is for research and learning. It is not investment
advice and is unlikely to produce a profitable strategy on its own.
The point of the system is to be honest and leakage free, so that
whatever the backtest tells you about a strategy is actually true.

## Decisions baked in

| Decision | Choice |
| -------- | ------ |
| Prediction target | Directional up/down probability (sigmoid head) |
| Model scope | One LSTM per ticker |
| Strategy | Long only, all in or all out |
| Live scheduling | Standalone script triggered by cron or launchd |

See `config.yaml` for everything tunable without code changes.

## Quick start

```bash
# 1. Set up a virtual environment (Python 3.11+).
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Add your Alpaca paper trading credentials.
cp .env.example .env
# Edit .env and paste in ALPACA_API_KEY and ALPACA_SECRET_KEY.

# 3. (Optional) Edit config.yaml. Tickers live under data.tickers.

# 4. Train one model per ticker.
python scripts/run_train.py

# 5. Backtest the trained models on the test split.
python scripts/run_backtest.py

# 6. Run one paper trading decision (do this once per trading day).
python scripts/run_paper_trade.py

# 7. Run the leakage and indicator tests.
pytest tests/
```

## Daily paper trading (cron / launchd)

Example crontab line for 4:30pm Eastern on weekdays. Adjust the path
to your checkout.

```cron
30 16 * * 1-5  cd /path/to/lstm_stock_strategy && /path/to/.venv/bin/python scripts/run_paper_trade.py
```

Each run appends one row to `artifacts/logs/paper_trade_log.csv` and
refreshes `artifacts/logs/paper_portfolio_value.png`.

## Repository layout

| Path | Purpose |
| ---- | ------- |
| `config.yaml` | All hyperparameters and paths. Edit this, not the code. |
| `src/data/` | Fetches OHLCV via yfinance and caches as Parquet. |
| `src/features/` | Indicators and a composable feature pipeline. |
| `src/dataset/` | Chronological splits, scaler fit on train only, sequence builder. |
| `src/model/` | LSTM definition and the save/load bundle. |
| `src/backtest/` | Signal logic, backtest engine, metrics, plots. |
| `src/paper_trade/` | Alpaca paper trading client and the daily run script. |
| `scripts/` | Thin CLI entry points. |
| `tests/` | Unit tests, especially around the leakage critical pieces. |
| `artifacts/` | Cached data, saved models, reports, paper trading logs. Gitignored. |

## Outputs by stage

| Stage | Where to look |
| ----- | -------------- |
| Cached daily bars | `artifacts/data_cache/<TICKER>.parquet` |
| Trained model bundle | `artifacts/models/<TICKER>/{model.keras, scaler.pkl, spec.json}` |
| Backtest report | `artifacts/reports/<TICKER>/{metrics.csv, trades.csv, *.png}` |
| Paper trading audit log | `artifacts/logs/paper_trade_log.csv` |
| Paper trading chart | `artifacts/logs/paper_portfolio_value.png` |

## Leakage firewalls

Three structural protections keep future information out of model
inputs:

1. **`src/dataset/splits.py`** rejects any non DatetimeIndex or
   descending index, and computes train, validation, and test slices
   strictly in time order with no overlap.
2. **`src/dataset/scaler.py`** wraps StandardScaler and raises if
   `fit()` is called more than once, so validation or test statistics
   cannot enter the input space.
3. **`src/dataset/sequences.py`** builds sliding windows with
   `numpy.sliding_window_view` after aligning features and target by
   index, so windows can only consist of consecutive past rows. The
   only place future information enters is the label, which looks
   forward `horizon` days and is the supervised target by design.

`pytest tests/` asserts each of these.

## Disclaimer

This project is provided for educational and research purposes only. It is
not financial advice. Nothing in this repository constitutes a recommendation
to buy, sell, or hold any security. Past performance, including backtest
results, does not guarantee or imply future performance. Trading any market
involves a real risk of loss, and you are solely responsible for any decisions
you make with this code or its outputs.

The Alpaca integration is wired to the paper trading endpoint only. The code
forces `paper=True` and never imports the live trading URL. Do not modify
that without first understanding the risks of running an unproven model
against a real brokerage account.

## License

Released under the MIT License. See [LICENSE](LICENSE) for the full text.
