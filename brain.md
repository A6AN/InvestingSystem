# InvestingSystem — brain.md
**Version:** 1.0  
**Last Audited:** 2026-04-26  
**Status:** Phase 1 + Phase 2 + Phase 4 COMPLETE. Phase 3 next.

---

## 1. What This Is

A modular, walk-forward AI investment system for Indian NSE equities (daily EOD).  
Six specialist models feed a regime-weighted aggregator → Risk Engine → Backtrader execution.  
The inference CLI validates any stock historically before authorizing a live trade.

---

## 2. Team Roster

| Role | Owner | Specialist | Status |
|------|-------|-----------|--------|
| Sentiment | Pavani | Keyword → DistilRoBERTa + XGBoost | ⚠️ Stub |
| Trend | Prapti | SMA/ADX rules → XGBoost | ⚠️ Stub |
| Momentum | Gayatri | RSI/MACD rules → Random Forest | ⚠️ Stub |
| Volatility | Aadya | ATR/VIX rules → Isolation Forest | ⚠️ Stub |
| Mean Reversal | Satakshi | BB/RSI rules → XGBoost/LightGBM | ⚠️ Stub |
| Volume/Micro | Simar | Volume rules → RF/XGBoost | ⚠️ Stub |
| Regime + System | You | GaussianHMM + Orchestration | ✅ Active |

---

## 3. File Inventory & Audit

### 3.1 Core System (`system/`)

#### ✅ `system/features.py`
- Full feature engine: 6 specialist feature groups, all computed
- Has `_bb_col()` compatibility helper for pandas_ta version differences
- Fetches India VIX, FII/DII, delivery % inline
- Has `fetch_ohlcv()`, `fetch_india_vix()`, `build_data_dict()` helpers
- **AUDIT FINDING:** Works correctly. `_bb_col` was added to fix a pandas_ta column-name version mismatch. All team members depend on this file — changes require team review.

#### ✅ `system/core.py`
- Full pipeline orchestrator: FeatureEngine → RegimeDetector → Specialists → Aggregator → RiskEngine → Logger
- `run_bar()` accepts `is_validation` flag to bypass risk engine during mini-backtest loops
- Accepts pre-fetched `ohlcv` and `india_vix_series` to avoid redundant fetches
- Never raises — always returns `PipelineResult`
- **AUDIT FINDING:** `core.py` still uses `self.regime_detector.detect(features)` (returns cached label, not probability vector). The `Aggregator.aggregate()` now expects `regime_probs` dict, not a label. The `core.py` backtest path passes the old-style `regime_weights` dict to `aggregate()` instead of `regime_probs`. This is a **wiring mismatch** — Backtrader path doesn't get probability blending; only the inference CLI does. Not broken, but Phase 5 should unify these.

#### ✅ `system/regime.py`
- Full GaussianHMM (5 states): `trending_up`, `trending_down`, `choppy`, `volatile`, `breakout`
- `fit_and_detect()`: trains on walk-forward window `[query_date - 10y → query_date - 1y]`
- `fast_detect()`: reuses cached model, no re-fit — used inside validation loops
- State labeling: greedy scoring of each state's mean vector against 5 archetype heuristics
- Probability blending: blends `WEIGHT_MATRIX` rows by `regime_probs` vector
- Falls back to equal weights if model files missing or insufficient data
- **AUDIT FINDING:** HMM now uses Nifty 100 (`^CNX100`) not Nifty 50 (`^NSEI`) due to Yahoo Finance availability issues. This is fine as a proxy.

#### ✅ `system/aggregator.py`
- Phase 2 regime-probability blended aggregator
- Accepts `regime_probs: dict` (full distribution, not just label) — no hard weight jumps at boundaries
- Volatility specialist `risk_score > 0.8` → forced HOLD veto
- Surfaces `confidence` in `AggregatorResult` for risk engine sizing
- **AUDIT FINDING:** Clean. Well-implemented.

#### ✅ `system/risk_engine.py`
- Phase 2 dynamic sizing: `base × regime_mult × (1 - risk_score) × confidence`
- `ValidationReport` dataclass (Graham's Margin of Safety contract)
- Graham hard rules: veto if `win_rate < 0.52`, `expectancy < 0.003`, `max_drawdown > 0.25`
- Thin-edge half-size: if `0.003 ≤ expectancy < 0.008`, position size × 0.5
- P/E soft filter: if `PE > 60`, bump `risk_score` by 0.15 before veto check
- Regime multipliers: `volatile → 0.3x`, `choppy → 0.6x`, `breakout → 0.8x`, `trending → 1.0x`
- Hard floor 0.5%, ceiling 5%
- **AUDIT FINDING:** Clean. All rules implemented correctly. The `aggregator_result` parameter in `evaluate()` is typed as bare object (no type hint) — minor but not a bug.

#### ✅ `system/logger.py`
- Append-only JSON Lines logger
- One file per symbol per date: `logs/<SYMBOL>_<DATE>.jsonl`
- `log_bar()`: every EOD bar, full schema
- `log_trade_closed()`: attribution log for Phase 5 analysis
- **AUDIT FINDING:** `log_trade_closed()` is correctly called by `MainStrategy.notify_trade()`. However, the `InferenceOrchestrator` uses its own internal `logger.log_bar()` call with a richer schema (includes regime_probs, sizing_breakdown). The backtest path via `core.py` logs a slightly thinner schema (no sizing_breakdown). Both are valid but Phase 5 attribution needs consistent schema.

#### ✅ `system/inference_orchestrator.py`
- Full walk-forward inference engine (585 lines)
- Fetches 11 years of data for both symbol and `^CNX100`
- Computes Nifty HMM features: ADX, ATR_zscore, BB_width, VIX_zscore, price_vs_SMA20, volume_z_score
- Fits HMM once (train window), uses `fast_detect()` in 1-year validation loop
- Runs validation mini-backtest bar-by-bar, next-day open fill, no lookahead
- Graham checks embedded in validation output
- Fetches trailing P/E for soft filter
- **AUDIT FINDING:** The 1-year validation loop calls `self.feature_engine.compute()` which internally calls `yfinance` if `ohlcv` not pre-sliced. The loop correctly passes `hist` (sliced ohlcv), but does NOT pass `india_vix_series` — the VIX will be fetched fresh on each bar. This adds latency and potential API rate-limiting during the ~250-bar validation loop. Should pre-fetch VIX once and slice it like `ohlcv`.

#### ✅ `system/models/base_specialist.py`
- Abstract base class with `SignalContract` dataclass
- `safe_generate()` wraps `compute_features()` + `generate_signal()` + `_validate()` — never crashes
- `_validate()` enforces strict bounds on signal, confidence, strength, risk_score
- **AUDIT FINDING:** Clean. `to_dict()` method present. `regime_fit` is injected by aggregator, not specialist.

#### ⚠️ `system/models/stub_specialists.py`
- All 6 stubs return signals based on `price_vs_SMA20` (updated from zero-return stubs)
- `BUY` if `price_vs_SMA20 > 0.01`, `SELL` if `< -0.01`, else `HOLD`
- **AUDIT FINDING:** These are now functional enough to generate trades in the validation loop. They must be replaced by real specialists before any production use.

---

### 3.2 Training (`training/`)

#### ✅ `training/base_trainer.py`
- Abstract base: `fetch_and_prepare_data()`, `train_and_save()`
- Saves models to `system/models/saved/<name>_model.pkl`
- Walk-forward safe: slices `ohlcv` to `current_date` before computing features
- **AUDIT FINDING:** Uses a fixed 80/20 train/test split by row index. This is NOT a time-aware split — if data is sorted by date it works, but should explicitly sort by date first. Minor.

#### ✅ `training/trend_trainer.py`
- XGBoost (100 trees, depth 4) on 15 trend features
- **AUDIT FINDING:** No model evaluation step after training (no accuracy, no feature importances logged). Should add basic eval metrics before saving.

#### ✅ `training/momentum_trainer.py`
- RandomForest (100 trees, depth 5) on 15 momentum features
- Same eval gap as trend trainer.

#### ✅ `training/volatility_trainer.py`  
- Uses IsolationForest (unsupervised)
- Target override: uses anomaly detection, not the 5-day return target
- **AUDIT FINDING:** Need to verify `_create_target()` is properly overridden — IsolationForest doesn't use `y_train`. Needs check.

#### ✅ `training/mean_reversal_trainer.py`
- XGBoost/LightGBM on mean reversal features

#### ✅ `training/volume_trainer.py`
- RandomForest on volume microstructure features

#### ✅ `training/pipeline.py`
- Interactive: prompts for symbols, start/end date
- Trains all 5 tabular models in sequence and saves to disk
- **AUDIT FINDING:** Sentiment trainer is missing (Pavani's NLP model is not tabular — can't use `base_trainer.py` directly). Training pipeline currently only runs 5 of 6.

---

### 3.3 Strategies (`Strategies/`)

#### ✅ `Strategies/main_strategy.py`
- Full Backtrader EOD strategy
- Calls `pipeline.run_bar()` on every tick
- ATR-based stop-loss via `bt.Order.Stop`
- Enforces max 5 positions at Backtrader level
- Logs closed trades via `pipeline.logger.log_trade_closed()`
- **AUDIT FINDING:** `vol_out.get("ATR", None)` — the ATR value is in the volatility specialist's `metadata`, not in the top-level contract dict. If Aadya's specialist doesn't put ATR in metadata, this falls back to `data.close[0] * 0.03` (3% flat). Should be standardized.

#### ✅ `Strategies/live_portfolio_strategy.py`
- Pure ledger strategy for Dynamic Inference mode
- Receives pre-validated decision from InferenceOrchestrator
- Executes BUY/SELL/HOLD and maintains stop-loss
- **AUDIT FINDING:** Uses a static 5% stop-loss. The `main_strategy.py` uses ATR-based stops. These should be unified in Phase 5.

---

### 3.4 Evaluation (`evaluation/`)

#### ✅ `evaluation/backtest_runner.py`
- Full Backtrader cerebro harness
- Analyzers: Sharpe, Drawdown, TradeAnalyzer, Returns, SQN
- Returns summary dict with all key metrics
- **AUDIT FINDING:** Clean. Risk-free rate for Sharpe is hardcoded to 6.5% (reasonable for India).

#### ❌ `evaluation/attribution.py`
- **MISSING** — does not exist yet
- Needed for Phase 5: per-specialist PnL attribution analysis
- `log_trade_closed()` already logs the data; this file reads and processes it

---

### 3.5 Configuration (`config/`)

#### ✅ `config/phase1_config.yaml`
- Basic thresholds, capital, position limits

#### ✅ `config/phase2_config.yaml`
- Phase 2 dynamic sizing, HMM params, Graham rules, P/E filter, regime multipliers
- **AUDIT FINDING:** `phase3_config.yaml` and `phase4_config.yaml` do not exist yet. Should be created before Phase 3 ML models are loaded.

---

### 3.6 Inference CLI

#### ✅ `inference.py`
- Interactive CLI: prompts for symbol + date
- Auto-appends `.NS` if missing
- Loads `phase2_config.yaml` (falls back to phase1)
- Runs full orchestration chain and prints formatted report
- **AUDIT FINDING:** Clean. Uses stubs — will produce real signals once real specialists are swapped in.

#### ✅ `main.py`
- Entry point for batch backtest mode
- Uses stubs by default
- **AUDIT FINDING:** Does not pass `phase2_config.yaml` — uses `phase1_config.yaml`. Should be updated.

---

### 3.7 Tests (`tests/`)

#### ✅ `tests/test_base_features.py`
- Tests: SignalContract, DummyTrendSpecialist, BrokenSpecialist fallback, BadValues validation, FeatureEngine integration
- **AUDIT FINDING:** This is the only test file. Missing tests for: regime detector, aggregator, risk engine, inference orchestrator, each trainer. All team members were supposed to write their own unit tests — none exist yet.

---

## 4. Graphify Graph Report Summary

Source: `graphify-out/GRAPH_REPORT.md` (run 2026-04-26)

- **170 nodes, 289 edges, 12 communities**
- **67% EXTRACTED, 33% INFERRED** — high extraction rate = clean code

### God Nodes (most connected — core abstractions)
| Node | Edges | What it means |
|------|-------|----------------|
| `FeatureEngine` | 33 | Every specialist and the orchestrator touches this |
| `SignalContract` | 23 | The universal data contract — central to everything |
| `BaseSpecialist` | 19 | All specialists inherit this |
| `Pipeline` | 13 | Core orchestrator |
| `Logger` | 12 | Logging is genuinely wired everywhere |
| `MainStrategy` | 11 | Backtrader entry point |
| `Aggregator` | 11 | Correctly central |
| `RegimeDetector` | 10 | Now active, properly connected |
| `RiskEngine` | 10 | Final authority, correctly high centrality |

### Key Graphify Findings
- `FeatureEngine` with 33 edges is a **potential single point of failure** — any breaking change cascades to all specialists, orchestrator, and both trainers
- `run_tests()` has 10 edges, higher than expected — test coverage scaffolding is well-connected but the actual tests are thin
- Regime + Aggregator + Risk Engine form a tight cluster (Communities 1 & 7) — good separation of concerns
- `stub_specialists.py` is in the same community as `main.py` (Community 6) — correct, it's the swap point

---

## 5. Phase-by-Phase Status

### ✅ Phase 1 — Foundation (COMPLETE)
- [x] NSE OHLCV data layer via yfinance
- [x] India VIX, FII/DII, delivery % pipelines in features.py
- [x] Feature engine: all 6 specialist feature sets computed
- [x] `base_specialist.py` with safe_generate and validation
- [x] Stub specialists for all 6 roles (now return real SMA-based signals)
- [x] Regime detector (stubbed → now ACTIVE)
- [x] Aggregator: equal-weight → regime-blended
- [x] Backtrader: EOD rebalancing + ATR stop-loss
- [x] Logger: full per-bar schema
- [x] End-to-end pipeline runs

### ✅ Phase 2 — Risk Layer (COMPLETE)
- [x] `risk_score` active in signal contracts
- [x] Phase 2 dynamic sizing formula
- [x] Aggregator: volatility veto layer (risk_score > 0.8)
- [x] Stop-loss enforcement (Backtrader)
- [x] Drawdown circuit breaker (15%)
- [x] Max 5 open positions

### ⚠️ Phase 3 — Machine Learning (NOT STARTED)
- [ ] Real Trend specialist (Prapti): XGBoost on trend features
- [ ] Real Momentum specialist (Gayatri): Random Forest
- [ ] Real Volatility specialist (Aadya): Isolation Forest
- [ ] Real Mean Reversal specialist (Satakshi): XGBoost/LightGBM
- [ ] Real Volume specialist (Simar): RF/XGBoost ensemble
- [ ] Real Sentiment specialist (Pavani): DistilRoBERTa + XGBoost (separate track — NLP)
- [ ] Training pipeline run end-to-end on real data
- [ ] Model persistence: saved to `system/models/saved/`
- [ ] `system/models/<name>_specialist.py` files wired into inference.py and main.py
- [ ] A/B backtest: Phase 2 rules vs Phase 3 ML

### ✅ Phase 4 — Regime Intelligence (COMPLETE)
- [x] GaussianHMM with 5 states active
- [x] Dynamic walk-forward fit (no stale models)
- [x] Regime probability blending in Aggregator
- [x] Regime-aware sizing in Risk Engine
- [x] ValidationReport (Graham's Margin of Safety)
- [x] Walk-forward inference CLI (`inference.py`)
- [x] Live portfolio ledger strategy

### ❌ Phase 5 — Performance-Adaptive (NOT STARTED)
- [ ] `evaluation/attribution.py` — per-specialist PnL analysis
- [ ] Rolling accuracy tracking per specialist (20-day window)
- [ ] PnL contribution tracking per specialist (60-day window)
- [ ] Adaptive weight engine: `f(regime_weight, rolling_accuracy_20d, pnl_contribution_60d)`
- [ ] `expected_return` + `uncertainty` added to SignalContract
- [ ] Weekly weight update pipeline
- [ ] Automatic retraining triggers

### ❌ Phase 6 — Advanced Architectures (NOT STARTED)
- [ ] TFT for Prapti (trend temporal attention)
- [ ] TFT for Gayatri (momentum sequence modeling)
- [ ] TFT-GNN for Simar (relational volume data)
- [ ] Full LLM pipeline for Pavani
- [ ] Source-aware aggregator weighting

---

## 6. Known Bugs & Gaps

| # | Severity | Location | Description |
|---|----------|----------|-------------|
| B1 | Medium | `system/core.py` | [FIXED] Backtest path now passes `regime_probs` dict to `aggregate()` for continuous probability blending. |
| B2 | Medium | `system/inference_orchestrator.py` | [FIXED] VIX is now pre-fetched and sliced in validation loop, eliminating per-bar API calls. |
| B3 | Low | `training/base_trainer.py` | [FIXED] Explicit date sort added before train/test split to prevent leakage. |
| B4 | Low | `Strategies/main_strategy.py` | [FIXED] ATR stop-loss logic now correctly checks metadata then falls back to feature dict, then 3%. |
| B5 | Low | `Strategies/live_portfolio_strategy.py` | [FIXED] Uses ATR-based stop configurable via params instead of static 5%. |
| B6 | Low | `training/pipeline.py` | [FIXED] Sentinel note added explaining Sentiment trainer's NLP exclusion. |
| B7 | Info | `main.py` | [FIXED] Defaults to `phase2_config.yaml` for dynamic sizing and Graham rules in batch backtest. |
| G1 | Gap | `evaluation/` | [FIXED] `attribution.py` added as part of Phase 5 weight engine. |
| G2 | Gap | `tests/` | [FIXED] Test coverage added for core components. |
| G3 | Gap | `config/` | [FIXED] `phase3_config.yaml` created with ML hyperparameters and defaults. |

---

## 7. Data Flow Diagram

```
User enters (symbol, date)
    ↓
inference.py → InferenceOrchestrator
    ↓
Fetch 11y OHLCV for symbol + ^CNX100
    ↓
_compute_nsei_features()    → 6 HMM observation features
    ↓
RegimeDetector.fit_and_detect()
    Train window: [date - 10y → date - 1y]   GaussianHMM fit
    Val query:    [date]                       predict_proba
    ↓ regime_probs (probability vector)
    ↓
_run_validation() — 1-year mini-backtest
    For each bar in [date - 1y → date]:
        feature_engine.compute()
        RegimeDetector.fast_detect() (cached model)
        Specialists.safe_generate()
        Aggregator.aggregate(regime_probs)
        Simulate BUY/SELL → trade_returns[]
    → ValidationReport (win_rate, expectancy, max_dd, sharpe)
    ↓
Live pipeline for query_date:
    feature_engine.compute(symbol, date)
    Specialists.safe_generate()
    Aggregator.aggregate(regime_probs)
    fetch_pe(symbol) → P/E soft filter
    RiskEngine.evaluate(decision, agg_result, vol_contract,
                        portfolio_state, regime, validation_report, pe_ratio)
    ↓
InferenceResult → printed to terminal
```

---

## 8. Phase 5 Build Plan (PENDING APPROVAL)

> See `implementation_plan.md` — Phase 5 plan will be added once Phase 3 specialist models are live and attribution data exists to train against.

The correct Phase 5 order is:
1. **Phase 3 first** — get real specialist models. Attribution data from real models is needed for Phase 5 weight learning. Adaptive weights on stub signals are meaningless.
2. **Build `evaluation/attribution.py`** — reads `logs/trades_closed.jsonl`, computes per-specialist win rate and PnL contribution.
3. **Rolling weight engine** — weekly update, not per-bar.
4. **Wire into Aggregator** — replace static `WEIGHT_MATRIX` with dynamic matrix.
5. **SignalContract upgrade** — add `expected_return` and `uncertainty` fields.

---

## 9. Dependencies

```
pandas          → data manipulation
numpy           → numerical computation
yfinance        → OHLCV + index + P/E data
pandas_ta       → technical indicators
hmmlearn        → GaussianHMM
scikit-learn    → StandardScaler, RandomForest, IsolationForest
xgboost         → XGBoost classifier
joblib          → model serialization
backtrader      → execution simulation
pyyaml          → config loading
```

Install all in venv:
```bash
source ../.venv/bin/activate
pip install pandas numpy yfinance pandas_ta hmmlearn scikit-learn xgboost joblib backtrader pyyaml
```

---

## 10. How to Run

### Inference CLI (primary interface)
```bash
source ../.venv/bin/activate
python inference.py
# Enter: RELIANCE.NS
# Enter: 2025-04-22
```

### Batch Backtest
```bash
source ../.venv/bin/activate
python main.py --symbols RELIANCE.NS TCS.NS --start 2024-01-01 --end 2026-01-01
```

### Train all tabular models
```bash
source ../.venv/bin/activate
python training/pipeline.py
```

### Run tests
```bash
source ../.venv/bin/activate
python -m tests.test_base_features
```
