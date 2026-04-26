# AI Investment System — Master Plan
**Version:** 4.1 Final  
**Status:** Active Sprint  
**Market:** Indian Equities — Individual NSE Stocks  
**Timeframe:** Daily (End-of-Day)  
**Team:** 6 Specialists + Regime Lead (You)

---

# 1. Philosophy & Principles

## Objective

Build a modular, testable, data-driven investment system that makes **structured decisions** — not predictions. The edge comes from disciplined process, risk control, and iterative improvement. Not from clever models.

## Core Principles

| Principle | What it means in practice |
|---|---|
| **Simplicity > Complexity** | Every component must earn its place |
| **No fake metrics** | Only compute what is real and verifiable |
| **Separation of concerns** | Data ≠ Features ≠ Models ≠ Execution ≠ Risk |
| **Every decision is testable** | If you can't backtest it, you can't ship it |
| **System > individual models** | No single model should have too much authority |
| **Log everything** | You cannot improve what you don't measure |
| **Earn the right to complexity** | Don't add ML to a rule-based system that doesn't run yet |
| **Parallel ownership** | Each specialist owns their model end-to-end |
| **No cross-contamination** | Specialists are blind to each other by design |

---

# 2. Team Roster

| Role | Name | Specialist | Core Question |
|------|------|-----------|---------------|
| **Sentiment** | **Pavani** | Sentiment Specialist | What is the market narrative and sentiment right now? |
| **Trend** | **Prapti** | Trend Specialist | Is there a clear directional trend and how strong is it? |
| **Momentum** | **Gayatri** | Momentum Specialist | Is price accelerating or decelerating? |
| **Volatility** | **Aadya** | Volatility Specialist | Is the market in a high-risk or low-risk state right now? |
| **Mean Reversal** | **Satakshi** | Mean Reversal Specialist | Is price extended from its mean and likely to revert? |
| **Volume** | **Simar** | Volume & Microstructure Specialist | What is the volume profile and order flow telling us about conviction? |
| **Regime + System** | **You** | Regime Detector + Orchestration | What environment are we in? Does the system run end-to-end? |

---

# 3. System Architecture

## High-Level Pipeline

```
Raw Data (NSE OHLCV Daily + Indian News/Social)
   ↓
Feature Engine (shared tabular features — team owned)
   ↓
Specialist Models (parallel, isolated, blind to each other)
   ├── Sentiment Specialist        [Pavani]
   ├── Trend Specialist            [Prapti]
   ├── Momentum Specialist         [Gayatri]
   ├── Volatility Specialist       [Aadya]
   ├── Mean Reversal Specialist    [Satakshi]
   └── Volume & Microstructure     [Simar]
   ↓
Regime Detector                   [You]    ← runs BEFORE aggregator
   ↓
Aggregator (regime-weighted ensemble of 6 signal contracts)
   ↓
Risk Engine (filter / size — final authority)
   ↓
Decision (BUY / SELL / HOLD)
   ↓
Execution (Backtrader — daily EOD rebalancing)
   ↓
Logger → Evaluation → Iteration
```

## Why Regime Runs Before Aggregator

The regime detector must run first so the aggregator knows what weights to apply to each specialist. If regime runs after, the aggregator is flying blind — it can't know whether to trust Prapti or Satakshi more right now. This order is non-negotiable.

## Design Rules

- **Specialists are isolated** — no specialist sees another's output. Ever.
- **Aggregator only sees structured signal contracts** — never raw data, never free text
- **Risk engine has final authority** — it can veto any decision regardless of score
- **Regime detector feeds the aggregator** — weights shift per detected regime
- **Daily EOD execution** — all signals computed after market close, orders placed for next open
- **Backtrader handles capital, position management, and trade execution**
- **Logger runs on every timestep, no exceptions**
- **TFT-GNN deferred to Phase 6** — tree-based models for all specialists now

---

# 4. Signal Contract (Evolving Schema)

Every specialist outputs a standardized signal contract. All specialists always output the same schema at any given phase.

## Phase 1–2 Contract (Foundation + Risk)
```json
{
  "specialist": "trend",
  "timestamp": "2026-04-25",
  "symbol": "RELIANCE.NS",
  "signal": -1 | 0 | 1,
  "confidence": 0.0–1.0,
  "strength": 0.0–1.0,
  "risk_score": 0.0–1.0
}
```

## Phase 3–4 Contract (ML + Regime)
```json
{
  "specialist": "trend",
  "timestamp": "2026-04-25",
  "symbol": "RELIANCE.NS",
  "signal": -1 | 0 | 1,
  "confidence": 0.0–1.0,
  "strength": 0.0–1.0,
  "risk_score": 0.0–1.0,
  "regime_fit": 0.0–1.0
}
```

## Phase 5+ Contract (Performance-Adaptive)
```json
{
  "specialist": "trend",
  "timestamp": "2026-04-25",
  "symbol": "RELIANCE.NS",
  "signal": -1 | 0 | 1,
  "confidence": 0.0–1.0,
  "strength": 0.0–1.0,
  "risk_score": 0.0–1.0,
  "regime_fit": 0.0–1.0,
  "expected_return": float,
  "uncertainty": 0.0–1.0
}
```

### Contract Rules
- Every specialist must always return a valid contract — never `None`, never partial
- If a specialist has no view: `signal=0, confidence=0, strength=0`
- `regime_fit` is injected by the aggregator after regime detection — specialists do not set it
- `metadata` can be added per specialist for debugging but is never used in aggregation

---

# 5. Specialist Model Design

## Design Philosophy

Each specialist has a **different lens** — not just a different algorithm on the same inputs. A specialist is defined by:
1. What features it sees (its input domain)
2. What question it is answering
3. What algorithm best answers that question
4. Who owns it end-to-end

Specialists do not share inputs with each other. They do not see each other's outputs.

---

## 5.1 Sentiment Specialist — Owner: Pavani

| Attribute | Specification |
|-----------|--------------|
| **Question** | What is the market narrative and sentiment for this specific stock right now? |
| **Market Context** | Indian equities — NSE listed stocks, sectoral indices, Hinglish language challenges |
| **Data Sources** | Economic Times headlines, Moneycontrol news, NSE corporate announcements, Twitter/X Indian finance accounts, Reddit r/IndianStreetBets / r/DalalStreetTalks |
| **Language Challenge** | Hinglish (Hindi-English code-switching), local sarcasm, meme-driven sentiment |
| **Inputs** | `sentiment_score`, `sentiment_volatility`, `news_volume`, `social_volume`, `hype_z_score`, `source_diversity`, `negative_keyword_ratio`, `promoter_sentiment_delta` |
| **Phase 1 Algorithm** | Rule-based keyword scoring + polarity lexicon (VADER or manual Indian finance keyword list) |
| **Phase 3 Algorithm** | **DistilRoBERTa** for social text + **FinBERT** for news headlines → embeddings fed into **XGBoost** classifier |
| **Why this model** | DistilRoBERTa empirically most robust across noisy social data. FinBERT tuned for financial news. XGBoost stable on tabular sentiment features. IndicBERT for Hinglish code-switching if needed. |
| **Primary output use** | Directional bias and regime context from narrative |
| **Python Stack** | `transformers` (DistilRoBERTa/FinBERT/IndicBERT), `torch`, `xgboost`, `praw`, `vaderSentiment` |
| **Special Notes** | Must handle Hinglish — route code-switched text to IndicBERT. LLM output must be parsed into signal contract. No free text reaches aggregator. High-impact events (RBI, Budget, elections) → force `risk_score = 0.9`. |

**Pavani's Development Track:**
- **Phase 1:** Keyword-based sentiment scoring + signal contract output
- **Phase 3:** DistilRoBERTa embeddings + XGBoost classifier
- **Phase 6:** Full LLM pipeline with structured output parser

---

## 5.2 Trend Specialist — Owner: Prapti

| Attribute | Specification |
|-----------|--------------|
| **Question** | Is there a clear directional trend and how strong is it? |
| **Market Context** | NSE daily data — ADX > 25 signals strong trend in Indian markets; trends persist 3–10 days in mid-caps |
| **Data Sources** | NSE OHLCV via `yfinance` / `nsepython` |
| **Inputs** | `SMA_5`, `SMA_20`, `SMA_50`, `EMA_12`, `EMA_26`, `ADX`, `ADX_DI_plus`, `ADX_DI_minus`, `price_vs_SMA20`, `price_vs_SMA50`, `Aroon_up`, `Aroon_down`, `trend_duration`, `higher_highs_count`, `lower_lows_count` |
| **Phase 1 Algorithm** | SMA crossover + ADX filter (rule-based) |
| **Phase 3 Algorithm** | **XGBoost** on trend features (primary) / **LightGBM** (alternative for speed) |
| **Why this model** | Trend features are structured/tabular. Tree models handle these better than neural nets. XGBoost hits 73% out-of-sample accuracy on financial time series. |
| **Primary output use** | Direction signal in trending regimes |
| **Python Stack** | `xgboost`, `lightgbm`, `pandas`, `numpy`, `pandas-ta` |
| **Special Notes** | Must label training data: trending (ADX>25 + sustained direction) vs non-trending. `regime_fit` high in trending regimes, low in choppy. Supertrend (7,3) as additional confirmation — widely followed on NSE, creates self-fulfilling properties. |

**Prapti's Development Track:**
- **Phase 1:** SMA crossover + ADX filter rule
- **Phase 3:** XGBoost trained on historical trend labels
- **Phase 6:** TFT upgrade for temporal attention on trend sequences

---

## 5.3 Momentum Specialist — Owner: Gayatri

| Attribute | Specification |
|-----------|--------------|
| **Question** | Is price accelerating or decelerating? |
| **Market Context** | Momentum persists in Indian mid-caps; fades in large-caps post-earnings; daily timeframe catches 3–7 day momentum bursts |
| **Data Sources** | NSE OHLCV + volume |
| **Inputs** | `momentum_5`, `momentum_10`, `momentum_20`, `RSI`, `RSI_divergence`, `rate_of_change`, `MACD`, `MACD_signal`, `MACD_hist`, `OBV`, `stochastic_k`, `stochastic_d`, `CCI`, `Williams_R`, `momentum_slope_change` |
| **Phase 1 Algorithm** | RSI + MACD threshold rules |
| **Phase 3 Algorithm** | **Random Forest** (primary) / **XGBoost** (secondary) |
| **Why this model** | Random Forest achieves strong monthly alpha on momentum portfolios. RF handles non-linear momentum interactions (RSI + MACD + OBV) better than linear models. |
| **Primary output use** | Confirmation or contradiction of trend signal |
| **Python Stack** | `scikit-learn` (RandomForest), `xgboost`, `pandas`, `numpy`, `pandas-ta` |
| **Special Notes** | Key challenge: momentum divergences (price makes new high, RSI doesn't) — build divergence detection as a feature. Indian stocks sustain RSI 75–85 in bull phases — calibrate thresholds on NSE data, not US defaults. |

**Gayatri's Development Track:**
- **Phase 1:** RSI + MACD threshold rules
- **Phase 3:** Random Forest classifier trained on momentum labels
- **Phase 6:** TFT upgrade for momentum sequence modeling

---

## 5.4 Volatility Specialist — Owner: Aadya

| Attribute | Specification |
|-----------|--------------|
| **Question** | Is the market in a high-risk or low-risk state right now? |
| **Market Context** | India VIX is the key reference; volatility clusters around budget, RBI policy, earnings seasons |
| **Data Sources** | NSE OHLCV, India VIX, F&O data (if available) |
| **Inputs** | `std_dev_10`, `std_dev_20`, `ATR`, `ATR_ratio`, `BB_width`, `BB_width_change`, `volume_z_score`, `India_VIX_level`, `VIX_change`, `parkinson_volatility`, `garman_klass_volatility`, `volatility_regime_flag` |
| **Phase 1 Algorithm** | Rule-based: ATR thresholds + volume z-score + India VIX level |
| **Phase 3 Algorithm** | **Isolation Forest** (anomaly detection) |
| **Phase 5 Algorithm** | Isolation Forest + **GRU** (volatility spillover prediction) |
| **Why this model** | Isolation Forest detects anomalous volatility spikes — ideal for risk-state classification. GRU outperforms LSTM for volatility clustering prediction. |
| **Primary output use** | Risk gating — high volatility suppresses position sizing or vetoes trades entirely |
| **Python Stack** | `scikit-learn` (IsolationForest), `torch` (GRU), `pandas`, `numpy`, `pandas-ta` |
| **Special Notes** | This specialist has **special authority** — its `risk_score` feeds directly into the Risk Engine. Must generate `risk_score` from Phase 1 onwards. Calendar flags: RBI MPC meetings, Budget day, expiry week, election results → automatic `risk_score` floor of 0.85. |

**Aadya's Development Track:**
- **Phase 1:** Rule-based ATR + VIX + volume thresholds + risk_score
- **Phase 3:** Isolation Forest for volatility anomaly detection
- **Phase 5:** GRU layer added for volatility spillover prediction

---

## 5.5 Mean Reversal Specialist — Owner: Satakshi

| Attribute | Specification |
|-----------|--------------|
| **Question** | Is price extended from its mean and likely to revert? |
| **Market Context** | Indian markets exhibit strong mean-reversion in choppy periods (post-budget consolidation, sideways Nifty phases); daily timeframe captures 2–5 day reversion windows |
| **Data Sources** | NSE OHLCV |
| **Inputs** | `BB_position`, `BB_width`, `price_vs_SMA50`, `price_vs_SMA200`, `RSI_14`, `RSI_extreme` (>70 or <30), `z_score_20`, `z_score_50`, `distance_to_pivot`, `support_distance`, `resistance_distance`, `reversion_velocity`, `mean_cross_count`, `consecutive_closes_above_bb` |
| **Phase 1 Algorithm** | Bollinger Band mean-reversion + RSI extreme rules |
| **Phase 3 Algorithm** | **XGBoost** / **LightGBM** ensemble |
| **Why this model** | Mean reversion is a structured/tabular prediction problem. XGBoost captures non-linear thresholds (e.g., RSI<20 + 3-sigma move = stronger reversal than either alone). |
| **Primary output use** | Primary signal in choppy/mean-reverting regimes; suppressed in trending |
| **Python Stack** | `xgboost`, `lightgbm`, `pandas`, `numpy`, `pandas-ta`, `scikit-learn` |
| **Special Notes** | This specialist is the **counterweight to Prapti's Trend Specialist** — they will often disagree, and the Aggregator resolves this via regime weighting. `regime_fit` HIGH in choppy markets, LOW in strong trends. |

**Satakshi's Development Track:**
- **Phase 1:** Bollinger Band mean-reversion + RSI extreme rules
- **Phase 3:** XGBoost trained on mean-reversion labels
- **Phase 6:** TFT upgrade for temporal mean-reversion patterns

---

## 5.6 Volume & Microstructure Specialist — Owner: Simar

| Attribute | Specification |
|-----------|--------------|
| **Question** | What is the volume profile and order flow telling us about conviction? |
| **Market Context** | Indian markets: delivery % vs trading % matters; bulk deals, block deals, promoter buying are signals; FII/DII flow data published daily by NSE |
| **Data Sources** | NSE OHLCV + volume, NSE bulk/block deals, FII/DII daily flow data, delivery percentage |
| **Inputs** | `volume_z_score`, `volume_ratio`, `OBV`, `OBV_slope`, `VWAP_distance`, `A/D_line`, `MFI`, `relative_volume`, `delivery_percentage`, `volume_trend_divergence`, `FII_flow_z_score`, `DII_flow_z_score`, `bulk_deal_flag`, `block_deal_flag`, `promoter_buying_flag` |
| **Phase 1 Algorithm** | Volume confirmation rules + delivery % thresholds + FII/DII direction flags |
| **Phase 3 Algorithm** | **Random Forest** / **XGBoost** ensemble |
| **Why this model** | Volume features are noisy but predictive when combined. Random Forest handles noise robustly. XGBoost captures interactions like "high volume + negative FII flow = distribution." |
| **Primary output use** | Conviction scoring — validates or invalidates other specialists' signals |
| **Python Stack** | `scikit-learn`, `xgboost`, `pandas`, `numpy`, `pandas-ta` |
| **Special Notes** | Delivery percentage and FII/DII flow are free and published daily by NSE — among the most underused signals in Indian systematic strategies. This specialist acts as a **conviction layer** — low volume should suppress other signals. |

**Simar's Development Track:**
- **Phase 1:** Volume confirmation + delivery % + FII/DII direction rules
- **Phase 3:** RF/XGBoost ensemble on volume profile features
- **Phase 6:** TFT-GNN upgrade (this specialist benefits most from GNN's relational data)

---

# 6. Regime Detection — Owner: You

| Attribute | Specification |
|-----------|--------------|
| **Question** | What environment are we in right now? |
| **Algorithm** | **Hidden Markov Model (HMM)** — GaussianHMM via `hmmlearn` |
| **Why this model** | HMM is the gold standard for regime detection. Unlike threshold rules (ADX>25), HMM uses full distributional information and accounts for temporal persistence. Viterbi algorithm finds the most likely state sequence. |
| **Alternative** | Gaussian Mixture Model (GMM) for simpler non-sequential clustering if HMM proves unstable |
| **States** | `trending_up`, `trending_down`, `choppy`, `volatile`, `breakout` |
| **Inputs** | `ADX`, `ATR_zscore`, `BB_width`, `price_range_ratio`, `volatility_specialist_risk_score`, `trend_specialist_strength` |
| **Output** | Current regime + regime probability vector |
| **Python Stack** | `hmmlearn` (GaussianHMM), `scikit-learn` (GMM fallback), `pandas`, `numpy` |
| **Phase 1** | Stubbed — returns `None`, aggregator uses static equal weights |
| **Phase 4** | Active GaussianHMM with 5 states, feeds regime-weighted aggregator |
| **Phase 5** | HMM + performance-adaptive regime weight updates |

**Your Development Track:**
- **Phase 1:** Stubbed regime detector (static equal weights)
- **Phase 4:** GaussianHMM active with 5 states
- **Phase 5:** HMM + rolling performance-based weight updates

---

# 7. Phase Roadmap

---

## Phase 1 — Foundation
**Goal:** End-to-end pipeline that runs, trades, and logs correctly on daily NSE data.

**Deliverables:**
- [ ] Data layer: NSE OHLCV daily, clean, aligned (yfinance)
- [ ] India VIX, FII/DII, delivery % data pipelines
- [ ] Feature engine: all 6 specialist feature sets computed (`features.py`)
- [ ] `base_specialist.py` written, reviewed, shared with all team members
- [ ] Pavani: keyword-based sentiment scoring + signal contract output
- [ ] Prapti: SMA crossover + ADX filter
- [ ] Gayatri: RSI + MACD threshold rules
- [ ] Aadya: ATR + VIX + volume thresholds + `risk_score`
- [ ] Satakshi: Bollinger Band mean-reversion + RSI extreme rules
- [ ] Simar: volume confirmation + delivery % + FII/DII direction rules
- [ ] Regime detector: stubbed (returns `None`)
- [ ] Aggregator: equal-weight sum of 6 signals + risk veto
- [ ] Backtrader: daily EOD rebalancing + stop-loss
- [ ] Logger: full per-bar schema
- [ ] End-to-end backtest: 3+ NSE stocks, 2 years data, no errors

**Success Criteria:**
- System runs end-to-end on NSE daily data
- Trades execute correctly at next-day open
- Every decision fully explainable from logs
- All 6 specialist unit tests passing

---

## Phase 2 — Risk Layer
**Goal:** System makes risk-aware decisions, not just signal-aware ones.

**Deliverables:**
- [ ] `risk_score` active in all signal contracts
- [ ] Risk engine: position sizing logic (`position_size = base_size * (1 - risk_score) * confidence`)
- [ ] Aggregator: risk veto layer (halt if `vol_risk_score > 0.8`)
- [ ] Trade filtering: no new positions in extreme volatility
- [ ] Stop-loss enforcement in Backtrader
- [ ] Drawdown circuit breaker: halt at 15% portfolio drawdown
- [ ] Max 5 open positions enforced

---

## Phase 3 — Machine Learning
**Goal:** Every specialist graduates from rules to a trained ML model.

**Deliverables:**
- [ ] Training data pipelines + label generation per specialist
- [ ] Pavani: DistilRoBERTa + FinBERT embeddings → XGBoost
- [ ] Prapti: XGBoost on trend features
- [ ] Gayatri: Random Forest on momentum features
- [ ] Aadya: Isolation Forest for volatility anomaly detection
- [ ] Satakshi: XGBoost/LightGBM on mean-reversion features
- [ ] Simar: RF/XGBoost ensemble on volume features
- [ ] Model persistence: `save_model()` / `load_model()` per specialist
- [ ] Confidence calibration: Platt scaling or isotonic regression
- [ ] A/B backtest: Phase 2 rule-based vs Phase 3 ML — must show improvement

---

## Phase 4 — Regime Intelligence
**Goal:** System knows what environment it's in and weights specialists accordingly.

**Deliverables:**
- [ ] Regime detector: GaussianHMM with 5 states (You)
- [ ] `regime_fit` added to signal contract
- [ ] Aggregator: regime-weighted scoring using fit matrix
- [ ] Per-specialist rolling accuracy tracking
- [ ] Per-specialist PnL attribution
- [ ] Weight engine: `f(regime, base_weight)`

---

## Phase 5 — Performance-Adaptive
**Goal:** Weights evolve based on real performance, not just hand-tuned values.

**Deliverables:**
- [ ] `expected_return` + `uncertainty` added to signal contract
- [ ] Rolling accuracy and PnL contribution per specialist computed weekly
- [ ] Weight engine: `f(regime_weight, rolling_accuracy_20d, pnl_contribution_60d)`
- [ ] Walk-forward validation framework
- [ ] Automatic model retraining triggers

---

## Phase 6 — Advanced Architectures
**Goal:** TFT-GNN integration where it adds genuine value.

**Deliverables:**
- [ ] TFT for Prapti (trend temporal attention)
- [ ] TFT for Gayatri (momentum sequence modeling)
- [ ] TFT-GNN for Simar (relational stock volume data)
- [ ] Pavani: Full LLM pipeline with structured output parser
- [ ] Aggregator: source-aware weighting (`source_type: ml | llm`)

---

# 8. Aggregator Design (Phase-by-Phase)

## Phase 1 — Equal Weight Sum
```python
final_score = sum(signal * confidence * strength for each specialist)

if final_score > BUY_THRESHOLD:   → BUY
if final_score < SELL_THRESHOLD:  → SELL
else:                             → HOLD
```

## Phase 2 — Risk-Filtered
```python
if volatility_specialist.risk_score > 0.8:
    → HOLD  # veto regardless of score

if mean(all_risk_scores) > RISK_VETO_THRESHOLD:
    → HOLD
```

## Phase 4 — Regime-Weighted
```python
# Regime runs first, then aggregator applies weights
current_regime = regime_detector.detect()

effective_weight = base_weight * REGIME_FIT_MATRIX[specialist][current_regime]
final_score = sum(signal * confidence * strength * effective_weight)
```

## Phase 5 — Performance-Adaptive
```python
# Weights updated weekly, not every bar
effective_weight = f(regime_weight, rolling_accuracy_20d, pnl_contribution_60d)
final_score = sum(signal * confidence * strength * effective_weight)
```

---

# 9. Regime-to-Specialist Weight Matrix

Your HMM outputs the current regime. The Aggregator uses this matrix to weight each specialist. These are starting (hand-tuned) values — Phase 5 derives them from performance data.

| Regime | Pavani (Sentiment) | Prapti (Trend) | Gayatri (Momentum) | Aadya (Volatility) | Satakshi (MeanRev) | Simar (Volume) |
|--------|-------------------|----------------|-------------------|-------------------|-------------------|----------------|
| **trending_up** | 1.0 | **1.5** | **1.2** | 0.5 | 0.3 | 1.0 |
| **trending_down** | 1.0 | **1.5** | **1.2** | 0.5 | 0.3 | 1.0 |
| **choppy** | 1.0 | 0.3 | 0.5 | 0.8 | **1.4** | **1.2** |
| **volatile** | 0.8 | 0.5 | 0.3 | **1.5** | 0.4 | 0.8 |
| **breakout** | **1.3** | **1.2** | **1.4** | 0.6 | 0.2 | **1.3** |

*(Phase 1–3: static equal weights. Phase 4: this matrix activates. Phase 5: performance-derived.)*

---

# 10. Risk Engine

Sits between aggregator and execution. Has final authority. Always runs.

## Phase 1 — Basic Rules
- Max position size: 5% of portfolio per trade
- Max open exposure: 30% of total capital
- Max 5 open positions at any time
- No trade if Aadya's `risk_score > 0.8`

## Phase 2 — Signal-Aware Sizing
```python
position_size = base_size * (1 - risk_score) * confidence
```

## Phase 4 — Regime-Aware Sizing
```python
position_size = base_size * regime_multiplier * (1 - risk_score) * confidence
```

## Regime Multipliers
| Regime | Multiplier |
|---|---|
| trending_up / trending_down | 1.0 |
| breakout | 0.8 |
| choppy | 0.6 |
| volatile | 0.3 |

## Hard Rules (All Phases)
- Stop-loss mandatory on every trade (Backtrader enforced)
- Drawdown circuit breaker: halt all new trades if portfolio drawdown > 15%
- If India VIX > 25: no new positions opened
- Never size up in volatile regime
- Never average down into a losing position

---

# 11. Data Layer

## Primary Sources

| Data | Source | Access | Used By |
|------|--------|--------|---------|
| NSE OHLCV Daily | `yfinance` / `nsepython` | Python library | All specialists |
| India VIX | NSE India website | `nsepython` | Aadya |
| FII/DII flows | NSE daily reports | `nsepython` | Simar |
| Delivery percentage | NSE bhavcopy | `nsepython` | Simar |
| Bulk/Block deals | NSE daily files | Direct download | Simar |
| Corporate announcements | NSE/BSE filings | `nsepython` | Pavani |
| Economic Times | economictimes.indiatimes.com | RSS / scraper | Pavani |
| Moneycontrol | moneycontrol.com | RSS / scraper | Pavani |
| Reddit | PRAW | `praw` library | Pavani |

**Not Alpha Vantage for sentiment** — Indian stock coverage is too thin and pre-scored output removes Pavani's ability to reason over raw text. Alpha Vantage is acceptable for OHLCV backup only.

## Feature Engine (`system/features.py`) — Team Owned

Changes to `features.py` require team review. It affects everyone.

```python
FEATURE_REGISTRY = {
    # --- Pavani (Sentiment) ---
    "sentiment_score", "sentiment_volatility", "news_volume",
    "social_volume", "hype_z_score", "source_diversity",
    "negative_keyword_ratio", "promoter_sentiment_delta",

    # --- Prapti (Trend) ---
    "SMA_5", "SMA_20", "SMA_50", "EMA_12", "EMA_26",
    "ADX", "ADX_DI_plus", "ADX_DI_minus",
    "price_vs_SMA20", "price_vs_SMA50",
    "Aroon_up", "Aroon_down", "trend_duration",
    "higher_highs_count", "lower_lows_count",

    # --- Gayatri (Momentum) ---
    "momentum_5", "momentum_10", "momentum_20",
    "RSI", "RSI_divergence", "rate_of_change",
    "MACD", "MACD_signal", "MACD_hist",
    "OBV", "stochastic_k", "stochastic_d",
    "CCI", "Williams_R", "momentum_slope_change",

    # --- Aadya (Volatility) ---
    "std_dev_10", "std_dev_20", "ATR", "ATR_ratio",
    "BB_width", "BB_width_change",
    "volume_z_score", "India_VIX_level", "VIX_change",
    "parkinson_volatility", "garman_klass_volatility",
    "volatility_regime_flag",

    # --- Satakshi (Mean Reversal) ---
    "BB_position", "z_score_20", "z_score_50",
    "RSI_extreme", "mean_cross_count",
    "price_vs_SMA50", "price_vs_SMA200",
    "support_distance", "resistance_distance",
    "reversion_velocity", "consecutive_closes_above_bb",

    # --- Simar (Volume & Microstructure) ---
    "volume_ratio", "relative_volume", "VWAP_distance",
    "A/D_line", "MFI", "delivery_percentage",
    "volume_trend_divergence", "FII_flow_z_score",
    "DII_flow_z_score", "bulk_deal_flag",
    "block_deal_flag", "promoter_buying_flag"
}
```

---

# 12. File Structure

```
project/
│
├── data/
│   ├── raw/                    # yfinance/nsepython downloads
│   ├── processed/              # cleaned, aligned features
│   └── sentiment/              # Pavani's text corpus
│
├── system/
│   ├── core.py                 # Pipeline orchestration (You)
│   ├── features.py             # Shared feature engine (TEAM OWNED)
│   ├── regime.py               # Regime detector (You)
│   ├── aggregator.py           # Phase-by-phase aggregator logic (You)
│   ├── risk_engine.py          # Risk rules and position sizing (You)
│   ├── logger.py               # Full timestep + attribution logging (You)
│   │
│   └── models/
│       ├── base_specialist.py              # Abstract base class (You)
│       ├── sentiment_specialist.py         # Pavani
│       ├── trend_specialist.py             # Prapti
│       ├── momentum_specialist.py          # Gayatri
│       ├── volatility_specialist.py        # Aadya
│       ├── mean_reversal_specialist.py     # Satakshi
│       └── volume_microstructure_specialist.py  # Simar
│
├── strategies/
│   └── main_strategy.py         # Backtrader strategy wrapper (You)
│
├── evaluation/
│   ├── backtest_runner.py
│   ├── attribution.py           # Per-specialist PnL analysis
│   └── reports/
│
├── config/
│   ├── phase1_config.yaml
│   ├── phase2_config.yaml
│   ├── phase3_config.yaml
│   └── phase4_config.yaml
│
├── logs/
│   └── (auto-generated, append-only)
│
├── tests/
│   ├── test_features.py
│   ├── test_sentiment.py        # Pavani
│   ├── test_trend.py            # Prapti
│   ├── test_momentum.py         # Gayatri
│   ├── test_volatility.py       # Aadya
│   ├── test_mean_reversal.py    # Satakshi
│   ├── test_volume_micro.py     # Simar
│   ├── test_regime.py           # You
│   ├── test_aggregator.py       # You
│   └── test_risk_engine.py      # You
│
├── requirements.txt
└── main.py
```

---

# 13. Base Specialist Contract

Every specialist **must** inherit from `BaseSpecialist`. No exceptions.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import pandas as pd

@dataclass
class SignalContract:
    specialist: str
    timestamp: str
    symbol: str
    signal: int           # -1, 0, 1
    confidence: float     # 0.0–1.0
    strength: float       # 0.0–1.0
    risk_score: float     # 0.0–1.0
    regime_fit: float = 0.0       # injected by aggregator — do not set
    expected_return: float = 0.0  # Phase 5+
    uncertainty: float = 0.0      # Phase 5+
    metadata: dict = field(default_factory=dict)  # debug only, never aggregated


class BaseSpecialist(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def compute_features(self, data: dict) -> dict:
        """Compute this specialist's features from raw data."""
        pass

    @abstractmethod
    def generate_signal(self, features: dict) -> SignalContract:
        """Generate signal contract from features."""
        pass

    def safe_generate(self, data: dict) -> SignalContract:
        """
        Always returns a valid contract. Never crashes the pipeline.
        Pipeline always calls this — never generate_signal() directly.
        """
        try:
            features = self.compute_features(data)
            contract = self.generate_signal(features)
            self._validate(contract)
            return contract
        except Exception as e:
            return SignalContract(
                specialist=self.name,
                timestamp=data.get("timestamp", ""),
                symbol=data.get("symbol", ""),
                signal=0,
                confidence=0.0,
                strength=0.0,
                risk_score=0.5,
                metadata={"error": str(e), "fallback": True}
            )

    def _validate(self, contract: SignalContract):
        assert contract.signal in (-1, 0, 1), "signal must be -1, 0, or 1"
        assert 0.0 <= contract.confidence <= 1.0, "confidence out of range"
        assert 0.0 <= contract.strength <= 1.0, "strength out of range"
        assert 0.0 <= contract.risk_score <= 1.0, "risk_score out of range"

    # Phase 3+ only — rule-based specialists leave these as pass
    def train(self, data: pd.DataFrame) -> None:
        pass

    def save_model(self, path: str) -> None:
        pass

    def load_model(self, path: str) -> None:
        pass
```

**Critical:** Pipeline always calls `safe_generate()` — never `generate_signal()` directly. One broken specialist must never crash the whole pipeline.

---

# 14. Logging Schema

Every EOD bar, log everything. Append-only.

```json
{
  "date": "YYYY-MM-DD",
  "symbol": "RELIANCE.NS",
  "price_close": 1450.25,
  "regime": "trending_up | trending_down | choppy | volatile | breakout",
  "india_vix": 14.2,
  "specialist_outputs": {
    "sentiment":      { "signal": 1, "confidence": 0.7, "strength": 0.6, "risk_score": 0.2 },
    "trend":          { "signal": 1, "confidence": 0.85, "strength": 0.8, "risk_score": 0.1 },
    "momentum":       { "signal": 1, "confidence": 0.75, "strength": 0.7, "risk_score": 0.15 },
    "volatility":     { "signal": 0, "confidence": 0.9, "strength": 0.6, "risk_score": 0.3 },
    "mean_reversal":  { "signal": -1, "confidence": 0.5, "strength": 0.4, "risk_score": 0.2 },
    "volume_micro":   { "signal": 1, "confidence": 0.8, "strength": 0.75, "risk_score": 0.1 }
  },
  "regime_fit_applied": {
    "sentiment": 1.0, "trend": 1.5, "momentum": 1.2,
    "volatility": 0.5, "mean_reversal": 0.3, "volume_micro": 1.0
  },
  "aggregator": {
    "raw_score": 2.85,
    "decision": "BUY",
    "risk_vetoed": false,
    "veto_reason": null
  },
  "risk_engine": {
    "position_size_pct": 4.2,
    "regime_multiplier": 1.0,
    "circuit_breaker_active": false
  },
  "execution": {
    "trade_executed": true,
    "entry_price": 1452.0,
    "stop_loss": 1380.4
  }
}
```

## Attribution Log (Per Closed Trade)
For every closed trade, compute counterfactual PnL per specialist:
- "If only Prapti's signal drove all decisions, what would the outcome have been?"
- This data tunes the regime fit matrix in Phase 4 and adaptive weights in Phase 5.

---

# 15. Development Rules (Team Contract)

1. Every specialist inherits from `BaseSpecialist` — no exceptions
2. Pipeline always calls `safe_generate()` — never `generate_signal()` directly
3. Signal contract fields are frozen — no additions without team agreement
4. Never mix system logic with Backtrader logic
5. Iterate one phase at a time — Phase 1 must run before Phase 3 ML begins
6. Regime detector always runs — even if returning `None` in Phase 1
7. Risk engine always runs — it is never bypassed
8. Logs are append-only — never overwrite a run
9. Every specialist owner writes their own unit tests
10. Integration is done by You only — specialists touch only their own file
11. No specialist imports another specialist's module
12. Changes to `features.py` require team review — it affects everyone
13. If a decision can't be explained from the logs, the system has failed
14. TFT-GNN is deferred to Phase 6 — no exceptions

---

# 16. Evaluation Framework

## Backtest Metrics Per Run

| Metric | Why |
|---|---|
| Total return | Baseline |
| Sharpe ratio | Risk-adjusted return |
| Sortino ratio | Downside risk specifically |
| Max drawdown | Worst-case loss |
| Win rate | Signal quality proxy |
| Avg win / avg loss | Reward-risk per trade |
| Trade frequency | System activity |
| Regime-conditional return | Does system work in the right conditions |
| Daily turnover | EOD rebalancing cost check |

## Per-Specialist Attribution
After every backtest, compute for each specialist:
- Counterfactual return if only this specialist drove decisions
- Regime-conditional win rate
- Contribution score: did this specialist improve or hurt the final decision

## Walk-Forward Validation (Phase 5+)
- Train on period A → test on period B → roll forward
- Minimum 6-month test window
- Log regime distribution of each test period
- Indian market-specific: separate train/test around budget, RBI policy, expiry

---

# 17. Weekend Sprint Plan

This weekend targets Phase 1 only. Phase 3 ML is a separate week minimum — do not start ML before the rule-based system is validated end-to-end.

| Time | Target | Owner | Done |
|------|--------|-------|------|
| **Day 0 (now)** | GitHub repo + `base_specialist.py` + `features.py` skeleton + NSE sample data | You | [ ] |
| **Day 1 morning** | Each specialist: features computed, Phase 1 rule-based logic written | Pavani, Prapti, Gayatri, Aadya, Satakshi, Simar | [ ] |
| **Day 1 afternoon** | Aggregator + Backtrader + Logger running end-to-end on 1 stock | You | [ ] |
| **Day 1 evening** | All 6 specialist unit tests passing | Everyone | [ ] |
| **Day 2 morning** | Full backtest on 3+ NSE stocks, 2 years data — no errors | Team | [ ] |
| **Day 2 afternoon** | Phase 2 Risk: risk scores + position sizing + stop-loss + circuit breaker | You + Aadya | [ ] |
| **Day 2 evening** | Attribution log running, every decision explainable from logs | Team | [ ] |

---

# 18. Long-Term Edge

The edge does not come from smarter models. It comes from:

- **6 specialists with genuinely differentiated views** — Pavani, Prapti, Gayatri, Aadya, Satakshi, Simar will disagree, and that's the point
- **A regime detector that runs first** — so the aggregator always knows whose opinion to trust right now
- **A risk engine that has final authority** — preventing catastrophic loss regardless of signal strength
- **India-specific data** (VIX, delivery %, FII/DII flow) that most systematic strategies ignore
- **Attribution logging** that makes the system measurably better after every backtest
- **Discipline to not add complexity before it's earned** — TFT-GNN waits until Phase 6

The system is not built to predict perfectly or win every trade. It is built to make consistent, explainable, risk-controlled decisions — and get measurably better over time.