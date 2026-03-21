# Astar Island — Long-Term Improvement Plan

**Created:** 2026-03-21 ~15:00 CET
**Competition ends:** 2026-03-22 15:00 CET (~24 hours remaining)
**Current status:** Rank 225/244, weighted_score=103.69, best round score ~39 (Round 11)
**Target:** Maximize weighted_score = round_score × 1.05^round_number

---

## Diagnosis: Why We Score Poorly

Our Round 14 analysis revealed the core problem:

| Metric | Us | Top Teams (est.) |
|---|---|---|
| Weighted KL | 0.786 | ~0.05 |
| Score | 9.85 | ~85 |
| vs Uniform baseline | 21% better | ~85% better |

**Root cause: Each API query returns ONE stochastic simulation result, but the ground truth is a PROBABILITY DISTRIBUTION from hundreds of simulations.** We treat each observation as near-certain truth (93% confidence), when the ground truth for that same cell might be 40% Empty, 25% Settlement, 25% Ruin, 10% Forest.

### What we're doing wrong (by impact):

1. **Single-observation overconfidence** (HUGE): Seeing Settlement once → predicting 93% Settlement. But GT might be 40% Settlement. This is the #1 source of KL loss.
2. **Not using cross-query empirical distributions** (LARGE): We run 10 queries per seed (9 base + 1 extra). Each returns a DIFFERENT stochastic outcome. If we observe the same cell 3 times and see Settlement/Settlement/Empty, that's a 67/33 empirical distribution — far better than 93% for any class.
3. **MC simulator inaccuracy** (MEDIUM): Our simulator doesn't match the real one precisely. Default params differ from hidden params. MC predictions add noise rather than signal for cells near settlements.
4. **Prior distributions are hand-tuned guesses** (MEDIUM): The terrain-aware priors in `prediction_builder.py` are reasonable but not calibrated against actual ground truth data.

---

## Improvement Tiers (Prioritized by Expected Impact)

### TIER 1: Fix Observation-Based Predictions (Expected: +20-40 points)

**This is the single biggest win. Everything else is secondary.**

#### 1A. Use Multiple Observations as Empirical Distributions

**Current behavior:** Each query gives one simulation result per cell. With 10 queries per seed and overlapping viewports, some cells are observed multiple times. The `ObservationStore` already tracks `class_counts[seed, y, x, class]` — but `prediction_builder.py` treats them as "all observations agree" rather than building empirical distributions.

**Fix:** When a cell has N observations, use `class_counts / N` directly as the base probability distribution, blended with a weak prior. For cells observed only once, use a MUCH softer confidence (e.g., 0.4 for observed class instead of 0.93).

**Why this works:** The competition's ground truth IS an empirical distribution from many simulations. Our observations are samples from that same process. More samples → better estimate.

**Implementation:**
- In `prediction_builder.py` `_fill_observed_cells()`: Replace the current Dirichlet posterior with direct empirical distribution when obs_count > 1
- For obs_count == 1: Use softer confidence (0.35-0.50 for the observed class, spread rest across plausible alternatives)
- For obs_count >= 3: Trust the empirical distribution heavily (weight 0.8-0.9)
- Key insight: even obs_count=1 should be much less confident than current 0.93

**Files:** `prediction_builder.py` (lines 77-104)

#### 1B. Maximize Overlapping Observations

**Current behavior:** 9 base queries per seed tile the full map (3×3 grid of 15×15). Extra queries target high-interest areas. With 50 total / 5 seeds = 10 per seed, we only get 1 extra.

**Fix:** Redesign query strategy to maximize cell observation count for high-entropy areas:
- Keep 9 queries for full coverage (non-negotiable — we need to see every cell at least once)
- Use remaining 1 query per seed to re-observe the highest-uncertainty region
- Consider reducing to 8 base queries (with 13×13 overlap tiling) to free up 5 extra adaptive queries across all seeds

**Better idea:** Cross-seed observation sharing. All 5 seeds use the SAME hidden params — meaning the probability distributions should be identical for all seeds (only the sim_seed stochastic noise differs). Observations from seed 0 query 1 inform seed 1's predictions.

**Implementation (cross-seed):**
- After all observations are collected, create an "aggregate store" that combines observations across all 5 seeds for the same cell positions
- Each cell gets up to 50 observations (10 per seed × 5 seeds) instead of just 10
- Use this aggregate for probability estimation

**Files:** `observation_store.py` (add `aggregate_across_seeds()` method), `prediction_builder.py`, `main.py`

#### 1C. Soften ALL Probability Distributions

**Current behavior:** Observed cells get 93% for observed class. This is catastrophic when wrong (KL divergence is very high for peaked wrong predictions).

**Fix:** Apply a "temperature" or "softening" to all predictions:
- Even with 100% empirical agreement, cap at ~70% confidence (because GT rarely exceeds 70% for any class in uncertain areas)
- Use entropy-adaptive softening: high-entropy GT cells need softer predictions
- Increase PROBABILITY_FLOOR from 0.01 to 0.02-0.03 for non-static cells

**Implementation:** Add a temperature parameter to `_apply_floor_and_normalize()` that caps the maximum probability for any class based on the cell's expected variability (estimated from settlement distance).

**Files:** `prediction_builder.py`, `config.py`

---

### TIER 2: Leverage Ground Truth Data from Past Rounds (Expected: +5-15 points)

#### 2A. Build Calibration Dataset from Rounds 11, 13, 14

We have ground truth (via `/analysis/{round_id}/{seed_index}`) for completed rounds. This gives us exact probability distributions for every cell, keyed by initial terrain and settlement distance.

**Build a lookup table:**
- Key: (initial_terrain, settlement_distance_bucket, neighbor_settlement_count)
- Value: average probability distribution [P(empty), P(settlement), P(port), P(ruin), P(forest), P(mountain)]
- This directly tells us "a plains cell 3 tiles from a settlement is typically 40% empty, 20% settlement, 15% ruin, ..."

**Implementation:**
1. Create `calibration.py` — fetches GT for past rounds, builds lookup table
2. Modify `prediction_builder.py` to use lookup table as the prior instead of hand-tuned values
3. Store calibration data in `.calibration/` directory

**Files:** New `calibration.py`, modify `prediction_builder.py`

#### 2B. Validate Each Change Against Past Rounds

After downloading GT for Rounds 11, 13, 14:
1. Generate predictions using the OLD code
2. Generate predictions using the NEW code  
3. Compute score for both against GT
4. Only deploy changes that improve score on at least 2 of 3 rounds

**Implementation:** Create `backtest.py` that:
- Loads saved observations for past rounds
- Generates predictions with current code
- Scores against GT
- Reports per-round, per-seed, and per-class breakdown

**Files:** New `backtest.py`

---

### TIER 3: Monte Carlo Improvements (Expected: +2-8 points)

#### 3A. Parameter Fitting from Observations

The existing CMA-ES fitter (`param_fitter.py`) works but is expensive. Key improvement: fit params BEFORE MC runs, not after.

**Quick win:** Use observation settlement stats (population, food, wealth, defense) to estimate hidden params directly (no optimization needed):
- If observed settlements have high population → growth_rate is probably above default
- If many ruins → winter severity is probably high
- These heuristics can set better starting params for MC

**Files:** `world_builder.py` (improve `calibrate_settlements_from_observations`), `param_fitter.py`

#### 3B. Increase MC Runs for Better Convergence

Currently using 400 MC runs. With param_noise=0.15, many runs use wildly different params. Focus MC budget:
- If params are fitted: 400 runs, noise=0.0 (trust the fit)
- If params are default: 200 runs, noise=0.3 (explore broadly)
- Key: MC is only useful for unobserved cells. With full coverage, MC matters less.

**Files:** `main.py` (adjust mc-runs and param-noise defaults)

---

### TIER 4: Query Strategy Refinements (Expected: +1-3 points)

#### 4A. Smarter Extra Query Placement

Current extra queries target "high interest" areas. Better: target areas where our single observation showed a class that's surprising given the initial terrain (e.g., settlement appeared on forest far from other settlements → high uncertainty → observe again).

**Files:** `query_strategy.py`

#### 4B. Consider Sacrificing One Seed's Coverage

Radical idea: Use 4 seeds with 12 queries each (48 total) for better overlap coverage, and 1 seed with only 2 queries. The 4 well-observed seeds will score much higher, and the 5th seed gets MC-only predictions.

Risk: The 5th seed drags down the average. Only worth it if the improvement on 4 seeds exceeds the loss on 1.

**Files:** `query_strategy.py`, `main.py`

---

## Verification Framework

### Local Testing

```bash
# Quick sanity check (~30 seconds)
python local_test.py --quick --detailed

# Full local benchmark (~5 minutes)  
python local_test.py --gt-runs 500 --mc-runs 400 --detailed

# With parameter fitting (~10 minutes)
python local_test.py --fit --fit-gens 25 --detailed
```

**Baseline scores to beat (local_test, map_seed=42):**
- Uniform: ~21
- Initial terrain: ~35
- MC (default params): ~45
- MC + Obs overlap: ~70
- Oracle (true params): ~99

### Backtesting Against Past Rounds

```bash
# Score our current code against Round 14 GT
python backtest.py --round 14

# Compare two strategies
python backtest.py --round 14 --compare old new

# Score against all available rounds
python backtest.py --all
```

### Per-Round Verification Protocol

For EVERY code change:

1. **Local test first:** `python local_test.py --quick --detailed`
   - Must not regress from previous local score
   - Check per-class and per-distance breakdown for regressions

2. **Backtest against past rounds:** `python backtest.py --all`
   - Must improve score on at least 2 of 3 available rounds
   - Pay special attention to cells near settlements (highest entropy weight)

3. **Deploy to competition:** Submit to active round
   - After round completes, run analysis: `python main.py --analyze-round <id>`
   - Compare actual score to backtest prediction

### Score Tracking Log

Maintain a running log of changes and their measured impact:

```
| Change | Local Score | R11 Backtest | R13 Backtest | R14 Backtest | Live Score |
|--------|-------------|-------------|-------------|-------------|------------|
| Baseline (pre-fixes) | ? | 39.12 | 23.82 | 9.85 | 9.85 |
| Bug fixes (R15) | ? | ? | ? | ? | pending |
| Tier 1A: empirical obs | ? | ? | ? | ? | ? |
| Tier 1B: cross-seed | ? | ? | ? | ? | ? |
| Tier 1C: soften probs | ? | ? | ? | ? | ? |
| Tier 2A: calibration | ? | ? | ? | ? | ? |
```

---

## Execution Timeline

### Round-by-Round Strategy

**Rounds use ~2h45m windows. Competition runs until Mar 22 15:00 CET.**
**Current time: ~15:00 CET Mar 21. Remaining: ~24 hours, ~8 more rounds.**

#### Phase 1: Foundation (Next 2-3 hours, during Round 16-17)

1. Create `backtest.py` — ability to score against past round GT
2. Download and cache GT data for Rounds 11, 13, 14
3. Measure current baseline on backtests
4. Implement Tier 1A (empirical observation distributions)
5. Implement Tier 1C (soften all predictions)
6. Validate on backtests → submit to Round 16 or 17

**Expected improvement: +15-30 points**

#### Phase 2: Cross-Seed Learning (Next 2 hours)

1. Implement Tier 1B (cross-seed observation sharing)
2. Validate on backtests
3. Submit to next round

**Expected improvement: +5-10 points**

#### Phase 3: Calibration (Next 2-3 hours)

1. Implement Tier 2A (build calibration lookup from past round GT)
2. Replace hand-tuned priors with data-driven priors
3. Validate and submit

**Expected improvement: +5-10 points**

#### Phase 4: Polish (Remaining rounds)

1. Fine-tune softening parameters based on observed scores
2. MC improvements if time permits
3. Submit each round, analyze results, iterate

---

## Key Principles

1. **Observations are samples, not truth.** Every observation is ONE stochastic outcome. Treat them as samples from the target distribution.

2. **Softness wins.** A slightly-too-soft prediction loses little (exp(-3 × small_kl)). A slightly-too-peaked prediction can lose everything (KL diverges when you assign 1% to something that's actually 30%).

3. **Later rounds count exponentially more.** Round 20's weight is 2.65 vs Round 15's 2.08. Focus on getting the code RIGHT, not submitting early.

4. **Verify before deploy.** Every change must show improvement on backtests before going live. No blind submissions.

5. **Cross-seed = free data.** Same hidden params across all 5 seeds means 50 queries give us ~50 independent observations of the same probability distribution.
