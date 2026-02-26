# CSAO Rail Recommendation System
### Cart Super Add-On · Zomathon Hackathon

---

## What We Are Building

A real-time, cart-aware recommendation system that suggests the most relevant add-on items to a user mid-order. Every time the user adds an item to their cart, the CSAO rail updates with a fresh ranked list of 8–10 complementary items — driving AOV, attach rate, and cart-to-order conversion simultaneously.

> **Core insight:** This is not a yes/no classification problem. It is a sequential, multi-objective ranking problem. Each cart addition changes the state, and the goal is not just the next accepted item — it is the total value and completeness of the entire session.

### Why Ranking and Not Classification

- Users see only 8–10 slots on the rail — a ranking model gets the best candidates into these limited slots
- Multiple objectives must be balanced simultaneously: acceptance, AOV, abandonment risk, timing, position quality
- Cart context changes with every addition — the system must adapt sequentially after each item is added
- Cuisine coherence, dietary constraints, and quantity saturation must all be enforced

---

## Full Pipeline — Step by Step

**Total pipeline latency: 22–37ms. Well within the 200–300ms production constraint.**

---

### Step 1 — Input Ingestion

- Current cart items with cuisine tags, categories, and quantities
- Full restaurant menu pre-tagged with cuisine, category, availability, and margin percentage
- Session context: time of day, zone, meal period (breakfast / lunch / dinner / late night)
- User order history across all restaurants — not just the current one — to understand expense patterns and cuisine preferences

---

### Step 2 — Hard Pre-Filters

Deterministic rules. Applied before any ML model is involved. Run in this exact order.

#### Filter A — Availability and Margin

- Remove out-of-stock items immediately
- Remove items where restaurant profit margin is below 10%
- Verify active promotions are still valid and prices have not changed

#### Filter B — Dietary Toggle *(Optional — Only If Set This Session)*

- **Veg toggle ON** → hard exclude all non-veg items from the entire candidate pool
- **Vegan toggle ON** → hard exclude dairy, eggs, meat, and honey
- **Non-veg toggle ON** → no filter applied, all items eligible
- **No toggle set** → infer from cart composition only. If cart already contains non-veg items, show both. If cart is all-veg and no toggle is set, default to veg recommendations but do not hard filter.
- Session-level signal — overrides user profile preference for this session entirely

#### Filter C — Cuisine Coherence

- Identify the dominant cuisine tag in the current cart (most common tag across all cart items)
- Keep only candidates matching that dominant cuisine — Chinese cart gets Chinese add-ons only
- Empty cart → use the restaurant's primary cuisine as default
- Combo or thali in cart → candidates must complement that combo specifically — no suggesting another sabzi when a thali is already present

#### Filter D — Quantity Saturation

- For each candidate's subcategory, count how many of that subcategory are already in the cart
- Exclude candidate if count meets or exceeds the configurable maximum (e.g. rice capped at 3, beverages at 2)
- Prevents suggesting a fifth biryani variant to a cart that already has three rice dishes
- For group orders where cart size exceeds 5 items, saturation thresholds scale up proportionally

#### Filter E — Deduplication and Recommendation Fatigue

- Remove exact duplicates of items already in cart
- Track per-category ignore count this session — if a category has been displayed and ignored 3 consecutive times, suppress it for the remainder of the session

---

### Step 3 — Candidate Generation

Pre-computed item embeddings plus exact dot product search. No FAISS required.

- Item embeddings generated offline once on Day 1 using **Sentence Transformers** (`all-MiniLM-L6-v2`) on item names and descriptions. Stored to disk. Never recomputed at runtime.
- At request time: compute dot product similarity between query vector (current cart + user context) and all filtered candidate item vectors using numpy matrix multiplication
- Sort by score, take top 50 candidates
- After cuisine and availability filtering the candidate pool is 30–80 items. Exact numpy search is faster than FAISS at this scale with zero approximation error and no additional infrastructure.

---

### Step 4 — Feature Assembly

#### Static Features — Pre-Computed and Cached *(< 1ms lookup via Python dict)*

| Category | Features |
|----------|----------|
| User | Segment, RFM metrics, city, dietary preference history, profile veg days setting (specific days of week marked veg in Zomato profile) |
| Item | Price, category, veg flag, bestseller flag, popularity score (overall + by meal time), semantic embedding, margin flag, prep time, restaurant add-on success rate |
| Restaurant | Cuisine type, price tier, rating, average prep time |

#### Dynamic Features — Computed Per Request *(2–5ms)*

- **Cart stage:** 0 = empty, 1 = has main, 2 = main + carb, 3 = nearly complete meal
- **Effective categories** after combo and thali decomposition
- **Gap detection:** missing beverage, missing dessert, missing bread — candidate fills gap flag (highest-signal feature)
- **Complement score:** item-level pairing boost — biryani raises salan score, main course raises bread score
- **Price anchor:** candidate price relative to the first item added to cart
- **Session signals:** time since last cart addition, consecutive rejection count, abandonment risk composite score
- **Distance-to-discount:** gap to nearest threshold, closes gap flag, free delivery unlock flag, nudge urgency score, overshoot risk flag
- **Peak hour mode:** lunch-C2O (12–2pm), dinner-AOV (7–10pm), late-night-impulse
- **Seasonal adjustment:** beverage and dessert preference weights shifted by current month
- **Profile veg days signal:** if today matches a day the user has marked as veg in their Zomato profile settings, apply a hard veg filter equivalent to the veg toggle — this is a declared preference, not an inference

---

### Step 5 — GRU Cart Encoder + LightGBM Ranker

**Two components working together. GRU encodes the sequential cart trajectory. LightGBM ranks candidates against five business-aligned label targets using a native LambdaRank objective.**

#### GRU Cart Encoder *(PyTorch, trained once)*

Processes cart items in the exact order they were added. Produces a 64-dimensional hidden state representing the trajectory of the meal. Biryani then Salan encodes differently than Salan added alone. This vector captures the evolving meal story that mean pooling or bag-of-items approaches cannot. The hidden state is extracted after training and used as 64 numeric features fed directly into LightGBM alongside all other features.

#### LightGBM Ranker *(primary model)*

Takes all features — GRU hidden state vector + static features + dynamic features — as a flat feature matrix. Trains five separate models, one per business objective, each with its own label target. Fast to train (minutes not hours), natively supports LambdaRank, produces interpretable feature importance out of the box.

#### Five Business-Aligned Label Targets

| Model | Predicts | Business Metric | Label |
|-------|----------|-----------------|-------|
| Accept | P(user adds item) | Rail Order Share + Attach Rate | Was item added to cart? |
| AOV | Expected value contribution | Average Order Value | Item price × accepted / 500 |
| Abandon | P(cart abandoned) | Cart-to-Order Ratio *(subtracted)* | Session ended without order? |
| Timing | P(fits current cart stage) | Average Items per Order | Accepted at appropriate cart stage? |
| Anchor | P(strong position 1 item) | Position 1 Attach Rate | Accepted when shown at position 1? |

#### Final Business Score

Weights are tunable post-deployment without retraining.

```
Score = 0.30 × p_accept  +  0.30 × e_aov  −  0.20 × p_abandon  +  0.10 × p_timing  +  0.10 × p_anchor
```

#### Loss Function

- **LambdaRank** (`objective='lambdarank'`) on the Accept model — native LightGBM objective, directly optimises NDCG as a list-level metric without custom loss implementation
- **Regression** (`objective='regression'`) on the AOV model
- **Binary classification** (`objective='binary'`) on Abandon, Timing, Anchor models
- Each model trained and evaluated independently — no convergence coupling, no gradient balancing issues

#### Why LightGBM Over a Custom Neural Ranker

- Trains in minutes — multiple retraining cycles possible within a single day
- `lambdarank` is a native objective — no custom loss code needed
- Feature importance via SHAP is trivial and built in
- No ONNX export needed — LightGBM inference is already under 10ms
- Interpretable predictions — judges can see exactly which features drive each recommendation
- If one model fails, the other four are unaffected

#### Peak Hour Objective Switching

| Time | Mode | Priority |
|------|------|----------|
| 12–2pm | Lunch rush | Upweight C2O and low prep time — maximise order completion |
| 7–10pm | Dinner | Upweight AOV and premium complements — higher spend intent |
| Late night | Impulse | Upweight low-price impulse items |

---

### Step 6 — Post-Ranking Processing

- **Diversity:** maximum two items per subcategory in the final rail
- **Category mix:** spread across complement, side, drink, dessert where available
- **Price shock check:** remove candidates priced more than 30% above or below cart average
- **Margin cap:** no more than 30% of final rail may be ultra-high-margin items
- **Time-of-day exclusions:** no breakfast items at dinner, no heavy mains at breakfast
- **Distance-to-discount override:** if nudge urgency exceeds threshold, gap-closing item forced to position 1 with explicit explanation text

---

### Step 7 — Position-Aware Final Output

- **Position 1:** anchor head score or D-t-D override — the hook that drives rail engagement
- **Positions 2–10:** remaining candidates by full business score
- One-line explanation text for position 1 item derived from top contributing feature
- Maximum 200 candidates enter ranking. Output top 8–10 for UI.

---

## Distance-to-Discount Nudge

Detects how close the cart is to unlocking a discount or free delivery threshold. Surfaces the most relevant gap-closing item at position 1 with an explicit nudge message. Directly impacts C2O, AOV, and attach rate simultaneously.

| Threshold Type | Example | User Psychology | Business Impact |
|----------------|---------|-----------------|-----------------|
| Free Delivery | Free delivery above ₹199 | Highest urgency — users strongly dislike paying delivery fees | C2O lift + AOV |
| Free Item | Free dessert above ₹399 | Tangible reward, highly motivating | AOV + Attach Rate |
| Percentage Discount | 20% off above ₹499 | Moderate urgency | AOV |

### Overshoot Protection

If gap to threshold is ₹49 and we recommend a ₹400 item, the user crosses the threshold but feels manipulated. Items that overshoot the gap by more than 1.8× are moved to the back of the candidate list — not removed, but not the primary nudge recommendation.

---

## Baseline Model — Market Basket Analysis

One baseline only. Industry standard, credible to judges, and its failures map directly to each design decision in the main system.

| MBA Limitation | What It Means | Our System's Answer |
|----------------|--------------|---------------------|
| Same rules for every user | Budget and premium users get identical recommendations | Segment-aware features in LightGBM ranker |
| Ignores cart stage | May recommend dessert after first item added | Timing model trained on cart stage acceptance patterns |
| Ignores abandonment risk | No awareness of users about to leave | Abandon model trained on sessions that did not convert |
| Ignores position | Position 1 chosen same as position 8 | Anchor model optimises specifically for position 1 |
| Ignores cuisine match | May suggest Indian sides for Chinese cart | Cuisine coherence hard filter enforces match |
| Cold start failure | New restaurant has zero rules, zero output | Semantic embeddings provide fallback signal |
| No threshold awareness | Unaware of discount proximity | Distance-to-discount features and position override |

---

## User Segments and Behavioural Logic

| Segment | Key Behaviour | Best Add-On Strategy |
|---------|--------------|----------------------|
| Budget / Student | Adds items only under ₹50 or with BOGO offer | Small Coke, single Gulab Jamun, extra Pav — lowest friction first |
| Premium / Gourmet | Price insensitive, values experience completion | Premium shakes, artisanal desserts, large sides |
| Health-Conscious | Avoids fried food and sugary drinks | Fresh lime soda no sugar, salad, roasted papad |
| Family / Group | Large carts, needs multi-serve packaging | 2L bottles, 4-pack brownies, bucket fries — quantity-aware |
| Occasional | Low order history, sparse data | Cuisine-level popularity + semantic embedding fallback |

### Additional Contextual Signals

- **Profile veg days setting:** Zomato allows users to mark specific days of the week as veg days directly in their profile. If today is one of those days, it is treated as a hard veg filter — identical in effect to the user toggling veg manually for this session. This is a declared user preference read from the profile, not inferred from order history.
- **Seasonal preference:** cold drinks and ice cream weighted higher in summer. Hot beverages and warm desserts weighted higher in winter. Applied to beverage and dessert categories based on current month.

---

## System Design and Latency

### Latency Budget

| Component | Time | Method |
|-----------|------|--------|
| Static feature fetch (Python dict) | < 1ms | In-memory key-value lookup |
| Hard pre-filters — all layers | 5–10ms | Deterministic rules, no ML |
| Dynamic feature computation | 5ms | Cart state arithmetic |
| Candidate similarity scoring (numpy) | < 1ms | Exact dot product, 30–80 items |
| GRU cart encoder inference | < 5ms | PyTorch forward pass, small model |
| LightGBM ranker (5 models) | 5–10ms | Native tree inference, no ONNX needed |
| Post-ranking processing | 5–10ms | Diversity, price shock, margin cap rules |
| **Total** | **22–37ms** | **Well within 200–300ms constraint** |

### Feature Store — Python Dictionary

Pre-computed static features are stored in a Python dictionary loaded into memory at startup. Sub-millisecond lookup. Functionally identical to Redis for a single-process demo. In production this would be replaced with a Redis cluster for persistence and multi-server access — a drop-in replacement requiring no changes to feature engineering or model code.

### Why Not FAISS

FAISS is built for approximate nearest neighbour search across millions of vectors. After cuisine coherence and availability filtering, the candidate pool is 30–80 items. Exact dot product search via numpy is faster at this scale because FAISS index loading overhead exceeds the search cost itself. Exact search also has zero approximation error.

### LightGBM Inference Speed

LightGBM tree inference requires no export, no runtime engine, and no additional tooling. Five models (one per business objective) run in 5–10ms total. This is faster than ONNX Runtime inference for a neural ranker and requires zero serving infrastructure beyond the trained model files.

### Production Guardrails

| Risk | Guardrail | Metric | Auto-Stop If |
|------|-----------|--------|--------------|
| Recommendation fatigue | Max 2 same subcategory in rail | CTR | Drop > 5% |
| Price shock | Candidate within ±30% of cart average | Abandonment | Any increase |
| Cultural mismatch | Cuisine coherence hard filter | Complaint rate | > 0.1% |
| Over-commercial feel | < 30% ultra-high-margin items in rail | NPS | Any drop |
| Latency degradation | p95 below 250ms always | p95 latency | > 250ms |

---

## Evaluation Framework

### Data Split

Temporal split strictly enforced. Train on weeks 1–3, validate on week 4, test on week 5. No random shuffling at any stage — no future information leaks into training.

### Offline Metrics

- **AUC-ROC:** overall discrimination of the accept model
- **Precision@8:** how many of 8 displayed items were accepted — maps directly to attach rate
- **NDCG@8:** ranking quality — are accepted items appearing at the top positions
- Each of the five LightGBM models evaluated independently — clean separation of business objectives

### Breakdown Analysis

All metrics reported by user segment (Budget, Mid, Premium, Occasional, Cold Start) and by cart stage (0–3). MBA degrades at stages 2–3. Our ranker shows the biggest advantage at high cart stages — this contrast is the clearest evidence that sequential cart understanding is working.

### Business Metric Translation

| Offline Metric | Business Metric | Connection | Target |
|----------------|-----------------|------------|--------|
| Precision@8 | Attach Rate | Higher P@8 = more shown items accepted | > 15% improvement |
| NDCG@8 | AOV | Better ranking surfaces higher-value items first | > 5% AOV lift |
| Abandon model loss | C2O Ratio | Lower predicted abandonment = more completed orders | Maintained or improved |
| AUC | Rail Order Share | Better discrimination = more sessions with ≥1 add-on | > 0.75 |

### A/B Test Design

- **Control:** Market Basket Analysis
- **Treatment:** GRU encoder + LightGBM five-objective system
- **Randomisation unit:** user ID — prevents within-user contamination
- **Split:** 50/50 · **Duration:** minimum 14 days · **Sample size:** 100,000+ sessions per variant

**Primary success metrics:** AOV lift > 5%, attach rate improvement > 15%

**Auto-stop guardrails:**
- Cart abandonment increases more than 1.5%
- Refunds or returns increase more than 1%
- Day-7 user retention drops more than 1%
- p95 serving latency exceeds 250ms

### Production Readiness Checklist

- [ ] p95 latency confirmed below 250ms
- [ ] Coverage above 98% — cold start handled for all user types
- [ ] AOV lift above 4% confirmed in A/B test
- [ ] Zero margin-negative recommendations
- [ ] Dietary toggle verified with 100% accuracy on labelled test cases

---

## Tech Stack

| Library | Category | Usage |
|---------|----------|-------|
| `pandas` | Data | All dataframes, feature engineering, temporal train/test splitting |
| `numpy` | Data + Search | Exact dot product candidate scoring, NDCG and Precision@K computation, vectorised feature operations |
| `faker` | Data | Realistic restaurant names, user profiles, city distributions for synthetic data |
| `scikit-learn` | Data + Eval | Label encoding, StandardScaler, train/test split, AUC-ROC and precision metrics |
| `mlxtend` | Baseline | Apriori algorithm and `association_rules` for Market Basket Analysis baseline |
| `torch` | GRU encoder | Small GRU cart encoder only — processes cart item sequence, outputs 64-dim trajectory vector used as features by LightGBM. Not used for the ranker. |
| `lightgbm` | Ranker | Five separate models with business-aligned labels. Accept model uses `lambdarank` objective natively. AOV uses regression. Abandon, Timing, Anchor use binary classification. Fast training, built-in feature importance, no serving infrastructure needed. |
| `sentence-transformers` | Embeddings | Run once offline on Day 1 using `all-MiniLM-L6-v2`. Semantic embeddings for all item names and descriptions stored as numpy array. Never called at runtime. |
| `matplotlib` | Visualisation | Training loss curves per head, metric comparison between MBA and main system |
| `seaborn` | Visualisation | Segment-by-metric heatmaps, feature distribution plots, cart stage breakdown charts |
| `plotly` | Presentation | Interactive AOV lift projection and A/B test power analysis for slides |
| `jupyter` | Environment | All development, training, and demo. Judges interact with notebooks directly. |

### Installation

```bash
pip install pandas numpy scikit-learn faker mlxtend
pip install lightgbm
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers
pip install matplotlib seaborn plotly jupyter
```

---

## Four-Day Execution Plan

**5 hours per day · 20 hours total**

| Day | Focus | Tasks |
|-----|-------|-------|
| **1** | Data Foundation | Synthetic data: users, restaurants, items including combos and platters. Session simulation with sequential cart events, dietary toggles as session state, combo expansion logic. All hard filter layers. Three-layer feature pipeline. Run Sentence Transformers offline, store embeddings. Verify 8–12% positive label rate. |
| **2** | Baseline + Candidates | Build and fully evaluate MBA baseline. Record all numbers — these are your comparison targets. Build item-item co-occurrence matrix for candidate generation. Assemble complete feature matrix with temporal split. Verify pipeline end-to-end on a sample session. |
| **3** | Main Model | Build small GRU cart encoder in PyTorch — train to produce 64-dim cart trajectory vector. Extract GRU features. Engineer all five label targets from session data. Train five LightGBM models: Accept (lambdarank), AOV (regression), Abandon, Timing, Anchor (binary). Evaluate vs MBA: AUC, Precision@8, NDCG@8. Run segment and cart stage breakdowns. |
| **4** | Polish | Business impact calculations and AOV lift projection. A/B test design written with guardrails. Architecture diagram on Excalidraw. Demo notebook: combo detection, dietary toggle, D-t-D nudge, cold start vs warm user. Slides covering all six evaluation criteria. |

---

## How Every Evaluation Criterion Is Covered

| Criterion | Weight | Coverage |
|-----------|--------|----------|
| Data Preparation & Feature Engineering | 20% | Synthetic data with dietary toggles, combos, sparse users, peak hours, D-t-D offers, sequential cart events, temporal split, Sentence Transformer semantic enrichment |
| Ideation & Problem Formulation | 15% | Ranking not classification. Five business metrics mapped to five direct optimisation targets. MBA failure analysis motivates every design decision. |
| Model Architecture & AI Edge | 20% | GRU encoder for sequential cart logic as learned feature generator. Five LightGBM models with business-aligned label targets and native LambdaRank objective. Sentence Transformers as AI Edge — semantic item understanding at zero runtime cost. Fast, interpretable, production-ready serving. |
| Model Evaluation & Fine-Tuning | 15% | Temporal split. NDCG@8, Precision@8, AUC per segment and cart stage. Each of the five LightGBM models evaluated independently. Segment-level breakdown identifies where system outperforms baseline most. |
| System Design & Production Readiness | 15% | Full latency budget 22–37ms. Python dict as feature store. No FAISS — correct tool choice. LightGBM inference needs no serving infrastructure. Hard filter ordering. Position override logic. Guardrail auto-stop rules. |
| Business Impact & A/B Testing | 15% | Every model head maps to a named metric. D-t-D drives C2O, AOV, attach rate simultaneously. AOV lift formula. A/B design with primary metrics and guardrail thresholds. |

---

> *Every component exists because a specific business metric or operational constraint required it — not because it was technically interesting.*

---

*CSAO Rail Recommendation System · Zomathon · v2*
