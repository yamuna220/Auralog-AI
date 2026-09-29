# AURAlog AI

**AI-Powered Privacy Exposure & De-Anonymization Risk Simulator**

Built for Bharat Innovation Challenge (BIC) 2.0 — Cybersecurity & Ethical Tech.

AURAlog AI demonstrates how multiple individually harmless public
social-media posts can be correlated to reveal sensitive information —
approximate location, daily routines, education/workplace clues, timing
patterns, and identity-linkage signals — that no single post reveals on
its own. It is an **ethical, defensive privacy-awareness tool**: it never
performs real-world identification, and works only on synthetic/demo data
or text the user enters themselves.

## What it does

1. Accepts multiple public social-media posts (synthetic/demo only).
2. Extracts privacy-relevant signals from each post (AI-assisted).
3. Correlates signals across posts (deterministic rule engine).
4. Calculates a transparent, explainable privacy exposure score
   (deterministic — **the LLM never decides the final score**).
5. Builds and visualizes an inference/attack graph.
6. Explains what an attacker could plausibly infer, explicitly labeling
   each claim as **directly stated**, **inferred**, or **speculative**.
7. Generates safer, sanitized rewrites of flagged posts.
8. Re-scores the sanitized posts and shows a before/after comparison.

## Architecture

```
USER POSTS
 -> INPUT VALIDATION
 -> AI SIGNAL EXTRACTION          (services/gemini_service.py)
 -> NORMALIZED SIGNALS            (core/models.py)
 -> CROSS-POST CORRELATION        (core/correlation_engine.py — deterministic)
 -> DETERMINISTIC RISK SCORING    (core/risk_engine.py — deterministic)
 -> INFERENCE GRAPH               (core/graph_engine.py)
 -> EXPLANATION                   (services/gemini_service.py, given the score)
 -> SANITIZATION                  (core/sanitizer.py + services/gemini_service.py)
 -> RE-ANALYSIS                   (same pipeline, run again)
 -> BEFORE/AFTER COMPARISON       (ui/sanitizer_view.py)
```

The scoring and correlation logic is pure, deterministic Python with no
LLM involvement — every number in the final exposure score is traceable to
a specific extracted signal or a specific correlation rule. The LLM is used
only for semantic extraction, narrative explanation, and rewriting —
never to invent the score itself.

```
auralog/
├── app.py                    Streamlit entrypoint + navigation
├── config.py                 Central config: weights, thresholds, demo mode
├── core/                     Pure pipeline logic (no Streamlit, no Gemini SDK)
│   ├── models.py             Shared dataclasses (Post, Signal, Correlation, RiskReport...)
│   ├── signal_engine.py      Per-post AI signal extraction (delegates to services/)
│   ├── correlation_engine.py Cross-post correlation rules (deterministic)
│   ├── risk_engine.py        Exposure scoring formula (deterministic)
│   ├── graph_engine.py       Inference graph construction
│   ├── sanitizer.py          Post rewriting + re-scoring orchestration
│   └── demo_data.py          Canned example posts/signals for DEMO MODE
├── services/
│   ├── gemini_service.py     Single access point for all Gemini calls
│   └── analysis_service.py   Orchestrates the full pipeline end to end
├── ui/                        Streamlit page renderers (thin — call into services/)
│   ├── home.py
│   ├── input_page.py
│   ├── dashboard.py
│   ├── graph_view.py
│   └── sanitizer_view.py
├── state/
│   └── session.py            Single owner of st.session_state
├── utils/
│   └── json_safety.py        Strict validation of LLM JSON output
└── tests/                     Unit tests for the deterministic engines
```

## Running the Application

```bash
pip install -r requirements.txt
cp .env.example .env   # optional — fill in GEMINI_API_KEY, or leave blank
streamlit run app.py
```

The application opens a full single-page dashboard. No manual page navigation is required — the workflow flows top to bottom:

```
Enter Posts → Analyze → Risk Report → Signals → Correlations
→ Inference Chains → Visualization → Sanitize → Before/After Verification
```

**Demo Mode** (no API key needed):
1. Open the application — it auto-detects Demo Mode and shows a banner.
2. Click **Load Demo Scenario** — populates the canonical 3-post example.
3. Click **Analyze Privacy Exposure** — runs the full deterministic pipeline.
4. Scroll through the complete report.
5. Click **Sanitize My Posts** — invokes the sanitizer and verifies risk reduction.
6. View the before/after comparison and verified badge.

**API Key configuration** (in priority order):
| Method | How |
|--------|-----|
| Streamlit secrets | `~/.streamlit/secrets.toml` → `GEMINI_API_KEY = "..."` |
| Environment variable | `GEMINI_API_KEY=...` in `.env` or shell |
| Sidebar input | Enter directly in the running app — never displayed or logged |



## Privacy Exposure Scoring

**1. Why Gemini does not determine the final score.** Gemini's role is
semantic — it reads post text and extracts candidate signals, and it can
suggest its own narrative correlations/inference chains. But an LLM's
confidence in its own guess is not a measurement, and asking it to also
grade "how risky is this" would make the score unreproducible (a rerun
could give a different number) and unexplainable (you can't point to a
formula). So AURAlog splits the job: Gemini extracts, and a completely
separate, deterministic engine (`core/risk_engine.py`) — plain Python
arithmetic, no API calls — turns those extracted signals into the actual
number. Feed it the same signals twice and you get the exact same score
both times.

**2. How signals contribute.** Every `PrivacySignal` gets a point value:

```
signal_points = confidence × sensitivity_weight × category_weight × 100
```

`confidence` is how strongly the post's own text supports the signal
(from the AI extraction). `sensitivity_weight` (low/medium/high/critical →
0.25/0.55/0.80/1.00) is how revealing that specific detail is.
`category_weight` (see `config.CATEGORY_WEIGHTS`) is how sensitive that
*category* of information is in general — Location and Identity are
weighted highest, Lifestyle lowest. When several signals land in the same
category, they aren't just added up — the second-strongest counts for half
as much as the strongest, the third for a quarter, and so on
(`config.DIMINISHING_RETURNS_FACTOR`), so five weak hints about the same
thing can't outscore one strong one by simply piling up.

**3. How correlations amplify exposure.** A single clue in isolation is
one thing; the same clue *plus* another clue from a different post that
plausibly relates to it is more revealing than either alone — this is the
whole premise of the app. `core/correlation_engine.py` looks for specific
category pairings across different posts (e.g. a location clue + a campus
clue) and, when found, `core/risk_engine.py` adds a separate
`correlation_contribution` to the score on top of the raw signal score.
No correlations found → that contribution is simply zero, and the score
falls back to just the signal exposure.

**4. How inference chains contribute.** An inference chain represents
several correlations combining into one higher-order, hypothetical
conclusion (e.g. "these clues together might narrow a commute area"). Each
chain contributes points scaled by its own confidence and how many
correlations it links together — but its confidence can never reach 1.0
(enforced at the data-model level, not just here), because a chain of
inferences is never a confirmed fact.

**5. How category scores are calculated.** Signals are grouped by category
(Location, Routine, Time, Education, Workplace, Identity, Social
Relationships, Travel, Lifestyle, Personal Information); each category's
score is the diminishing-returns sum of its own signals' points, capped at
100. Only categories that actually have a signal appear in the breakdown.

**6. How the final score is normalized.** Three numbers — `signal_
contribution`, `correlation_contribution`, `inference_contribution` — are
each already bounded to 0-100. The overall score is a weighted average of
the three (55% / 25% / 20%, `config.SIGNAL_COMPONENT_WEIGHT` & co., weights
summing to 1.0), which keeps the final result automatically inside [0, 100]
with no separate clamping step needed — though the underlying `RiskAssessment`
model enforces that bound regardless.

**7. How risk levels are determined.** The final score maps to a plain
label via fixed boundaries (`config.RISK_LEVEL_UPPER_BOUNDS`): 0-24 LOW,
25-49 MODERATE, 50-74 HIGH, 75-100 CRITICAL.

**What the score means (and doesn't).** A score of 80 means *"high modeled
privacy exposure under AURAlog's own scoring methodology"* — a measure of
how much a determined reader could plausibly piece together from the
supplied text, by this application's own explicit, inspectable rules. It
does **not** mean an 80% chance of being attacked, doxxed, or identified.
Every generated explanation ends by restating this distinction, and it's
worth repeating anywhere the score is shown in the UI.

## Inference Chain Engine

**1. What an inference chain means.** A single correlation links two (or a
few) signals — "these two clues seem related." An inference chain goes one
level higher: it's the narrative that emerges when *several already-detected
correlations, taken together*, tell a bigger story than any one of them
alone — while still being explicit that it's a *potential*, not a confirmed,
conclusion.

**2. Why multiple harmless signals can become more revealing when
combined.** A cafe near a metro station is unremarkable on its own. A
university commute is unremarkable on its own. But a cafe-near-a-metro clue
*plus* a university-commute clue *plus* a recurring-routine clue, once the
correlation engine links them across posts, can plausibly narrow a
person's general area and daily pattern more than any single post reveals
— that compounding effect is exactly what an inference chain is meant to
surface, always hedged ("may reveal", "could narrow" — never "is").

**3. How the graph is constructed.** Conceptually: signals are nodes,
correlations are edges. `core/inference_engine.py` recognizes five named,
semantically meaningful 3-category patterns (Location+Time+Routine,
Location+Education+Routine, Education+Identity+Social Relationships,
Workplace+Location+Routine, Travel+Location+Time) and only promotes one
into a full chain when at least 2 of its 3 possible category-pair edges are
backed by a *real* correlation — with exactly 3 categories, 2 edges is the
precise point at which all 3 are provably connected, not just "technically
reachable."

**4. How chains are separated (avoiding the "mega-chain" problem).**
Naively, "every signal transitively connected to every other signal becomes
one chain" sounds reasonable but breaks down fast: one hub-like signal that
happens to sit in several unrelated correlations would otherwise drag
everything into one meaningless blob. Named-pattern matching sidesteps this
by requiring a specific, bounded category triple with real edge support —
not open-ended transitive closure. Any correlation left over (not consumed
by a named pattern) becomes its own standalone chain rather than being
force-merged into, or dropped from, a narrative it doesn't actually
support — so a single, independent 2-signal relationship stays exactly
that: its own separate, valid chain.

**5. How strength and confidence differ.** Same split as correlations:
`confidence` is how strongly the *supplied evidence* supports the
inference (bounded below 1.0 — never "proof"); `strength` is how strong the
*modeled relationship itself* is, built from the participating
correlations' own strengths (diminishing returns — more evidence helps, but
less each time), signal confidence/sensitivity, and whether the underlying
text shows a genuinely repeated pattern versus a one-off mention. Chain
*length* alone never makes a chain strong — five weakly-related signals
score lower than two strongly-related ones.

**6. How evidence is traced back to posts.** Every chain stores its own
`signal_ids`, `correlation_ids`, and `post_ids` — nothing in a chain is
invented; every one of those ids refers to a real, already-validated object
elsewhere in the same analysis. The breadcrumb a judge can always follow:
`Post → Signal → Correlation → Inference Chain`, all the way back to the
original post text.

**7. How the system avoids unsupported conclusions.** A chain's id is
derived deterministically from its own sorted signal ids and pattern name
(a stable hash, never a random UUID) — same evidence always yields the
same chain. Its `explanation` and `resulting_inference` text is generated
entirely from the actual contributing signals/correlations by template,
never by a second Gemini call. And, like every derived value in this app,
a chain's confidence can never reach 1.0 — enforced by the data model
itself, not just by convention.

```
Posts
  ↓
Signals
  ↓
Correlations
  ↓
Inference Chains
  ↓
Privacy Exposure
```

## Privacy Sanitization

**1. Product Lifecycle.** AURAlog follows a strict four-step privacy remediation methodology:
```text
DETECT  →  EXPLAIN  →  FIX  →  VERIFY
```

**2. Never Trust an LLM Rewrite on Faith.** AURAlog does not assume a rewritten post is safe simply because Gemini generated it. Instead, every sanitized post is fed back into the **exact same deterministic analysis pipeline** to verify actual risk reduction:
```text
Original Posts → Analysis → Risk Assessment → Sanitizer → Sanitized Posts → Analysis AGAIN → Before/After Comparison
```

**3. Safety Check & Automatic Reversion.** If a proposed rewrite introduces new privacy signals or causes a risk score regression (`sanitized_score > original_score`), AURAlog automatically marks the status as `unsafe` and reverts to the original post or a safer deterministic fallback.

**4. Deterministic Fallback.** If Gemini is unavailable, rate-limited, or returns malformed JSON, `core/sanitizer.py` applies targeted, rule-based regular expression replacements based on detected signal categories (Location, Exact Time, Routine, Identity/Education) without hardcoding output.


## Running tests

```bash
pytest
```

Every test — including all Gemini-service, risk-engine, correlation-engine,
and inference-engine tests — runs against mocked/hand-built input and never
requires a real API key or network access.

## Project status

This is a staged build.

- **Data models** (`core/models.py`) — fully implemented and validated
  (Pydantic), including the safety rule that inferred/derived claims can
  never carry full (1.0) confidence.
- **Gemini integration** (`services/gemini_service.py`,
  `services/prompt_templates.py`) — fully implemented: batched
  `analyze_posts()` call, typed error handling (auth/rate-limit/network/
  malformed-JSON with one corrective retry), Demo Mode fallback, and
  `sanitize_post()`. Gemini never produces a final risk score.
- **Deterministic correlation, inference, and scoring**
  (`core/correlation_engine.py`, `core/inference_engine.py`,
  `core/risk_engine.py`) — fully implemented: rule-based cross-post
  correlation detection, named-pattern inference-chain construction (with
  mega-chain prevention), category scoring, the weighted signal/
  correlation/inference formula, risk-level classification, deterministic
  explanation text, and before/after comparison. See "Privacy Exposure
  Scoring" and "Inference Chain Engine" above.
- **UI wiring** — the Streamlit pages exist and navigate correctly but
  don't yet render the richer `RiskAssessment`/`InferenceChain` fields
  (category breakdown, top risk factors, chain explanations) or the
  correlation/inference graph; that's the next stage.
