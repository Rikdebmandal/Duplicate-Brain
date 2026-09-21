# Personal Cognitive Digital Twin

A behavioural decision-prediction system. It learns patterns from an individual's
recorded decisions and estimates how they would probably respond to a new
situation — with the evidence behind every estimate.

**What it is not:** it is not a simulation of a person's mind, and it produces no
psychological or medical assessment. Every trait it reports is a statistical
estimate carrying a confidence and an evidence count; every prediction is a
probability, not a statement of what someone will do.

The design principle throughout: **model behaviour from evidence, predict
probabilistically, explain with evidence, learn from mistakes, and make
uncertainty visible.**

---

## Quick start

### Docker (Postgres + pgvector)

```bash
cp .env.example .env
# set SECRET_KEY, and optionally ANTHROPIC_API_KEY
docker compose up --build
```

Then: API on <http://localhost:8000> (docs at `/docs`), UI on <http://localhost:3000>.

### Local (no database server needed)

The stack falls back to SQLite when `DATABASE_URL` is unset, so it runs with
nothing installed but Python and Node.

```bash
cd backend
python -m venv .venv && . .venv/Scripts/activate   # Linux/macOS: . .venv/bin/activate
pip install -r requirements-dev.txt
alembic upgrade head
python -m seeds.seed_data --reset       # 37-decision demo history
uvicorn app.main:app --reload
```

```bash
cd frontend
npm install
npm run dev
```

Sign in as `demo@example.com` / `demo-password-123`.

---

## What a prediction looks like

Given the scenario from the brief, run against the seeded demo history:

```
Predicted decision   reject the offer — 86%
Confidence           high (0.86), from 37 recorded decisions

Important factors    (contribution to "reject the offer")
  Financial reward             ---   magnitude 0.99
  Family / relationship cost   ++    magnitude 0.57
  Uncertainty                  ++    magnitude 0.57

Similar decisions you actually made
  47%  "A competitor offered roughly double my salary to move to Dubai..."
       Chose: Stay and decline
       "Double is a lot of money. But there is nobody else near my parents now."
  42%  Chose: Decline and stay employed
  42%  Chose: Stay out of it

Alternative                  accept the offer — 14%
```

Every number above is traceable: the factor magnitudes carry the phrases that
produced them, and the similar decisions are real rows in the user's history.

---

## How a prediction is produced

```
scenario text
      ↓
  parse            12 factor magnitudes + option stances, from a lexicon.
      ↓            Deterministic, offline, every number traceable to words.
  profile          Trait-weighted utility model. Decomposes exactly into
      ↓            one signed contribution per factor.
  ┌────────────────┬──────────────────┬─────────────────┐
  │ statistical    │ retrieval        │ LLM reasoning   │
  │ conditional    │ nearest past     │ optional,       │
  │ logit on the   │ decisions vote   │ opt-in,         │
  │ person's own   │ by stance        │ never required  │
  │ decisions      │                  │                 │
  └────────────────┴──────────────────┴─────────────────┘
      ↓
  fuse             Weighted geometric pool. Weights scale with the evidence
      ↓            each layer actually had.
  calibrate        Temperature fitted on recorded outcomes.
      ↓
  explain          Observed counts and inferred estimates, kept separate.
```

Each layer is optional except the utility model. A missing layer does not break
the prediction: its weight is redistributed, the confidence score drops, and the
response says which layers ran and why the others did not.

### Why these particular choices

**Factors are magnitudes, never judgements.** `risk: 0.8` says the situation
carries risk, not that risk is bad. Whether that pulls towards or away from an
option comes from two other things: the factor's `sign_under_approach` (does the
active option *take on* this factor or *spend down* a protected good) and the
person's traits. Keeping situation separate from person is what lets the same
scenario produce different predictions for different people, and what makes
explanations decomposable.

**Stances are resolved contrastively.** Classifying options one at a time
produces "both options are an approach" — every branch of a decision usually
starts with an active verb. *"Take the stock options"* and *"Take the cash bonus"*
both read as accept, yet one takes on risk and the other protects against it.
Activeness is therefore scored per option and normalised across the set, using
both an accept/decline lexicon and a second semantic axis. Where the two axes
disagree, confidence drops rather than one silently winning; where the options
are not separable at all (*"the smaller flat"* vs *"the larger flat"*), every
option is reported at 0.5 with near-zero confidence and the decision contributes
almost nothing to trait inference.

**Traits are Beta posteriors, not numbers.** Each relevant decision contributes a
weighted pseudo-observation; weights fold in factor magnitude (a scenario with no
risk in it says nothing about risk tolerance), importance, and recency (a
three-year half-life, so genuine preference change shows up). The posterior mean
is the reported value and the posterior spread becomes the confidence, so a trait
backed by two decisions is *reported as* barely known rather than reported as a
number.

**A big reward damps risk aversion; it does not create appetite.** People become
less deterred by risk when the upside is large. Modelling that as a *lift* on
risk tolerance turns a merely risk-neutral person into an actively risk-seeking
one whenever the money is good — which shows up immediately in a counterfactual
sweep as "more risk makes them more likely to accept". Scaling aversion towards
zero instead is both correct and stable.

**Pooling is geometric, not arithmetic.** With a weighted average, one confident
layer drags the result to an extreme on its own. With a weighted geometric mean, a
layer can veto but no single layer can force certainty past what the others
support — the right behaviour for a system whose main failure mode is
over-confidence.

**Confidence measures evidence, not sharpness.** It is deliberately not a
function of the predicted probability. Evidence volume enters twice — as a term
and as a multiplier capping the whole score — so a cleanly-parsed scenario with
two agreeing layers cannot report medium confidence off a single recorded
decision.

**Models are selected on Brier score, not accuracy.** A person who accepts 80% of
the time gives a "predict accept always" baseline 80% accuracy while telling you
nothing about *when* they decline. An honest 0.6 beats an over-confident 0.95 that
is wrong when it matters.

---

## Observed vs inferred

The distinction the brief insists on is enforced structurally, not by wording
conventions. The API returns them in separate keys:

```jsonc
"observed": [                       // counts over records
  "In 6 of 9 recorded decisions where family / relationship cost was
   substantial, chose the safer option."
],
"inferred": [                       // model estimates, always hedged
  { "trait": "family_priority", "value": 0.68, "confidence": 0.48,
    "statement": "On family priority, the evidence weakly suggests a tendency
                  towards high weight (estimate 0.68, confidence 0.48)." }
]
```

Every trait row carries `status: "inferred" | "user_specified"`, a
`credible_interval`, an `evidence_count`, and the ids of the decisions that moved
it most. The UI renders the credible interval as a band behind the value, so a
confident estimate is visually distinct from a guess. `GET /profile/traits/{key}/evidence`
returns the individual weighted observations behind any trait.

If the model has someone wrong, they can say so: a trait override pins the value,
and a `trait_correction` feeds back as strong (but not absolute) evidence.

---

## The learning loop

Recording what actually happened does four things:

1. scores the prediction (Brier, log loss, hit/miss);
2. **promotes the resolved situation into the decision history**, so it becomes
   training evidence like any other decision — this is the step that makes the
   twin improve rather than merely keep score;
3. refits the post-fusion calibration temperature, correcting systematic over- or
   under-confidence;
4. rebuilds the profile and retrains the statistical layer on a cadence.

`GET /analytics/performance` reports accuracy, precision, recall, F1, ROC-AUC,
Brier, log loss and expected calibration error, plus a reliability diagram, a
per-confidence-band breakdown and a per-category breakdown.

---

## Evaluation

`POST /analytics/evaluate`, or `python -m scripts.evaluate --email demo@example.com`.

Two things make the harness honest, and both are easy to get wrong:

- **Time-based splitting.** Decisions are ordered by when they happened and cut
  into train / validation / test blocks. The test block is always in the future
  relative to training, because the question the system claims to answer is
  "given the past, what happens next".
- **No profile leakage.** Traits and fitted parameters are recomputed *from the
  training block alone*. Reusing the stored profile would leak the test decisions
  into the person model, since the stored profile was built from every decision on
  record. That single mistake would inflate every number reported.

On the seeded 37-decision history (train 22 / validation 7 / **test 8**):

| approach  | accuracy | Brier | log loss | ROC-AUC | ECE   |
|-----------|---------:|------:|---------:|--------:|------:|
| baseline  |    0.750 | 0.404 |    0.596 |   0.688 | 0.103 |
| profile   |    0.625 | 0.406 |    0.589 |   0.734 | 0.201 |
| ml        |    0.750 | 0.409 |    0.596 |   0.781 | 0.190 |
| retrieval |    0.625 | 0.480 |    0.673 |   0.672 | 0.120 |
| hybrid    |    0.750 | 0.405 |    0.596 |   0.750 | 0.111 |

**Read this honestly: on eight test decisions these differences are noise.** The
base-rate baseline is genuinely competitive, which is what you should expect from
a person with a strong behavioural default — the seeded persona declines the
active option about two-thirds of the time. The one signal that is arguably real
is ranking quality: the learned models separate options better (AUC 0.73–0.78)
than the baseline (0.69) while matching it on accuracy. That is what you would
hope for, but eight decisions cannot establish it.

This is the number to watch as a real history accumulates. A twin that cannot beat
its own base rate is not yet worth trusting, and the system is built to say so
rather than hide it.

---

## Counterfactuals

Move any factor and re-run. The scenario text, option set and every other factor
are held fixed, so the difference is attributable to the override alone. The LLM
layer is deliberately skipped on a re-run: it would re-interpret the prose and
reintroduce variation unrelated to the slider that moved.

`POST /predictions/{id}/landscape` sweeps every factor across its range and
reports where the predicted choice flips — turning a single prediction into an
answer to "what would have to be true for me to choose differently".

---

## API

Full interactive documentation at `/docs`. The endpoints named in the brief:

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/v1/decisions` | Record a historical decision |
| `GET` | `/api/v1/decisions` | List decisions (filter, search, paginate) |
| `POST` | `/api/v1/scenario/predict` | Predict a new scenario |
| `POST` | `/api/v1/prediction/{id}/feedback` | Record the actual decision |
| `GET` | `/api/v1/profile` | Full behavioural profile |
| `GET` | `/api/v1/profile/traits` | Traits with confidence and evidence |
| `GET` | `/api/v1/analytics` | Dashboard summary |

Plus: bulk import, decision timeline, trait evidence, trait override, the
questionnaire, layered memory, counterfactuals, the decision landscape, model
training, the evaluation harness, text analysis, data export and account deletion.

Errors are uniform:

```json
{ "error": { "code": "not_found", "message": "Prediction not found.", "details": {} } }
```

---

## Architecture

```
backend/
├── app/
│   ├── api/routes/          auth, decisions, profile, predictions, analytics
│   ├── models/              SQLAlchemy ORM (16 tables)
│   ├── schemas/             Pydantic request/response models
│   ├── services/
│   │   ├── decision_engine/ taxonomy, parser, utility model, features
│   │   ├── embeddings/      embedder + vector store (pgvector / NumPy)
│   │   ├── profile/         questionnaire, trait inference, profile service
│   │   ├── prediction/      retrieval, fusion, engine
│   │   ├── explainability/  factor attribution, reasoning traces
│   │   ├── feedback/        outcomes, calibration, retraining
│   │   └── llm/             optional Anthropic reasoning layer
│   ├── ml/                  trainer, metrics, evaluation harness
│   └── database/            portable column types, session management
├── alembic/                 migrations
├── seeds/                   demo decision history
├── scripts/                 CLI evaluation
└── tests/                   120 tests

frontend/src/
├── app/                     dashboard, decisions, predict, profile, performance
├── components/              UI primitives, SVG charts, prediction views
└── lib/                     typed API client
```

### Database

PostgreSQL with pgvector is the production target. Tables: `users`, `scenarios`,
`decisions`, `decision_options`, `decision_factors`, `behavioral_traits`,
`trait_evidence`, `questionnaire_responses`, `memory_items`, `predictions`,
`prediction_options`, `prediction_outcomes`, `feedback`, `embeddings`,
`model_artifacts`, `audit_logs`.

Portability is handled by type decorators: UUIDs are native on Postgres and
`CHAR(36)` elsewhere; JSON is `JSONB` on Postgres; embeddings are a `vector`
column with an IVFFlat cosine index on Postgres and a JSON array scanned with
NumPy elsewhere. **Both paths return identical rankings**, so the test suite
exercises the real retrieval logic rather than a stub.

### Embeddings

The default `hashing` backend is a signed hashing-trick projection of word
unigrams, bigrams and character 4-grams — deterministic, offline, microseconds.
Determinism matters here: a counterfactual re-run must retrieve *the same*
neighbours as the original prediction, or the comparison is meaningless.

It captures lexical overlap rather than deep semantics, so retrieval never relies
on it alone — similarity is `0.55 × text + 0.45 × factor-space`. The factor half
is what finds situations posing the same *trade-off* when the words differ
("quit a stable job for a startup" vs "leave a fixed deposit for equity"). Set
`EMBEDDING_BACKEND=sentence-transformers` for a local transformer instead.

### The LLM layer

Entirely optional and off by default. It runs only when an API key is configured
**and** the individual account has opted in. It receives only what the
deterministic layers already computed — the factor reading, the trait estimates
with confidences, and the retrieved decisions — and is instructed to cite only
those. Without it the system loses narrative explanations and keeps everything
else.

---

## Security and privacy

A decision history is a record of how someone thinks, and is treated accordingly.

- **Passwords**: `hashlib.scrypt` (memory-hard, standard library). This avoids the
  passlib 1.7.4 / bcrypt 4.x incompatibility that breaks fresh installs.
- **Sessions**: JWT, HS256, issuer-checked.
- **Encryption at rest**: Fernet field encryption for free text
  (`python -m app.security keygen`). Disk encryption protects a stolen volume;
  this protects a leaked dump. The app logs an error at startup in production if
  no key is set.
- **Isolation**: every owned row is filtered by `user_id`; cross-account access
  returns 404, not 403 — existence itself is not disclosed.
- **Audit log**: every read and write of decision data records who, what and when.
- **Export**: `GET /auth/export` returns everything in one JSON document.
- **Deletion**: `DELETE /auth/me` erases exhaustively rather than setting a flag.
  Only the audit entry recording the deletion survives, and it holds no content.
- **LLM opt-in**: text is never sent to a model unless the account has opted in.

### Ethical guardrails

- Every prediction is returned with a probability, a confidence label, an evidence
  count and a disclaimer.
- Traits are labelled `inferred` and shown with credible intervals.
- The questionnaire measures decision preferences only; no item screens for or
  infers any psychological or medical condition.
- The UI carries a standing note on every page.
- The system must not be used for medical, legal, employment, lending or other
  consequential determinations about a person. This is stated in the API
  description, in the prediction payload, and in the interface.

---

## Development

```bash
cd backend
make test        # 120 tests, 87% coverage
make lint        # ruff
make migrate     # alembic upgrade head
make seed        # demo history
make evaluate    # compare approaches on a time-based split
```

```bash
cd frontend
npm run typecheck
npm run build
```

Tests run against SQLite with no server, so `pytest` works on a clean checkout.
They cover factor extraction and negation, contrastive stance resolution
(including the regression where "take the stock options" and "take the cash
bonus" both read as approach), utility decomposition, the risk-damping sign,
Beta-posterior inference, geometric pooling, calibration, evaluation metrics, and
the full API lifecycle including cross-account isolation and account deletion.

Requires Python 3.12+ (Docker image uses 3.12; the source is 3.10-compatible) and
Node 20+.

---

## Known limitations

- **Factor extraction is lexical.** It reads the words that are there. Context
  never written down is invisible to it, and a word used in an unusual sense can
  mislead it. The LLM layer mitigates this when enabled; the extracted evidence
  phrases are always shown so a bad reading is visible rather than silent.
- **Small-*n* is the normal regime.** A person records tens of decisions, not
  thousands. Hence heavy regularisation, a statistical layer that refuses to fit
  below 12 decisions, gradient boosting held back until 40, and confidence gated
  on evidence volume.
- **Category coverage matters more than volume.** Accuracy degrades on decision
  types absent from the recorded history, which the per-category metrics expose.
- **Stated reasons are self-reports.** They are the richest signal available and
  are weighted accordingly, but they are how people explain themselves, not
  necessarily why they chose.
- **The seeded demo persona is synthetic.** It is internally consistent by
  construction and exists to exercise the pipeline. Numbers computed against it
  say nothing about real-world accuracy.
