# AInstein Configuration Guide

**Version:** 1.0.0  
**Last Updated:** 2025-10-14  
**Status:** 🟡 PROTOTYPE VALUES - Production validation required

---

## ⚠️ CRITICAL: Current Configuration Status

### Configuration Maturity Level: **PROTOTYPE**

The current configuration values are **educated guesses** based on:
- ✅ Limited prototype testing (~50 queries)
- ✅ Algorithmic validation (logic works correctly)
- ❌ NO production validation
- ❌ NO statistical calibration
- ❌ NO A/B testing
- ❌ NO user feedback analysis

**These values WORK algorithmically but are NOT optimized for production.**

---

## Configuration Architecture

### File Structure

```
config/
├── constants.py              # Phase 1: Hardcoded constants (current)
├── system_config.yaml        # Phase 2: YAML configuration (planned)
└── calibrated_config.yaml    # Phase 3: Empirically tuned (future)

src/config/
├── __init__.py
├── constants.py              # Current: All constants defined here
└── config_loader.py          # Future: Dynamic config loading
```

### Current State (Phase 1)

**Location:** `src/config/constants.py`

All configuration values are defined as dataclass constants:

```python
from src.config.constants import (
    CONFIDENCE,           # Confidence thresholds
    SEMANTIC_CONFIG,      # Semantic search settings
    RANKING_CONFIG,       # Ranking and prioritization
    CONTEXT_CONFIG,       # Context expansion settings
    LANG_CONFIG          # Language detection
)

# Usage example
if confidence_score >= CONFIDENCE.HIGH_CONFIDENCE_THRESHOLD:
    # No human review needed
    pass
```

---

## Configuration Categories

### 1. Confidence Scoring (`CONFIDENCE`)

**Purpose:** Determines when responses require human review

**Critical Parameters:**

| Parameter | Current Value | Meaning | Risk if Wrong |
|-----------|---------------|---------|---------------|
| `HIGH_CONFIDENCE_THRESHOLD` | 0.75 | Above this = no review | Too low: bad responses approved<br>Too high: good responses flagged |
| `KG_WITH_DEFINITION` | 0.95 | Confidence for KG with definition | Overconfidence in structured data |
| `KG_WITHOUT_DEFINITION` | 0.75 | Confidence for KG without definition | May trust incomplete data |

**Known Issues:**
- ❌ No empirical validation of thresholds
- ❌ No precision/recall analysis
- ❌ Thresholds are arbitrary estimates
- ❌ No correlation with actual quality

**How to Validate:**

```python
# Collect data: (predicted_confidence, actual_quality_label)
dataset = [
    (0.85, True),   # High confidence, actually good
    (0.60, False),  # Medium confidence, actually bad
    # ... collect 1000+ examples
]

# Find optimal threshold
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(
    actual_quality, predicted_confidence
)
# Find threshold that maximizes F1 score
```

**Tuning Guide:**

```python
# If too many bad responses are auto-approved:
CONFIDENCE.HIGH_CONFIDENCE_THRESHOLD = 0.80  # Increase (was 0.75)

# If too many good responses are flagged unnecessarily:
CONFIDENCE.HIGH_CONFIDENCE_THRESHOLD = 0.70  # Decrease (was 0.75)
```

---

### 2. Semantic Enhancement (`SEMANTIC_CONFIG`)

**Purpose:** Controls semantic search quality and quantity

**Critical Parameters:**

| Parameter | Current Value | Meaning | Risk if Wrong |
|-----------|---------------|---------|---------------|
| `MIN_SCORE_PRIMARY` | 0.40 | Minimum cosine similarity | Too low: irrelevant results<br>Too high: miss valid results |
| `TOP_K_PRIMARY` | 5 | Max semantic results | Too many: noise<br>Too few: miss context |
| `MAX_SEMANTIC_CANDIDATES` | 3 | Post-filter limit | Controls LLM context size |

**Model Dependency:**

Current thresholds are tuned for **all-MiniLM-L6-v2**:
- Embedding dimension: 384
- Similarity metric: Cosine similarity
- Range: [0, 1]

⚠️ **If you change the embedding model, you MUST recalibrate these thresholds.**

**Known Issues:**
- ❌ No precision@k / recall@k analysis
- ❌ Threshold chosen empirically without statistics
- ❌ May include false positives (irrelevant but high similarity)
- ❌ May exclude true positives (relevant but lower similarity)

**How to Validate:**

```python
# For each test query, get semantic results at different thresholds
thresholds = [0.30, 0.35, 0.40, 0.45, 0.50]
results = {}

for threshold in thresholds:
    results[threshold] = {
        'precision': calculate_precision_at_k(threshold, k=5),
        'recall': calculate_recall_at_k(threshold, k=5),
        'f1': calculate_f1_at_k(threshold, k=5)
    }

# Choose threshold with highest F1 score
optimal_threshold = max(results, key=lambda t: results[t]['f1'])
```

**Tuning Guide:**

```python
# For higher precision (fewer, more relevant results):
SEMANTIC_CONFIG.MIN_SCORE_PRIMARY = 0.50  # Increase (was 0.40)

# For higher recall (more results, some irrelevant):
SEMANTIC_CONFIG.MIN_SCORE_PRIMARY = 0.35  # Decrease (was 0.40)

# If LLM is overwhelmed with too much context:
SEMANTIC_CONFIG.MAX_SEMANTIC_CANDIDATES = 2  # Decrease (was 3)

# If responses lack sufficient context:
SEMANTIC_CONFIG.MAX_SEMANTIC_CANDIDATES = 5  # Increase (was 3)
```

---

### 3. Ranking and Prioritization (`RANKING_CONFIG`)

**Purpose:** Determines how candidates are ranked for LLM

**Critical Parameters:**

| Parameter | Current Value | Meaning | Risk if Wrong |
|-----------|---------------|---------|---------------|
| `PRIORITY_SCORE_DEFINITION` | 100 | KG with definition priority | Wrong sources prioritized |
| `PRIORITY_SCORE_NORMAL` | 80 | Standard source priority | Wrong ranking order |
| `PRIORITY_SCORE_CONTEXT` | 60 | Context/semantic priority | Noise prioritized over signal |
| `MAX_TOTAL_CANDIDATES` | 10 | Max candidates to LLM | Too many: overwhelm LLM<br>Too few: miss info |

**Known Issues:**
- ❌ Priority scores are COMPLETELY ARBITRARY (100/80/60/50)
- ❌ No mathematical basis for these values
- ❌ No validation that this scale works better than others
- ❌ No user feedback on ranking quality
- ❌ No ranking metrics (MRR, NDCG) calculated

**Why These Numbers Were Chosen:**

Honestly? **They weren't.** They are:
- Spaced apart enough to create clear separation (100 vs 80 vs 60)
- Easy to understand (round numbers)
- Seemed reasonable during prototyping

**There is NO empirical evidence that 100/80/60 is better than 90/70/50 or any other scale.**

**How to Validate:**

```python
# Collect data: (query, clicked_result_rank, all_result_ranks)
# Then calculate ranking metrics:

from sklearn.metrics import ndcg_score

# NDCG (Normalized Discounted Cumulative Gain)
# Higher = better ranking
ndcg = ndcg_score(actual_relevance, predicted_ranking)

# MRR (Mean Reciprocal Rank)
# Where was the first relevant result?
mrr = sum(1/rank for rank in first_relevant_ranks) / len(queries)

# Try different priority scales and compare NDCG/MRR
```

**Tuning Guide:**

```python
# If users consistently prefer semantic results over structured:
RANKING_CONFIG.PRIORITY_SCORE_CONTEXT = 70  # Increase (was 60)

# If users ignore context results:
RANKING_CONFIG.PRIORITY_SCORE_CONTEXT = 50  # Decrease (was 60)

# If responses are overwhelming with too much info:
RANKING_CONFIG.MAX_TOTAL_CANDIDATES = 7  # Decrease (was 10)
```

---

### 4. Context Expansion (`CONTEXT_CONFIG`)

**Purpose:** Controls conversation memory depth

**Critical Parameters:**

| Parameter | Current Value | Meaning | Risk if Wrong |
|-----------|---------------|---------|---------------|
| `MAX_HISTORY_TURNS` | 3 | Look back N turns | Too few: miss context<br>Too many: add noise |
| `MAX_CONCEPTS_FROM_HISTORY` | 5 | Max concepts extracted | Controls context size |

**Known Issues:**
- ❌ "3 turns" chosen arbitrarily
- ❌ No testing of 2 vs 3 vs 4 vs 5 turns
- ❌ No conversation quality metrics
- ❌ May include irrelevant historical context

**Tuning Guide:**

```python
# If follow-up questions lack context:
CONTEXT_CONFIG.MAX_HISTORY_TURNS = 5  # Increase (was 3)

# If follow-up responses include irrelevant info from history:
CONTEXT_CONFIG.MAX_HISTORY_TURNS = 2  # Decrease (was 3)
```

---

### 5. Language Detection (`LANG_CONFIG`)

**Purpose:** Detect query language (English/Dutch)

**Critical Parameters:**

| Parameter | Current Value | Meaning | Risk if Wrong |
|-----------|---------------|---------|---------------|
| `DUTCH_CHAR_THRESHOLD` | 2 | Min Dutch chars | Misclassify language |
| `LONG_WORD_PERCENTAGE` | 0.20 | % long words for Dutch | False positives/negatives |

**Known Issues:**
- ❌ Only supports English and Dutch
- ❌ Heuristics are COMPLETELY MADE UP
- ❌ No statistical basis for thresholds
- ❌ No validation on real queries
- ⚠️ **This is the most arbitrary configuration in the entire system**

**Why These Numbers?**

Because Dutch has more compound words (longer) and specific character combinations (ij, oe, ui). But the exact thresholds (2 chars, 20% long words) are **pure guesswork**.

---

## Monitoring and Tuning

### Key Metrics to Track

#### 1. Confidence Calibration

**Goal:** Confidence scores should correlate with actual quality

**Metric:** Confidence vs Quality Plot

```python
# Should be linear relationship
plt.scatter(predicted_confidence, actual_quality)
plt.plot([0,1], [0,1], 'r--')  # Perfect calibration line
plt.xlabel('Predicted Confidence')
plt.ylabel('Actual Quality')
plt.title('Confidence Calibration')
```

**What to look for:**
- Points on red line = perfect calibration
- Points above line = overconfidence
- Points below line = underconfidence

#### 2. Semantic Threshold Effectiveness

**Goal:** Find optimal precision/recall tradeoff

**Metric:** Precision-Recall Curve

```python
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(
    is_relevant, similarity_scores
)

plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Semantic Search P-R Curve')
```

**What to look for:**
- Find threshold with highest F1 score
- Balance between precision and recall

#### 3. Ranking Quality

**Goal:** Validate ranking algorithm effectiveness

**Metrics:**
- **MRR (Mean Reciprocal Rank)**: Average of 1/rank for first relevant result
- **NDCG@10**: Ranking quality considering position and relevance
- **Click-through rate**: Which ranked results users actually use

---

## Configuration Change Process

### 1. Local Testing

```python
# Edit src/config/constants.py
CONFIDENCE.HIGH_CONFIDENCE_THRESHOLD = 0.80  # Changed from 0.75

# Run tests
pytest tests/ -v

# Test with sample queries
python scripts/test_config_changes.py
```

### 2. A/B Testing (Recommended)

```python
# Create two configurations
CONFIG_A = ConfidenceThresholds(HIGH_CONFIDENCE_THRESHOLD=0.75)
CONFIG_B = ConfidenceThresholds(HIGH_CONFIDENCE_THRESHOLD=0.80)

# Randomly assign users
if user_id % 2 == 0:
    use_config(CONFIG_A)
else:
    use_config(CONFIG_B)

# Measure metrics for both groups
# Compare: response quality, review rate, user satisfaction
```

### 3. Gradual Rollout

```python
# Week 1: 10% of traffic
if random.random() < 0.10:
    use_new_config()
    
# Week 2: 50% of traffic
# Week 3: 100% of traffic
```

---

## Recommended Review Schedule

### Weekly Reviews
- ✅ Check error logs for configuration-related issues
- ✅ Review edge cases and failures
- ✅ Monitor response times

### Monthly Reviews
- ✅ Analyze quality metrics
- ✅ Compare actual vs predicted confidence
- ✅ Review user feedback
- ✅ Adjust thresholds if needed

### Quarterly Reviews
- ✅ Full configuration audit
- ✅ Recalibration with accumulated data
- ✅ A/B test major changes
- ✅ Update documentation

---

## Migration Path to Production

### Phase 1: Prototype Constants (Current) ✅

**Status:** Complete  
**What:** All values in `constants.py`  
**Limitations:** Hardcoded, not configurable

### Phase 2: YAML Configuration (Next)

**Goal:** External configuration file  
**Timeline:** 1-2 weeks  
**Deliverables:**
- `config/system_config.yaml`
- `src/config/config_loader.py`
- Environment variable overrides

### Phase 3: Empirical Calibration (Future)

**Goal:** Statistically validated parameters  
**Timeline:** 2-3 months  
**Requirements:**
- Collect 1000+ labeled queries
- Human quality evaluation
- Run calibration tool
- Update configuration with optimal values

---

## Troubleshooting

### Problem: Too many responses flagged for review

**Symptom:** `requires_human_review = True` too often

**Solution:**
```python
# Lower the confidence threshold
CONFIDENCE.HIGH_CONFIDENCE_THRESHOLD = 0.70  # From 0.75
```

### Problem: Bad responses being auto-approved

**Symptom:** Low quality responses with high confidence

**Solution:**
```python
# Raise the confidence threshold
CONFIDENCE.HIGH_CONFIDENCE_THRESHOLD = 0.80  # From 0.75

# Or lower source confidence scores
CONFIDENCE.KG_WITHOUT_DEFINITION = 0.70  # From 0.75
```

### Problem: Semantic search returning irrelevant results

**Symptom:** Low relevance despite high similarity scores

**Solution:**
```python
# Raise similarity threshold
SEMANTIC_CONFIG.MIN_SCORE_PRIMARY = 0.50  # From 0.40

# Or reduce result count
SEMANTIC_CONFIG.MAX_SEMANTIC_CANDIDATES = 2  # From 3
```

### Problem: Comparison queries not finding distinct concepts

**Symptom:** Same concept compared against itself

**Solution:**
```python
# Already fixed in Phase 1
# But can adjust semantic fallback threshold
SEMANTIC_CONFIG.MIN_SCORE_COMPARISON = 0.50  # From 0.45
```

---

## Summary: Current Configuration Status

| Category | Maturity | Validated | Production Ready |
|----------|----------|-----------|------------------|
| **Confidence Scoring** | 🟡 Prototype | ❌ No | ❌ No |
| **Semantic Enhancement** | 🟡 Prototype | ❌ No | ❌ No |
| **Ranking System** | 🟡 Prototype | ❌ No | ❌ No |
| **Context Expansion** | 🟡 Prototype | ❌ No | ❌ No |
| **Language Detection** | 🔴 Guesswork | ❌ No | ❌ No |

### Overall Assessment: **NOT PRODUCTION READY**

**But:** Algorithms work correctly, just need parameter tuning.

**Next Steps:**
1. ✅ Phase 1 Complete: Constants extracted
2. ⏳ Phase 2: YAML configuration system
3. ⏳ Phase 3: Empirical calibration with real data

---

## References

### Academic Papers on Configuration Tuning

1. **Confidence Calibration:**
   - Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017)

2. **Information Retrieval Metrics:**
   - Manning et al., "Introduction to Information Retrieval" (2008)

3. **Ranking Evaluation:**
   - Järvelin & Kekäläinen, "Cumulated Gain-Based Evaluation of IR Techniques" (2002)

### Tools

- **scikit-learn**: Precision-recall curves, ROC analysis
- **matplotlib**: Visualization of calibration curves
- **scipy**: Statistical tests for A/B comparisons

---

**Last Updated:** 2025-10-14  
**Review Date:** 2026-01-14 (Quarterly)  
**Owner:** EA Assistant Development Team
