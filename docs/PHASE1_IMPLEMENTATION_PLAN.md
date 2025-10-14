# Phase 1 Implementation Plan - Configuration Constants

**Status:** Ready for Implementation  
**Estimated Time:** 2-4 hours  
**Risk Level:** LOW (additive changes, no breaking changes)

---

## Overview

This plan implements Phase 1 of the configuration management system by:
1. Creating `src/config/constants.py` with all hardcoded values extracted
2. Updating `src/agents/ea_assistant.py` to use constants
3. Updating `src/safety/grounding.py` to use constants
4. Adding comprehensive documentation
5. Creating calibration tooling for future use

---

## File Changes Required

### 1. NEW FILE: `src/config/__init__.py`

**Action:** Create new file

**Content:**
```python
"""Configuration module for AInstein EA Assistant."""

from .constants import (
    CONFIDENCE,
    SEMANTIC_CONFIG,
    RANKING_CONFIG,
    CONTEXT_CONFIG,
    LANG_CONFIG,
    PERFORMANCE_CONFIG,
    COMPARISON_TERM_STOP_WORDS,
    QUERY_TERM_STOP_WORDS,
    REQUIRED_CITATION_PREFIXES,
    COMPARISON_PATTERNS,
    get_config_summary,
)

__all__ = [
    'CONFIDENCE',
    'SEMANTIC_CONFIG',
    'RANKING_CONFIG',
    'CONTEXT_CONFIG',
    'LANG_CONFIG',
    'PERFORMANCE_CONFIG',
    'COMPARISON_TERM_STOP_WORDS',
    'QUERY_TERM_STOP_WORDS',
    'REQUIRED_CITATION_PREFIXES',
    'COMPARISON_PATTERNS',
    'get_config_summary',
]
```

---

### 2. NEW FILE: `src/config/constants.py`

**Action:** Create new file (already provided in artifact)

**Status:** ✅ Complete - see artifact `constants_py`

---

### 3. MODIFY: `src/agents/ea_assistant.py`

**Action:** Replace all hardcoded values with constant imports

**Changes Required:**

#### Change 1: Add imports at top of file

```python
# Add after existing imports
from src.config.constants import (
    CONFIDENCE,
    SEMANTIC_CONFIG,
    RANKING_CONFIG,
    CONTEXT_CONFIG,
    LANG_CONFIG,
    PERFORMANCE_CONFIG,
    COMPARISON_TERM_STOP_WORDS,
    QUERY_TERM_STOP_WORDS,
    COMPARISON_PATTERNS,
)
```

#### Change 2: Replace confidence scores

**Find and replace:**

```python
# OLD (line ~850):
base_confidence = 0.95 if has_definition else 0.75

# NEW:
base_confidence = CONFIDENCE.KG_WITH_DEFINITION if has_definition else CONFIDENCE.KG_WITHOUT_DEFINITION
```

```python
# OLD (line ~880):
"confidence": 0.85,

# NEW:
"confidence": CONFIDENCE.TOGAF_DOCUMENTATION,
```

```python
# OLD (line ~910):
confidence=0.70,

# NEW:
confidence=CONFIDENCE.DOCUMENT_CHUNKS,
```

```python
# OLD (line ~950):
base_confidence = 0.75

# NEW:
base_confidence = CONFIDENCE.ARCHIMATE_ELEMENTS
```

```python
# OLD (line ~960):
base_confidence += 0.10  # Exact match bonus
base_confidence += 0.05  # Partial match bonus

# NEW:
base_confidence += CONFIDENCE.EXACT_TERM_MATCH_BONUS
base_confidence += CONFIDENCE.PARTIAL_TERM_MATCH_BONUS
```

#### Change 3: Replace semantic enhancement values

```python
# OLD (line ~1200):
min_score=0.40

# NEW:
min_score=SEMANTIC_CONFIG.MIN_SCORE_PRIMARY
```

```python
# OLD (line ~1220):
min_score=0.45

# NEW:
min_score=SEMANTIC_CONFIG.MIN_SCORE_CONTEXT
```

```python
# OLD (line ~1250):
top_k=5

# NEW:
top_k=SEMANTIC_CONFIG.TOP_K_PRIMARY
```

```python
# OLD (line ~1280):
return semantic_candidates[:3]

# NEW:
return semantic_candidates[:SEMANTIC_CONFIG.MAX_SEMANTIC_CANDIDATES]
```

```python
# OLD (line ~1150):
enable_semantic = os.getenv("ENABLE_SEMANTIC_ENHANCEMENT", "true").lower() == "true"

# NEW:
enable_semantic = os.getenv(
    "ENABLE_SEMANTIC_ENHANCEMENT", 
    str(SEMANTIC_CONFIG.ENABLED_BY_DEFAULT).lower()
).lower() == "true"
```

#### Change 4: Replace ranking configuration

```python
# OLD (line ~1450):
priority_map = {
    'definition': 100,
    'normal': 80,
    'context': 60
}
base_score = priority_map.get(priority, 50)
bonus = (confidence * 20) if confidence > 0.5 else (semantic_score * 20)

# NEW:
priority_map = {
    'definition': RANKING_CONFIG.PRIORITY_SCORE_DEFINITION,
    'normal': RANKING_CONFIG.PRIORITY_SCORE_NORMAL,
    'context': RANKING_CONFIG.PRIORITY_SCORE_CONTEXT
}
base_score = priority_map.get(priority, RANKING_CONFIG.PRIORITY_SCORE_FALLBACK)
bonus = (confidence * RANKING_CONFIG.CONFIDENCE_BONUS_MULTIPLIER) if confidence > RANKING_CONFIG.CONFIDENCE_BONUS_THRESHOLD else (semantic_score * RANKING_CONFIG.CONFIDENCE_BONUS_MULTIPLIER)
```

```python
# OLD (line ~1480):
return ranked[:10]

# NEW:
return ranked[:RANKING_CONFIG.MAX_TOTAL_CANDIDATES]
```

```python
# OLD (line ~1150):
if total_structured < 3:

# NEW:
if total_structured < RANKING_CONFIG.MIN_STRUCTURED_RESULTS:
```

#### Change 5: Replace context expansion values

```python
# OLD (line ~1550):
for message in session_data["messages"][-3:]:

# NEW:
for message in session_data["messages"][-CONTEXT_CONFIG.MAX_HISTORY_TURNS:]:
```

```python
# OLD (line ~1560):
previous_concepts = list(set(previous_concepts))[:5]

# NEW:
previous_concepts = list(set(previous_concepts))[:CONTEXT_CONFIG.MAX_CONCEPTS_FROM_HISTORY]
```

```python
# OLD (line ~1580):
top_k=2

# NEW:
top_k=SEMANTIC_CONFIG.TOP_K_CONTEXT
```

#### Change 6: Replace language detection values

```python
# OLD (line ~750):
temperature=0.1,
max_tokens=10

# NEW:
temperature=LANG_CONFIG.LLM_TEMPERATURE,
max_tokens=LANG_CONFIG.LLM_MAX_TOKENS
```

```python
# OLD (line ~800):
if dutch_chars > 2 or (long_words > len(words) * 0.2 and len(words) > 3):

# NEW:
if dutch_chars > LANG_CONFIG.DUTCH_CHAR_THRESHOLD or (long_words > len(words) * LANG_CONFIG.LONG_WORD_PERCENTAGE and len(words) > 3):
```

#### Change 7: Replace stop words

```python
# OLD (line ~1650):
stop_words = ['the', 'a', 'an', 'is', 'are', 'what', 'which', 'difference', 'in', 'of', 'to', 'from', 'with']

# NEW:
stop_words = COMPARISON_TERM_STOP_WORDS
```

```python
# OLD (line ~1100):
stop_words = {
    'what', 'is', 'the', 'a', 'an', 'of', 'for', 'in', 'on', 'at',
    'to', 'how', 'why', 'when', 'where', 'are', 'do', 'does'
}

# NEW:
stop_words = QUERY_TERM_STOP_WORDS
```

#### Change 8: Replace comparison patterns

```python
# OLD (line ~1700):
comparison_patterns = [
    r'(?:difference|differences)\s+between\s+(.+?)\s+and\s+(.+?)(?:\?|$)',
    r'(.+?)\s+(?:vs|versus|compared\s+to|vs\.)\s+(.+?)(?:\s+in\s+|\?|$)',
    r'(?:compare|comparison)\s+(?:of\s+)?(?:the\s+)?(.+?)\s+(?:with|and|to)\s+(.+?)(?:\?|$)',
    r'(.+?)\s+or\s+(.+?)(?:\s+-\s+|\?|$)',
]

# NEW:
comparison_patterns = COMPARISON_PATTERNS
```

#### Change 9: Replace performance values

```python
# OLD (line ~1250):
semantic_duration = (time.time() - semantic_start) * 1000
logger.info(f"Semantic enhancement took {semantic_duration:.2f}ms")

# NEW:
semantic_duration = (time.time() - semantic_start) * 1000
logger.info(f"Semantic enhancement took {semantic_duration:.{PERFORMANCE_CONFIG.TIME_PRECISION_DECIMALS}f}ms")
```

---

### 4. MODIFY: `src/safety/grounding.py`

**Action:** Use constants for citation patterns

**Changes Required:**

#### Change 1: Add imports

```python
# Add after existing imports
from src.config.constants import REQUIRED_CITATION_PREFIXES
```

#### Change 2: Replace hardcoded prefixes

```python
# OLD (line ~30):
REQUIRED_CITATION_PREFIXES = [
    "archi:id-",
    "skos:",
    "iec:",
    "togaf:adm:",
    "togaf:concepts:",
    "archimate:research:",
    "entsoe:",
    "lido:",
    "doc:",
    "external:"
]

# NEW:
# Import from constants instead
# (Remove the hardcoded list, use import instead)
```

**Note:** The `ENHANCED_CITATION_PATTERNS` in `grounding.py` can remain as-is since they are implementation-specific regex patterns, not configuration values.

---

### 5. NEW FILE: `docs/CONFIGURATION.md`

**Action:** Create documentation (already provided in artifact)

**Status:** ✅ Complete - see artifact `config_docs`

---

### 6. NEW FILE: `scripts/calibrate_config.py`

**Action:** Create calibration tool (already provided in artifact)

**Status:** ✅ Complete - see artifact `calibrate_config`

---

### 7. NEW FILE: `scripts/test_config_changes.py`

**Action:** Create test script for validating constant usage

**Content:**
```python
#!/usr/bin/env python3
"""
Test script to verify configuration constants work correctly.

This script validates:
1. All constants import correctly
2. Values are within expected ranges
3. No import errors or circular dependencies
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_constants_import():
    """Test that constants module imports without errors."""
    print("Testing constants import...")
    try:
        from src.config.constants import (
            CONFIDENCE,
            SEMANTIC_CONFIG,
            RANKING_CONFIG,
            CONTEXT_CONFIG,
            LANG_CONFIG,
            PERFORMANCE_CONFIG,
            get_config_summary
        )
        print("✅ All constants imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_constant_values():
    """Test that constant values are reasonable."""
    print("\nTesting constant values...")
    from src.config.constants import CONFIDENCE, SEMANTIC_CONFIG, RANKING_CONFIG
    
    checks = [
        ("Confidence threshold in [0,1]", 0 <= CONFIDENCE.HIGH_CONFIDENCE_THRESHOLD <= 1),
        ("Semantic score in [0,1]", 0 <= SEMANTIC_CONFIG.MIN_SCORE_PRIMARY <= 1),
        ("Ranking scores ordered", RANKING_CONFIG.PRIORITY_SCORE_DEFINITION > RANKING_CONFIG.PRIORITY_SCORE_CONTEXT),
        ("Top-K positive", SEMANTIC_CONFIG.TOP_K_PRIMARY > 0),
        ("Max candidates positive", RANKING_CONFIG.MAX_TOTAL_CANDIDATES > 0),
    ]
    
    all_passed = True
    for check_name, result in checks:
        if result:
            print(f"  ✅ {check_name}")
        else:
            print(f"  ❌ {check_name}")
            all_passed = False
    
    return all_passed

def test_config_summary():
    """Test that config summary generates correctly."""
    print("\nTesting config summary...")
    try:
        from src.config.constants import get_config_summary
        summary = get_config_summary()
        
        required_keys = ['version', 'confidence', 'semantic', 'ranking', 'warnings']
        missing = [k for k in required_keys if k not in summary]
        
        if missing:
            print(f"  ❌ Missing keys in summary: {missing}")
            return False
        
        print("  ✅ Config summary generated successfully")
        print(f"\n{'-'*60}")
        print("Configuration Summary:")
        print(f"{'-'*60}")
        import json
        print(json.dumps(summary, indent=2))
        print(f"{'-'*60}")
        return True
        
    except Exception as e:
        print(f"  ❌ Summary generation failed: {e}")
        return False

def test_ea_assistant_import():
    """Test that EA assistant imports with constants."""
    print("\nTesting EA assistant import with constants...")
    try:
        # This will fail if there are syntax errors in the updated code
        from src.agents.ea_assistant import ProductionEAAgent
        print("  ✅ EA assistant imports successfully with constants")
        return True
    except Exception as e:
        print(f"  ❌ EA assistant import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("="*70)
    print("CONFIGURATION CONSTANTS VALIDATION")
    print("="*70)
    
    tests = [
        test_constants_import,
        test_constant_values,
        test_config_summary,
        test_ea_assistant_import,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n❌ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED - Configuration constants working correctly")
        return 0
    else:
        print(f"\n❌ {total - passed} TESTS FAILED - Review errors above")
        return 1

if __name__ == '__main__':
    sys.exit(main())
```

---

## Implementation Steps

### Step 1: Create Configuration Structure (10 minutes)

```bash
# Create config directory
mkdir -p src/config
touch src/config/__init__.py

# Create constants file
# Copy content from artifact 'constants_py' to src/config/constants.py

# Create docs directory if not exists
mkdir -p docs

# Copy documentation
# Copy content from artifact 'config_docs' to docs/CONFIGURATION.md
```

### Step 2: Update ea_assistant.py (60-90 minutes)

**Process:**
1. Add imports at top of file
2. Use Find & Replace in your editor for each change listed above
3. Test after each major section

**Recommended order:**
1. Add imports ✓
2. Replace confidence scores ✓
3. Replace semantic values ✓
4. Replace ranking config ✓
5. Replace context config ✓
6. Replace language detection ✓
7. Replace stop words ✓
8. Replace comparison patterns ✓

**Validation after each step:**
```bash
# Check syntax
python -m py_compile src/agents/ea_assistant.py

# Check imports
python -c "from src.agents.ea_assistant import ProductionEAAgent; print('✅ Imports OK')"
```

### Step 3: Update grounding.py (10 minutes)

```bash
# Simple change - just use imported constants
# Update imports and remove hardcoded REQUIRED_CITATION_PREFIXES
```

### Step 4: Create Test Script (5 minutes)

```bash
# Create test script
# Copy content above to scripts/test_config_changes.py
chmod +x scripts/test_config_changes.py
```

### Step 5: Create Calibration Tool (5 minutes)

```bash
# Create calibration tool
# Copy content from artifact 'calibrate_config' to scripts/calibrate_config.py
chmod +x scripts/calibrate_config.py
```

### Step 6: Validation (15 minutes)

```bash
# Run test script
python scripts/test_config_changes.py

# Run existing tests
pytest tests/ -v -k "not integration"

# Test with sample query
python scripts/test_sample_query.py "What is reactive power?"
```

---

## Testing Checklist

### Unit Tests
- [ ] `test_constants_import()` - Constants import correctly
- [ ] `test_constant_values()` - Values within valid ranges
- [ ] `test_config_summary()` - Summary generates correctly
- [ ] `test_ea_assistant_import()` - EA assistant imports with constants

### Integration Tests
- [ ] Simple query works ("What is reactive power?")
- [ ] Comparison query works ("active vs reactive power")
- [ ] Follow-up query works (conversation context)
- [ ] Semantic enhancement activates correctly
- [ ] Grounding check passes

### Manual Verification
- [ ] No hardcoded values remain in ea_assistant.py
- [ ] All imports resolve correctly
- [ ] Configuration summary shows correct values
- [ ] Documentation is complete

---

## Rollback Plan

If issues occur:

```bash
# Quick rollback
git checkout main -- src/agents/ea_assistant.py src/safety/grounding.py

# Remove new files
rm -rf src/config/
rm docs/CONFIGURATION.md
rm scripts/calibrate_config.py
```

---

## Success Criteria

✅ **Phase 1 Complete When:**
1. All hardcoded values extracted to `constants.py`
2. `ea_assistant.py` uses constants throughout
3. `grounding.py` uses constants
4. All tests pass
5. Documentation complete
6. Calibration tool available for future use

---

## Next Steps (Phase 2)

After Phase 1 is complete and stable:

1. Create `config/system_config.yaml`
2. Implement `src/config/config_loader.py`
3. Add environment variable override support
4. Update code to use config loader instead of constants
5. Add hot-reload capability

**Estimated time for Phase 2:** 1-2 weeks

---

## Commit Strategy

Commit after each major step:

```bash
# Commit 1
git add src/config/
git commit -m "feat(config): create configuration constants module

- Extract all hardcoded values to constants.py
- Add dataclass-based configuration
- Include validation on import
- Add configuration summary function

Refs: HARDCODED_SOLUTIONS_AUDIT.md"

# Commit 2
git add src/agents/ea_assistant.py
git commit -m "refactor(agent): replace hardcoded values with configuration constants

- Use CONFIDENCE constants for scoring
- Use SEMANTIC_CONFIG for semantic search
- Use RANKING_CONFIG for prioritization
- Use CONTEXT_CONFIG for conversation memory
- Use LANG_CONFIG for language detection

All values now centralized in src/config/constants.py"

# Commit 3
git add src/safety/grounding.py
git commit -m "refactor(safety): use configuration constants for citation prefixes"

# Commit 4
git add docs/CONFIGURATION.md
git commit -m "docs: add comprehensive configuration guide

- Document all configuration categories
- Explain calibration requirements
- Provide tuning guidance
- Include monitoring recommendations"

# Commit 5
git add scripts/calibrate_config.py scripts/test_config_changes.py
git commit -m "feat(tools): add configuration calibration and testing tools

- calibrate_config.py: Statistical parameter optimization
- test_config_changes.py: Validation script for constants
- Support for empirical threshold tuning"

# Final commit
git commit -m "chore: Phase 1 configuration management complete

All hardcoded values extracted and centralized.
System ready for Phase 2 (YAML configuration)."
```

---

**Ready to implement!** 🚀
