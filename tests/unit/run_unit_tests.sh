#!/bin/bash

# Unit Test Runner for Fake Citation Fix
# Tests grounding, citation validation, and knowledge source integration

echo "=================================================="
echo "Running Unit Tests for Fake Citation Fix"
echo "=================================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track results
PASSED=0
FAILED=0

echo "📋 Test Suite: Citation Validation"
echo "--------------------------------------------------"

# Test 1: Grounding Check
echo -e "\n${YELLOW}Test 1: Grounding Check${NC}"
if pytest tests/unit/test_grounding.py -v --tb=short; then
    echo -e "${GREEN}✓ Grounding tests passed${NC}"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗ Grounding tests failed${NC}"
    FAILED=$((FAILED + 1))
fi

# Test 2: Citation Validator
echo -e "\n${YELLOW}Test 2: Citation Validator${NC}"
if pytest tests/unit/test_citation_validator.py -v --tb=short; then
    echo -e "${GREEN}✓ Citation validator tests passed${NC}"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗ Citation validator tests failed${NC}"
    FAILED=$((FAILED + 1))
fi

# Summary
echo ""
echo "=================================================="
echo "Test Results Summary"
echo "=================================================="
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All unit tests passed!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Run integration tests: pytest tests/integration/test_fake_citation_prevention.py -v"
    echo "  2. Test with real queries: python test_query.py"
    echo "  3. Start web demo: python run_web_demo.py"
    exit 0
else
    echo -e "${RED}✗ Some tests failed. Please review the errors above.${NC}"
    exit 1
fi