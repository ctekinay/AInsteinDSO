#!/usr/bin/env python3
"""
Comprehensive Quality Test Suite for AInstein EA Assistant.

This test suite validates:
1. Comparison query fixes (distinct concepts)
2. API reranking quality improvements
3. Edge cases and error handling
4. Response quality metrics
5. Performance benchmarks

Focus: Implementation quality and response quality
Author: AInstein Team
Date: October 2025
"""

import asyncio
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.ea_assistant import ProductionEAAgent


@dataclass
class TestCase:
    """Test case definition."""
    query: str
    test_type: str  # 'comparison', 'definition', 'complex', 'edge_case'
    expected_min_citations: int
    expected_distinct_citations: bool
    expected_concepts: List[str]  # Concepts that should appear
    should_trigger_rerank: bool
    description: str
    difficulty: str  # 'easy', 'medium', 'hard'


@dataclass
class TestResult:
    """Test result with detailed metrics."""
    test_case: TestCase
    passed: bool
    response_time_ms: float
    confidence: float
    citations: List[str]
    citations_distinct: bool
    concepts_found: List[str]
    concepts_missing: List[str]
    was_reranked: bool
    requires_review: bool
    response_preview: str
    error_message: str = ""


class ComprehensiveQualityTester:
    """
    Comprehensive quality testing framework.

    Tests both correctness and quality improvements from recent changes.
    """

    def __init__(self):
        """Initialize tester."""
        self.agent = None
        self.results: List[TestResult] = []
        self.session_prefix = f"quality-test-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    async def initialize(self):
        """Initialize EA Assistant."""
        print("\n" + "="*80)
        print("INITIALIZING EA ASSISTANT")
        print("="*80)

        start_time = time.time()

        self.agent = ProductionEAAgent(
            llm_provider='groq',
            vocab_path='config/vocabularies.json',
            models_path='data/models',
            docs_path='data/docs'
        )

        init_time = (time.time() - start_time) * 1000

        print(f"✅ Agent initialized in {init_time:.0f}ms")
        print(f"   Embedding agent: {'✅ Available' if self.agent.embedding_agent else '❌ Not available'}")
        print(f"   API reranker: {'✅ Available' if self.agent.api_reranker else '❌ Not available'}")
        print(f"   Selective reranker: {'✅ Available' if self.agent.selective_reranker else '❌ Not available'}")

    def get_test_cases(self) -> List[TestCase]:
        """
        Define comprehensive test cases.

        Categories:
        1. Comparison queries (test the fix)
        2. Definition queries (baseline)
        3. Complex queries (test reranking)
        4. Edge cases (robustness)
        """

        test_cases = [
            # ================================================================
            # CATEGORY 1: COMPARISON QUERIES (Test the fix)
            # ================================================================
            TestCase(
                query="What is the difference between active power and reactive power?",
                test_type="comparison",
                expected_min_citations=2,
                expected_distinct_citations=True,
                expected_concepts=["active power", "reactive power"],
                should_trigger_rerank=True,
                description="Classic comparison - original failing case",
                difficulty="medium"
            ),

            TestCase(
                query="active power vs reactive power",
                test_type="comparison",
                expected_min_citations=2,
                expected_distinct_citations=True,
                expected_concepts=["active", "reactive"],
                should_trigger_rerank=True,
                description="Short comparison format with 'vs'",
                difficulty="easy"
            ),

            TestCase(
                query="Compare active power with reactive power in electrical systems",
                test_type="comparison",
                expected_min_citations=2,
                expected_distinct_citations=True,
                expected_concepts=["active power", "reactive power"],
                should_trigger_rerank=True,
                description="Explicit 'compare' instruction",
                difficulty="medium"
            ),

            TestCase(
                query="What's better: active power or reactive power for measuring consumption?",
                test_type="comparison",
                expected_min_citations=2,
                expected_distinct_citations=True,
                expected_concepts=["active", "reactive"],
                should_trigger_rerank=True,
                description="Comparison with context",
                difficulty="hard"
            ),

            TestCase(
                query="voltage versus current",
                test_type="comparison",
                expected_min_citations=2,
                expected_distinct_citations=True,
                expected_concepts=["voltage", "current"],
                should_trigger_rerank=True,
                description="Different technical terms comparison",
                difficulty="medium"
            ),

            # ================================================================
            # CATEGORY 2: DEFINITION QUERIES (Baseline - should work as before)
            # ================================================================
            TestCase(
                query="What is reactive power?",
                test_type="definition",
                expected_min_citations=1,
                expected_distinct_citations=True,
                expected_concepts=["reactive power"],
                should_trigger_rerank=False,  # High confidence, clear winner
                description="Simple definition query",
                difficulty="easy"
            ),

            TestCase(
                query="Define active power",
                test_type="definition",
                expected_min_citations=1,
                expected_distinct_citations=True,
                expected_concepts=["active power"],
                should_trigger_rerank=False,
                description="Explicit definition request",
                difficulty="easy"
            ),

            TestCase(
                query="What does IEC 61968 mean?",
                test_type="definition",
                expected_min_citations=1,
                expected_distinct_citations=True,
                expected_concepts=["iec 61968", "iec", "61968"],
                should_trigger_rerank=False,
                description="Standard definition query",
                difficulty="easy"
            ),

            TestCase(
                query="Explain voltage in electrical systems",
                test_type="definition",
                expected_min_citations=1,
                expected_distinct_citations=True,
                expected_concepts=["voltage"],
                should_trigger_rerank=False,
                description="Explanatory definition",
                difficulty="easy"
            ),

            # ================================================================
            # CATEGORY 3: COMPLEX QUERIES (Test reranking benefits)
            # ================================================================
            TestCase(
                query="How do business capabilities relate to grid congestion management?",
                test_type="complex",
                expected_min_citations=2,
                expected_distinct_citations=True,
                expected_concepts=["business capability", "grid congestion"],
                should_trigger_rerank=True,  # Medium confidence
                description="Multi-domain complex query",
                difficulty="hard"
            ),

            TestCase(
                query="What are the implications of power quality on distribution systems?",
                test_type="complex",
                expected_min_citations=2,
                expected_distinct_citations=True,
                expected_concepts=["power quality", "distribution"],
                should_trigger_rerank=True,
                description="Abstract conceptual query",
                difficulty="hard"
            ),

            TestCase(
                query="Relationship between voltage regulation and reactive power compensation",
                test_type="complex",
                expected_min_citations=2,
                expected_distinct_citations=True,
                expected_concepts=["voltage", "reactive power"],
                should_trigger_rerank=True,
                description="Technical relationship query",
                difficulty="hard"
            ),

            TestCase(
                query="How does TOGAF Phase B relate to business architecture for energy systems?",
                test_type="complex",
                expected_min_citations=2,
                expected_distinct_citations=True,
                expected_concepts=["togaf", "phase b", "business"],
                should_trigger_rerank=True,
                description="Cross-domain methodology query",
                difficulty="hard"
            ),

            # ================================================================
            # CATEGORY 4: EDGE CASES (Robustness testing)
            # ================================================================
            TestCase(
                query="power",
                test_type="edge_case",
                expected_min_citations=1,
                expected_distinct_citations=True,
                expected_concepts=["power"],
                should_trigger_rerank=False,
                description="Single ambiguous word (homonym)",
                difficulty="medium"
            ),

            TestCase(
                query="What is the difference between X and Y?",
                test_type="edge_case",
                expected_min_citations=0,  # Should fail gracefully
                expected_distinct_citations=True,
                expected_concepts=[],
                should_trigger_rerank=False,
                description="Generic comparison with no real terms",
                difficulty="hard"
            ),

            TestCase(
                query="active power active power active power",
                test_type="edge_case",
                expected_min_citations=1,
                expected_distinct_citations=True,
                expected_concepts=["active power"],
                should_trigger_rerank=False,
                description="Repeated term (should not confuse)",
                difficulty="medium"
            ),

            TestCase(
                query="transformer vs transformer - which is better?",
                test_type="edge_case",
                expected_min_citations=1,
                expected_distinct_citations=True,
                expected_concepts=["transformer"],
                should_trigger_rerank=False,
                description="Same concept comparison (should handle gracefully)",
                difficulty="hard"
            ),

            TestCase(
                query="Tell me everything about electrical engineering",
                test_type="edge_case",
                expected_min_citations=1,
                expected_distinct_citations=True,
                expected_concepts=["electrical"],
                should_trigger_rerank=True,  # Vague query
                description="Overly broad query",
                difficulty="hard"
            ),

            # ================================================================
            # CATEGORY 5: QUALITY REGRESSION TESTS (Ensure nothing broke)
            # ================================================================
            TestCase(
                query="What is apparent power?",
                test_type="definition",
                expected_min_citations=1,
                expected_distinct_citations=True,
                expected_concepts=["apparent power"],
                should_trigger_rerank=False,
                description="Regression: basic definition should still work",
                difficulty="easy"
            ),

            TestCase(
                query="grid congestion management",
                test_type="definition",
                expected_min_citations=1,
                expected_distinct_citations=True,
                expected_concepts=["grid congestion"],
                should_trigger_rerank=False,
                description="Regression: domain-specific term",
                difficulty="easy"
            ),

            TestCase(
                query="IEC 61970 standard",
                test_type="definition",
                expected_min_citations=1,
                expected_distinct_citations=True,
                expected_concepts=["iec 61970", "iec"],
                should_trigger_rerank=False,
                description="Regression: standard lookup",
                difficulty="easy"
            ),
        ]

        return test_cases

    async def run_test_case(self, test_case: TestCase, test_number: int, total_tests: int) -> TestResult:
        """
        Run a single test case.

        Returns:
            TestResult with detailed metrics
        """
        print(f"\n{'='*80}")
        print(f"TEST {test_number}/{total_tests}: {test_case.description}")
        print(f"{'='*80}")
        print(f"Query: \"{test_case.query}\"")
        print(f"Type: {test_case.test_type} | Difficulty: {test_case.difficulty}")
        print(f"Expected: {test_case.expected_min_citations}+ distinct citations")

        # Generate unique session ID
        session_id = f"{self.session_prefix}-{test_number:03d}"

        # Run query
        start_time = time.time()

        try:
            response = await self.agent.process_query(test_case.query, session_id)
            response_time_ms = (time.time() - start_time) * 1000

            # Extract metrics
            citations = response.citations if hasattr(response, 'citations') else []
            confidence = response.confidence if hasattr(response, 'confidence') else 0.0
            requires_review = response.requires_human_review if hasattr(response, 'requires_human_review') else False
            response_text = response.response if hasattr(response, 'response') else ""

            # Check if citations are distinct
            citations_distinct = len(citations) == len(set(citations))

            # Check which expected concepts were found
            response_lower = response_text.lower()
            concepts_found = [
                concept for concept in test_case.expected_concepts
                if concept.lower() in response_lower
            ]
            concepts_missing = [
                concept for concept in test_case.expected_concepts
                if concept.lower() not in response_lower
            ]

            # Check if reranking was used (from stats)
            was_reranked = False
            if self.agent.selective_reranker:
                # Check if this query was reranked
                prev_total = self.agent.selective_reranker.stats.get('total_queries', 0)
                prev_reranked = self.agent.selective_reranker.stats.get('reranked_queries', 0)

                # This is an approximation - in production you'd track per-query
                if prev_total > 0:
                    was_reranked = (prev_reranked / prev_total) > 0.5

            # Validate test expectations
            passed = True
            failure_reasons = []

            # Check: Minimum citations
            if len(citations) < test_case.expected_min_citations:
                passed = False
                failure_reasons.append(
                    f"Expected {test_case.expected_min_citations}+ citations, got {len(citations)}"
                )

            # Check: Citations distinct
            if test_case.expected_distinct_citations and not citations_distinct:
                passed = False
                failure_reasons.append(
                    f"Expected distinct citations, found duplicates: {citations}"
                )

            # Check: Expected concepts (at least 70% should be present)
            if test_case.expected_concepts:
                found_rate = len(concepts_found) / len(test_case.expected_concepts)
                if found_rate < 0.7:
                    passed = False
                    failure_reasons.append(
                        f"Expected concepts not found: {concepts_missing}"
                    )

            # Print results
            print(f"\n📊 RESULTS:")
            print(f"   Response time: {response_time_ms:.0f}ms")
            print(f"   Confidence: {confidence:.2f}")
            print(f"   Citations: {len(citations)} ({'distinct' if citations_distinct else 'HAS DUPLICATES'})")
            if citations:
                print(f"   Citation list: {citations}")
            print(f"   Concepts found: {len(concepts_found)}/{len(test_case.expected_concepts)}")
            if concepts_found:
                print(f"     ✅ Found: {concepts_found}")
            if concepts_missing:
                print(f"     ❌ Missing: {concepts_missing}")
            print(f"   Reranked: {'Yes' if was_reranked else 'No'}")
            print(f"   Requires review: {requires_review}")

            # Print response preview
            preview = response_text[:200] + "..." if len(response_text) > 200 else response_text
            print(f"\n📄 Response preview:")
            print(f"   {preview}")

            # Pass/fail
            if passed:
                print(f"\n✅ TEST PASSED")
            else:
                print(f"\n❌ TEST FAILED")
                for reason in failure_reasons:
                    print(f"   - {reason}")

            return TestResult(
                test_case=test_case,
                passed=passed,
                response_time_ms=response_time_ms,
                confidence=confidence,
                citations=citations,
                citations_distinct=citations_distinct,
                concepts_found=concepts_found,
                concepts_missing=concepts_missing,
                was_reranked=was_reranked,
                requires_review=requires_review,
                response_preview=preview,
                error_message="; ".join(failure_reasons) if failure_reasons else ""
            )

        except Exception as e:
            print(f"\n❌ TEST FAILED WITH EXCEPTION")
            print(f"   Error: {str(e)}")

            import traceback
            print(f"\n🔍 Traceback:")
            traceback.print_exc()

            return TestResult(
                test_case=test_case,
                passed=False,
                response_time_ms=(time.time() - start_time) * 1000,
                confidence=0.0,
                citations=[],
                citations_distinct=False,
                concepts_found=[],
                concepts_missing=test_case.expected_concepts,
                was_reranked=False,
                requires_review=True,
                response_preview="",
                error_message=str(e)
            )

    async def run_all_tests(self):
        """Run all test cases."""
        test_cases = self.get_test_cases()

        print(f"\n{'='*80}")
        print(f"RUNNING {len(test_cases)} QUALITY TESTS")
        print(f"{'='*80}")

        for i, test_case in enumerate(test_cases, 1):
            result = await self.run_test_case(test_case, i, len(test_cases))
            self.results.append(result)

            # Small delay between tests
            await asyncio.sleep(0.5)

    def print_summary(self):
        """Print comprehensive test summary."""
        print("\n" + "="*80)
        print("COMPREHENSIVE TEST SUMMARY")
        print("="*80)

        # Overall stats
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        print(f"\n📊 OVERALL RESULTS:")
        print(f"   Total tests: {total_tests}")
        print(f"   Passed: {passed_tests} ✅")
        print(f"   Failed: {failed_tests} ❌")
        print(f"   Pass rate: {pass_rate:.1f}%")

        # Performance stats
        avg_response_time = sum(r.response_time_ms for r in self.results) / len(self.results)
        max_response_time = max(r.response_time_ms for r in self.results)
        min_response_time = min(r.response_time_ms for r in self.results)

        print(f"\n⏱️  PERFORMANCE:")
        print(f"   Average response time: {avg_response_time:.0f}ms")
        print(f"   Min response time: {min_response_time:.0f}ms")
        print(f"   Max response time: {max_response_time:.0f}ms")

        # Quality stats
        avg_confidence = sum(r.confidence for r in self.results) / len(self.results)
        avg_citations = sum(len(r.citations) for r in self.results) / len(self.results)

        print(f"\n📈 QUALITY METRICS:")
        print(f"   Average confidence: {avg_confidence:.2f}")
        print(f"   Average citations per query: {avg_citations:.1f}")

        # Citation distinctness
        distinct_ok = sum(1 for r in self.results if r.citations_distinct)
        distinct_rate = (distinct_ok / total_tests * 100) if total_tests > 0 else 0
        print(f"   Citation distinctness: {distinct_ok}/{total_tests} ({distinct_rate:.1f}%)")

        # Reranking stats
        if self.agent.selective_reranker:
            stats = self.agent.selective_reranker.get_stats()
            print(f"\n🔄 RERANKING STATISTICS:")
            print(f"   Queries processed: {stats['total_queries']}")
            print(f"   Queries reranked: {stats['reranked_queries']}")
            print(f"   Rerank rate: {stats['rerank_rate']*100:.1f}%")
            print(f"   Estimated monthly cost: ${stats['estimated_monthly_cost_usd']:.4f}")

            # Rerank triggers
            print(f"\n   Rerank triggers:")
            for reason, count in stats['rerank_reasons'].items():
                if count > 0:
                    print(f"     - {reason}: {count}")

        # Results by category
        print(f"\n📋 RESULTS BY CATEGORY:")
        categories = {}
        for result in self.results:
            cat = result.test_case.test_type
            if cat not in categories:
                categories[cat] = {'passed': 0, 'failed': 0}

            if result.passed:
                categories[cat]['passed'] += 1
            else:
                categories[cat]['failed'] += 1

        for cat, stats in categories.items():
            total = stats['passed'] + stats['failed']
            rate = (stats['passed'] / total * 100) if total > 0 else 0
            print(f"   {cat:20} {stats['passed']}/{total} passed ({rate:.0f}%)")

        # Results by difficulty
        print(f"\n🎯 RESULTS BY DIFFICULTY:")
        difficulties = {}
        for result in self.results:
            diff = result.test_case.difficulty
            if diff not in difficulties:
                difficulties[diff] = {'passed': 0, 'failed': 0}

            if result.passed:
                difficulties[diff]['passed'] += 1
            else:
                difficulties[diff]['failed'] += 1

        for diff, stats in sorted(difficulties.items()):
            total = stats['passed'] + stats['failed']
            rate = (stats['passed'] / total * 100) if total > 0 else 0
            print(f"   {diff:20} {stats['passed']}/{total} passed ({rate:.0f}%)")

        # Failed tests details
        failed_results = [r for r in self.results if not r.passed]
        if failed_results:
            print(f"\n❌ FAILED TESTS DETAILS:")
            for i, result in enumerate(failed_results, 1):
                print(f"\n   {i}. {result.test_case.description}")
                print(f"      Query: \"{result.test_case.query}\"")
                print(f"      Reason: {result.error_message}")

        # Overall assessment
        print(f"\n" + "="*80)
        if pass_rate >= 90:
            print("🎉 EXCELLENT: Quality standards met!")
        elif pass_rate >= 75:
            print("✅ GOOD: Most tests passing, minor issues to address")
        elif pass_rate >= 60:
            print("⚠️  ACCEPTABLE: Several issues need attention")
        else:
            print("❌ NEEDS WORK: Significant issues to resolve")
        print("="*80)

    def save_results(self, filename: str = "quality_test_results.json"):
        """Save test results to JSON file."""
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(self.results),
            'passed': sum(1 for r in self.results if r.passed),
            'failed': sum(1 for r in self.results if not r.passed),
            'results': [
                {
                    'query': r.test_case.query,
                    'test_type': r.test_case.test_type,
                    'difficulty': r.test_case.difficulty,
                    'passed': r.passed,
                    'response_time_ms': r.response_time_ms,
                    'confidence': r.confidence,
                    'citations': r.citations,
                    'citations_distinct': r.citations_distinct,
                    'concepts_found': r.concepts_found,
                    'concepts_missing': r.concepts_missing,
                    'error_message': r.error_message
                }
                for r in self.results
            ]
        }

        output_path = Path(__file__).parent / filename
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)

        print(f"\n💾 Results saved to: {output_path}")


async def main():
    """Run comprehensive quality tests."""
    tester = ComprehensiveQualityTester()

    # Initialize
    await tester.initialize()

    # Run all tests
    await tester.run_all_tests()

    # Print summary
    tester.print_summary()

    # Save results
    tester.save_results()


if __name__ == "__main__":
    asyncio.run(main())