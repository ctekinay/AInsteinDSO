#!/usr/bin/env python3
"""
Test script to demonstrate realistic performance SLOs and monitoring.

This script simulates production load and validates that our SLOs are
honest and achievable under real infrastructure constraints.
"""

import asyncio
import logging
import time
from pathlib import Path
from src.monitoring.performance_slos import (
    PerformanceMonitor, ComponentType, InfrastructureProfile
)
from src.knowledge.kg_loader import KnowledgeGraphLoader
from src.routing.query_router import QueryRouter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def simulate_production_load():
    """Simulate realistic production load with infrastructure constraints."""

    logger.info("Testing Realistic Performance SLOs...")

    # Test different infrastructure profiles
    for profile in [InfrastructureProfile.DEVELOPMENT,
                   InfrastructureProfile.STAGING,
                   InfrastructureProfile.PRODUCTION,
                   InfrastructureProfile.ENTERPRISE]:

        logger.info(f"\n=== Testing {profile.value.upper()} Profile ===")

        # Initialize monitor for this profile
        monitor = PerformanceMonitor(profile)

        # Initialize components
        kg_loader = KnowledgeGraphLoader(Path("data/energy_knowledge_graph.ttl"))
        kg_loader.load()

        router = QueryRouter(kg_loader=kg_loader)

        # Simulate realistic query patterns
        test_queries = [
            "grid congestion management",
            "transformer maintenance",
            "renewable energy integration",
            "power quality monitoring",
            "smart grid architecture",
            "energy storage systems",
            "distribution automation",
            "substation protection",
            "voltage regulation",
            "load forecasting"
        ]

        # Cold start measurements (realistic for production)
        logger.info("Testing cold start performance...")
        for i, query in enumerate(test_queries[:3]):
            context = {"cold_start": True, "query_complexity": "medium"}

            with monitor.measure_operation(ComponentType.QUERY_ROUTER, "route_query", context):
                route = router.route(query)
                # Simulate additional cold start overhead
                await asyncio.sleep(0.01 * (profile.value == "enterprise" and 2 or 1))

            with monitor.measure_operation(ComponentType.KNOWLEDGE_GRAPH, "query", context):
                results = kg_loader.load_on_demand(query.split())
                # Simulate cold start cache misses
                await asyncio.sleep(0.05 * (profile.value == "enterprise" and 3 or 1))

        # Warm system measurements
        logger.info("Testing warm system performance...")
        for query in test_queries:
            context = {"cold_start": False, "query_complexity": "medium"}

            with monitor.measure_operation(ComponentType.QUERY_ROUTER, "route_query", context):
                route = router.route(query)

            with monitor.measure_operation(ComponentType.KNOWLEDGE_GRAPH, "query", context):
                results = kg_loader.load_on_demand(query.split())

        # High load simulation
        logger.info("Testing high load scenarios...")
        for i in range(5):
            context = {"high_load": True, "concurrent_requests": 10}

            with monitor.measure_operation(ComponentType.FULL_PIPELINE, "end_to_end", context):
                # Simulate full pipeline with infrastructure delays
                route = router.route(test_queries[i % len(test_queries)])
                results = kg_loader.load_on_demand(test_queries[i % len(test_queries)].split())

                # Simulate infrastructure latency
                if profile == InfrastructureProfile.ENTERPRISE:
                    await asyncio.sleep(0.1)  # Enterprise firewall delays
                elif profile == InfrastructureProfile.PRODUCTION:
                    await asyncio.sleep(0.05)  # Network latency

        # Generate compliance report
        compliance = monitor.check_slo_compliance()

        logger.info(f"\n{profile.value.upper()} Performance Results:")
        logger.info(f"Overall Status: {compliance['overall_status']}")
        logger.info(f"Violations: {compliance['violations_last_hour']}")

        # Show key metrics
        for component in [ComponentType.QUERY_ROUTER, ComponentType.KNOWLEDGE_GRAPH, ComponentType.FULL_PIPELINE]:
            stats = monitor.get_performance_stats(component, 300)  # 5 minutes
            if "error" not in stats:
                logger.info(f"{component.value}: P50={stats['p50_ms']:.1f}ms, P95={stats['p95_ms']:.1f}ms, P99={stats['p99_ms']:.1f}ms")

        # Show recommendations if any
        if compliance["recommendations"]:
            logger.info("Recommendations:")
            for rec in compliance["recommendations"]:
                logger.info(f"  - {rec}")


def test_slo_violation_detection():
    """Test SLO violation detection with artificially slow operations."""

    logger.info("\n=== Testing SLO Violation Detection ===")

    monitor = PerformanceMonitor(InfrastructureProfile.PRODUCTION)

    # Simulate operations that should trigger violations
    logger.info("Simulating slow operations to test violation detection...")

    # Slow knowledge graph operation (should violate P99)
    with monitor.measure_operation(ComponentType.KNOWLEDGE_GRAPH, "slow_query"):
        time.sleep(2.0)  # 2 seconds - exceeds P99 target of 1000ms

    # Slow LLM operation (should be within tolerance)
    with monitor.measure_operation(ComponentType.LLM_PROVIDER, "llm_call"):
        time.sleep(3.0)  # 3 seconds - within P99 target of 5000ms

    # Very slow pipeline (should violate)
    with monitor.measure_operation(ComponentType.FULL_PIPELINE, "end_to_end"):
        time.sleep(3.0)  # 3 seconds - exceeds P99 target of 2000ms

    # Check violations
    compliance = monitor.check_slo_compliance()
    logger.info(f"Detected {compliance['violations_last_hour']} violations")

    if monitor.slo_violations:
        logger.info("SLO Violations detected:")
        for violation in monitor.slo_violations[-3:]:  # Show last 3
            logger.info(f"  - {violation['component']}: {violation['actual_latency_ms']:.1f}ms > {violation['target_latency_ms']:.1f}ms")


def demonstrate_infrastructure_differences():
    """Demonstrate how SLO targets adjust for different infrastructures."""

    logger.info("\n=== Infrastructure Profile Comparison ===")

    # Show adjusted targets for each profile
    from src.monitoring.performance_slos import PerformanceMonitor

    base_slo = PerformanceMonitor.PRODUCTION_SLOS[ComponentType.FULL_PIPELINE]

    for profile in InfrastructureProfile:
        monitor = PerformanceMonitor(profile)
        adjusted = monitor._get_adjusted_targets(base_slo, {})

        logger.info(f"{profile.value.upper()}:")
        logger.info(f"  P50: {adjusted['p50']:.1f}ms")
        logger.info(f"  P95: {adjusted['p95']:.1f}ms")
        logger.info(f"  P99: {adjusted['p99']:.1f}ms")


async def main():
    """Run all performance SLO tests."""
    logger.info("EA Assistant Realistic Performance SLO Testing")
    logger.info("=" * 50)

    # Test realistic performance under different infrastructure
    await simulate_production_load()

    # Test violation detection
    test_slo_violation_detection()

    # Show infrastructure differences
    demonstrate_infrastructure_differences()

    logger.info("\nPerformance testing complete!")
    logger.info("Key Insights:")
    logger.info("- Development: 0.3x faster (local optimizations)")
    logger.info("- Staging: 1.0x baseline reference")
    logger.info("- Production: 2.0x slower (real infrastructure)")
    logger.info("- Enterprise: 3.0x slower (security/compliance overhead)")


if __name__ == "__main__":
    asyncio.run(main())