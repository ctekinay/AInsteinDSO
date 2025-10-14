#!/usr/bin/env python3
"""Test reactive power query routing to verify energy term detection."""

import asyncio
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agent.ea_assistant import ProductionEAAgent

async def test_reactive_power():
    print("🔋 TESTING REACTIVE POWER QUERY ROUTING\n")

    agent = ProductionEAAgent(llm_provider=None)

    # Test queries that should route to structured_model
    test_queries = [
        "reactive power management",
        "active power monitoring",
        "grid congestion analysis",
        "substation equipment selection",
        "IEC 61968 compliance check",
        "power quality assessment"
    ]

    for query in test_queries:
        print(f"Query: '{query}'")
        response = await agent.process_query(query, f"test-{hash(query)}")

        # Check routing
        route_status = "✅" if response.route == "structured_model" else "❌"
        print(f"  {route_status} Route: {response.route}")

        # Check citations
        citation_status = "✅" if response.citations else "❌"
        print(f"  {citation_status} Citations: {response.citations}")

        # Check energy terms in response
        energy_terms = ["power", "grid", "energy", "electrical", "IEC"]
        found_terms = [term for term in energy_terms if term.lower() in response.response.lower()]
        terms_status = "✅" if found_terms else "❌"
        print(f"  {terms_status} Energy terms: {found_terms}")

        print(f"  Confidence: {response.confidence:.2f}")
        print()

if __name__ == "__main__":
    asyncio.run(test_reactive_power())