#!/usr/bin/env python3
import inspect
from src.knowledge.kg_loader import KnowledgeGraphLoader

def check_sparql_implementation():
    """Check if SPARQL queries have comments that break parsing."""

    print("=== CHECKING SPARQL IMPLEMENTATION ===\n")

    # Get the source code of query_definitions
    source = inspect.getsource(KnowledgeGraphLoader.query_definitions)

    # Look for problematic patterns
    problems = []

    lines = source.split('\n')
    in_sparql = False
    sparql_start = -1

    for i, line in enumerate(lines):
        # Track when we're inside a SPARQL query string
        if 'sparql_query = f"""' in line or "sparql_query = '''" in line:
            in_sparql = True
            sparql_start = i
        elif in_sparql and '"""' in line:
            in_sparql = False

        # Check for comments inside SPARQL
        if in_sparql and '#' in line and not line.strip().startswith('PREFIX'):
            problems.append(f"Line {i}: Possible comment in SPARQL: {line.strip()[:50]}")

    if problems:
        print("❌ FOUND PROBLEMS IN SPARQL:")
        for p in problems:
            print(f"  {p}")
    else:
        print("✅ No SPARQL comment problems detected")

    # Show a sample of the method
    print("\n=== First 30 lines of query_definitions method ===")
    for i, line in enumerate(lines[:30]):
        print(f"{i:3}: {line}")

if __name__ == "__main__":
    check_sparql_implementation()