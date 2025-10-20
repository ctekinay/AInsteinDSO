#!/usr/bin/env python3
from src.archimate.parser import ArchiMateParser

def test_archimate_loading():
    """Test if ArchiMate models load correctly after fix."""

    parser = ArchiMateParser()

    # Load both models
    models = [
        "data/models/archi-4-archi.xml",
        "data/models/IEC 61968.xml"
    ]

    for model_path in models:
        print(f"\n=== Loading {model_path} ===")
        success = parser.load_model(model_path)

        if success:
            summary = parser.get_model_summary()
            print(f"✅ Loaded successfully!")
            print(f"   Total elements: {summary['total_elements']}")
            print(f"   By layer: {summary.get('elements_by_layer', {})}")

            # Test getting some elements
            capabilities = parser.get_elements_by_type("Capability")
            print(f"   Capabilities found: {len(capabilities)}")
            if capabilities:
                print(f"   First capability: {capabilities[0]}")
        else:
            print(f"❌ Failed to load")

    # Test citation search
    print("\n=== Testing Citation Search ===")
    results = parser.get_citation_candidates(["capability", "grid", "congestion"])
    print(f"Found {len(results)} citation candidates")
    for r in results[:5]:
        print(f"  - {r.name}: {r.get_citation_id()}")

if __name__ == "__main__":
    test_archimate_loading()