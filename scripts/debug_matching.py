#!/usr/bin/env python3
from src.archimate.parser import ArchiMateParser

def debug_matching():
    """See why queries don't match elements."""
    
    parser = ArchiMateParser()
    parser.load_model("data/models/IEC 61968.xml")
    parser.load_model("data/models/archi-4-archi.xml")
    
    print(f"Total elements loaded: {len(parser.elements)}")
    
    # Test the search
    test_terms = ["capability", "power", "grid", "congestion", "reactive"]
    
    for term in test_terms:
        print(f"\nSearching for '{term}':")
        results = parser.get_citation_candidates([term])
        print(f"  Found: {len(results)} matches")
        
        if len(results) == 0:
            # Check if elements contain this term
            matching = [e for e in parser.elements.values() if term in e.name.lower()]
            print(f"  Direct name matches: {len(matching)}")
            
            if len(matching) > 0:
                print(f"  First match: {matching[0].name} ({matching[0].id})")
    
    # Show what capabilities exist
    all_caps = parser.get_elements_by_type("Capability")
    print(f"\nTotal Capabilities: {len(all_caps)}")
    if all_caps:
        print("First 5 capabilities:")
        for cap in all_caps[:5]:
            print(f"  - {cap.name} ({cap.id})")

if __name__ == "__main__":
    debug_matching()