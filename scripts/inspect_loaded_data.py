#!/usr/bin/env python3
from src.archimate.parser import ArchiMateParser
from src.knowledge.kg_loader import KnowledgeGraphLoader
from pathlib import Path
import time

def inspect_loaded_data():
    """Inspect what data is actually loaded in memory."""
    
    print("=== INSPECTING LOADED DATA ===\n")
    
    # Check ArchiMate
    print("1. ArchiMate Elements:")
    parser = ArchiMateParser()
    
    for model in ["data/models/IEC 61968.xml", "data/models/archi-4-archi.xml"]:
        print(f"\n   Loading {model}...")
        if parser.load_model(model):
            # Get actual elements
            capabilities = parser.get_elements_by_type("Capability")
            print(f"   - Capabilities: {len(capabilities)}")
            
            if capabilities and len(capabilities) > 0:
                print(f"   - First capability:")
                cap = capabilities[0]
                print(f"     ID: {cap.id}")
                print(f"     Name: {cap.name}")
                print(f"     Citation: {cap.get_citation_id()}")
            
            # Check if we can find elements by query
            results = parser.get_citation_candidates(["grid", "congestion"])
            print(f"   - Found {len(results)} elements matching 'grid/congestion'")
    
    # Check Knowledge Graph
    print("\n2. Knowledge Graph:")
    kg = KnowledgeGraphLoader(Path("data/energy_knowledge_graph.ttl"))
    kg.load()
    
    for i in range(20):
        if kg.is_full_graph_loaded():
            break
        time.sleep(0.5)
    
    stats = kg.get_graph_stats()
    print(f"   - Loaded: {stats['loaded']}")
    print(f"   - Triples: {stats['triples']}")
    print(f"   - IEC terms: {stats.get('iec_terms', 0)}")
    
    # Try a direct SPARQL query
    if kg.graph:
        try:
            query = """
            SELECT (COUNT(*) as ?count) WHERE { 
                ?s ?p ?o 
                FILTER(CONTAINS(STR(?s), "Asset"))
            }
            """
            results = list(kg.graph.query(query))
            print(f"   - Concepts containing 'Asset': {results[0][0]}")
        except Exception as e:
            print(f"   - SPARQL test failed: {e}")
    
    print("\n=== END INSPECTION ===")

if __name__ == "__main__":
    inspect_loaded_data()