#!/usr/bin/env python3
import asyncio
from src.agent.ea_assistant import ProductionEAAgent
from src.archimate.parser import ArchiMateParser

async def test_citations_are_real():
    """Test that citations reference actual elements, not fake ones."""
    
    print("=== TESTING IF CITATIONS ARE REAL ===\n")
    
    # Load ArchiMate parser to check citations against
    parser = ArchiMateParser()
    parser.load_model("data/models/IEC 61968.xml")
    
    # Initialize agent
    agent = ProductionEAAgent()
    await asyncio.sleep(5)  # Wait for KG to load
    
    test_queries = [
        "What capabilities exist for power management?",
        "Show me capabilities",
        "What is reactive power?",
        "List some capabilities for grid congestion"
    ]
    
    all_citations_valid = True
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        
        try:
            response = await agent.process_query(query)
            print(f"Response preview: {response.response[:150]}...")
            print(f"Citations: {response.citations}")
            
            # Check each citation
            for citation in response.citations:
                if citation.startswith("archi:"):
                    # Extract the ID and check if it exists
                    element_id = citation.replace("archi:", "")
                    element = parser.get_element_by_id(element_id)
                    
                    if element:
                        print(f"  ✅ VALID: {citation} -> {element.name}")
                    else:
                        print(f"  ❌ FAKE: {citation} does NOT exist!")
                        all_citations_valid = False
                        
                elif citation == "iec:GridCongestion" or citation == "iec:61968":
                    print(f"  ❌ HARDCODED: {citation} is a fake hardcoded citation!")
                    all_citations_valid = False
                    
                elif citation.startswith("external:"):
                    print(f"  ⚠️  External source: {citation}")
                    
                else:
                    print(f"  ℹ️  Citation: {citation}")
                    
        except Exception as e:
            print(f"  Error: {e}")
            
    if all_citations_valid:
        print("\n✅ SUCCESS: All citations are real!")
    else:
        print("\n❌ FAILURE: Fake/hardcoded citations detected!")
    
    return all_citations_valid

if __name__ == "__main__":
    result = asyncio.run(test_citations_are_real())
    exit(0 if result else 1)