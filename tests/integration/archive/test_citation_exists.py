# test_citation_exists.py - FIXED VERSION
from pathlib import Path
import time

print("="*60)
print("Testing Citation Validation System")
print("="*60)

# Test 1: KG citations
print("\n1. Testing Knowledge Graph Citations...")
from src.knowledge.kg_loader import KnowledgeGraphLoader

# FIX: Pass Path object, not string
kg = KnowledgeGraphLoader(Path("data/energy_knowledge_graph.ttl"))
kg.load()

# Wait for graph to load
print("   Waiting for graph to load...")
for i in range(20):
    if kg.is_full_graph_loaded():
        print(f"   ✓ Graph loaded after {i*0.5:.1f}s")
        break
    time.sleep(0.5)
else:
    print("   ⚠️  Graph still loading after 10s, proceeding anyway...")

# Test citation_exists method
print("\n   Testing citation_exists() method:")
try:
    result1 = kg.citation_exists("skos:Asset")
    print(f"   • skos:Asset: {result1} (should be True)")
    
    result2 = kg.citation_exists("iec:GridCongestion")
    print(f"   • iec:GridCongestion: {result2} (should be False - FAKE)")
    
    result3 = kg.citation_exists("archi:id-cap-001")
    print(f"   • archi:id-cap-001: {result3} (should be False - FAKE)")
except AttributeError as e:
    print(f"   ⚠️  Method not implemented: {e}")
    print("   Need to add citation_exists() method to KnowledgeGraphLoader")

# Test 2: ArchiMate citations
print("\n2. Testing ArchiMate Citations...")
from src.archimate.parser import ArchiMateParser

parser = ArchiMateParser()
try:
    parser.load_model("data/models/IEC 61968.xml")
    
    valid_cits = parser.get_valid_citations()
    print(f"   ✓ Found {len(valid_cits)} valid ArchiMate citations")
    
    if valid_cits:
        print(f"   First 5 citations:")
        for cit in valid_cits[:5]:
            print(f"      • {cit}")
    else:
        print("   ⚠️  No citations found - check XML structure")
        
except FileNotFoundError as e:
    print(f"   ✗ File not found: {e}")
except AttributeError as e:
    print(f"   ⚠️  Method not implemented: {e}")
    print("   Need to add get_valid_citations() method to ArchiMateParser")

# Test 3: Citation Validator integration
print("\n3. Testing Citation Validator...")
try:
    from src.safety.citation_validator import CitationValidator
    
    validator = CitationValidator(
        kg_loader=kg,
        archimate_parser=parser,
        pdf_indexer=None  # Optional
    )
    
    test_citations = [
        "skos:Asset",
        "iec:GridCongestion", 
        "archi:id-cap-001",
        "external:https://example.com"
    ]
    
    print("   Validating test citations:")
    for citation in test_citations:
        try:
            is_valid = validator.validate_citation_exists(citation)
            status = "✓ VALID" if is_valid else "✗ FAKE"
            print(f"      {status}: {citation}")
        except Exception as e:
            print(f"      ⚠️  Error validating {citation}: {e}")
            
except ImportError as e:
    print(f"   ⚠️  CitationValidator not available: {e}")

print("\n" + "="*60)
print("Testing Complete")
print("="*60)