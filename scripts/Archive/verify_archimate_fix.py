#!/usr/bin/env python3
import inspect
from src.archimate.parser import ArchiMateParser

def verify_archimate_fix():
    """Check if the ArchiMate parser fix was actually applied."""
    
    print("=== VERIFYING ARCHIMATE FIX ===\n")
    
    # Get the source code of _parse_element method
    source = inspect.getsource(ArchiMateParser._parse_element)
    
    # Check for the old problematic code
    if 'return None' in source and 'without name' in source:
        print("❌ OLD CODE DETECTED - Elements without names are being rejected!")
        print("The fix was NOT properly applied.\n")
    
    # Check for the fix
    if 'element_type}_{element_id[-8:]' in source:
        print("✅ Fix detected - unnamed elements get generated names")
    else:
        print("⚠️ Fix not found - unnamed elements may still be rejected")
    
    # Show the actual code handling unnamed elements
    print("\n=== Code handling unnamed elements ===")
    lines = source.split('\n')
    for i, line in enumerate(lines):
        if 'not name' in line or 'unnamed' in line.lower():
            # Show context around this line
            start = max(0, i-2)
            end = min(len(lines), i+3)
            for j in range(start, end):
                marker = ">>>" if j == i else "   "
                print(f"{marker} {lines[j]}")
    
    # Now test if it actually works
    print("\n=== ACTUAL TEST ===")
    parser = ArchiMateParser()
    
    # Try loading the IEC model which has 2771 elements
    success = parser.load_model("data/models/IEC 61968.xml")
    
    if success:
        summary = parser.get_model_summary()
        total = summary.get('total_elements', 0)
        
        if total > 100:  # Should be 669+ based on claims
            print(f"✅ WORKING: Loaded {total} elements")
            return True
        elif total > 0:
            print(f"⚠️ PARTIAL: Only loaded {total} elements (expected 600+)")
            return False
        else:
            print(f"❌ BROKEN: Loaded 0 elements despite fix claims")
            return False
    else:
        print("❌ FAILED: Could not load model at all")
        return False

if __name__ == "__main__":
    result = verify_archimate_fix()
    exit(0 if result else 1)