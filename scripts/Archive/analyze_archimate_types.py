#!/usr/bin/env python3
from src.archimate.parser import ArchiMateParser
from collections import Counter

def analyze_elements():
    parser = ArchiMateParser()
    parser.load_model("data/models/IEC 61968.xml")
    parser.load_model("data/models/archi-4-archi.xml")
    
    # Count element types
    type_counts = Counter()
    for element in parser.elements.values():
        type_counts[element.type] += 1
    
    print("=== ELEMENT TYPES IN YOUR MODELS ===\n")
    for elem_type, count in type_counts.most_common():
        print(f"{elem_type}: {count}")
    
    # Show some examples
    print("\n=== SAMPLE ELEMENTS ===")
    for i, element in enumerate(list(parser.elements.values())[:10]):
        print(f"{i+1}. {element.name} (Type: {element.type}, Layer: {element.layer})")

if __name__ == "__main__":
    analyze_elements()