#!/usr/bin/env python3
import xml.etree.ElementTree as ET
from pathlib import Path

def analyze_archimate_structure():
    """Analyze the actual structure of ArchiMate XML files to fix parser."""
    
    models = {
        "archi-4-archi.xml": "Alliander Architecture", 
        "IEC 61968.xml": "IEC Standards Model"
    }
    
    for filename, description in models.items():
        filepath = Path("data/models") / filename
        print(f"\n{'='*60}")
        print(f"Analyzing: {filename}")
        print(f"Description: {description}")
        print(f"File exists: {filepath.exists()}")
        print(f"File size: {filepath.stat().st_size if filepath.exists() else 0} bytes")
        print('='*60)
        
        if not filepath.exists():
            print("FILE NOT FOUND!")
            continue
            
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
            
            # Check root element
            print(f"\nRoot element: {root.tag}")
            print(f"Namespaces in root: {root.attrib}")
            
            # Look for different possible element structures
            patterns_to_check = [
                (".//*[@xsi:type]", "Elements with xsi:type attribute"),
                (".//*[@{http://www.w3.org/2001/XMLSchema-instance}type]", "Elements with full namespace xsi:type"),
                (".//element", "Elements with 'element' tag"),
                (".//node", "Elements with 'node' tag"),
                (".//*[@name]", "Elements with 'name' attribute"),
                (".//*[@id]", "Elements with 'id' attribute")
            ]
            
            for pattern, description in patterns_to_check:
                try:
                    elements = root.findall(pattern)
                    if elements:
                        print(f"\n{description}: {len(elements)} found")
                        # Show first 3 examples
                        for i, elem in enumerate(elements[:3]):
                            elem_type = elem.get('{http://www.w3.org/2001/XMLSchema-instance}type') or elem.get('xsi:type') or elem.tag
                            elem_name = elem.get('name') or elem.get('label') or 'unnamed'
                            elem_id = elem.get('id') or elem.get('identifier') or 'no-id'
                            print(f"  Example {i+1}: type={elem_type}, name={elem_name}, id={elem_id}")
                except:
                    pass
            
            # Show all unique tag names in the file
            all_tags = set()
            for elem in root.iter():
                tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                all_tags.add(tag)
            
            print(f"\nUnique XML tags in file ({len(all_tags)} total):")
            for tag in sorted(list(all_tags)[:20]):  # Show first 20
                count = len(root.findall(f".//*[local-name()='{tag}']"))
                if count > 0:
                    print(f"  - {tag}: {count} occurrences")
                    
        except Exception as e:
            print(f"ERROR parsing XML: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    analyze_archimate_structure()