#!/usr/bin/env python3
import os
import re

def find_fake_citations():
    """Find where fake citations are hardcoded."""
    
    fake_patterns = [
        'archi:id-cap-001',
        'iec:GridCongestion', 
        'iec:61968',
        'iec:61970'
    ]
    
    # Search through all Python files
    for root, dirs, files in os.walk('src'):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                
                with open(filepath, 'r') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                for pattern in fake_patterns:
                    if pattern in content:
                        print(f"\n❌ FOUND '{pattern}' in {filepath}:")
                        
                        # Show the lines containing it
                        for i, line in enumerate(lines, 1):
                            if pattern in line:
                                print(f"   Line {i}: {line.strip()[:100]}")

if __name__ == "__main__":
    find_fake_citations()