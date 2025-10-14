import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from src.safety.citation_validator import CitationValidator

# See what methods are available
validator = CitationValidator(None, None, None)
methods = [m for m in dir(validator) if not m.startswith('_')]
print("Available methods:", methods)