# diagnose.py
import sys
from pathlib import Path

print("🔍 PYTHON PATH DIAGNOSTICS")
print("=" * 60)

# 1. Check Python path
print("\n1️⃣ Python Path:")
for i, path in enumerate(sys.path, 1):
    print(f"   {i}. {path}")

# 2. Check project structure
print("\n2️⃣ Project Structure:")
project_root = Path(__file__).parent
src_dir = project_root / "src"

print(f"   Project Root: {project_root}")
print(f"   src/ exists: {src_dir.exists()}")

if src_dir.exists():
    print(f"   src/__init__.py exists: {(src_dir / '__init__.py').exists()}")
    
    agents_dir = src_dir / "agents"
    print(f"   src/agents/ exists: {agents_dir.exists()}")
    
    if agents_dir.exists():
        print(f"   src/agents/__init__.py exists: {(agents_dir / '__init__.py').exists()}")
        print(f"   src/agents/ea_assistant.py exists: {(agents_dir / 'ea_assistant.py').exists()}")

# 3. Try import with path fix
print("\n3️⃣ Testing Import:")
sys.path.insert(0, str(project_root))

try:
    from src.agents.ea_assistant import ProductionEAAgent
    print("   ✅ Import successful!")
except Exception as e:
    print(f"   ❌ Import failed: {e}")

print("\n" + "=" * 60)