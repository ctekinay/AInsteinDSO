#!/usr/bin/env python3
import sys
import os

# Add project to path
sys.path.insert(0, '.')

print("Testing system components...")

# Test basic imports
try:
    from src.agents.ea_assistant import ProductionEAAgent
    print("✓ EA Assistant imports")
except Exception as e:
    print(f"✗ EA Assistant error: {e}")

# Test LLM Council
try:
    from src.agents.llm_council import LLMCouncil
    print("✓ LLM Council imports")
except Exception as e:
    print(f"✗ LLM Council error: {e}")

print("\nSystem ready for orchestrator!")