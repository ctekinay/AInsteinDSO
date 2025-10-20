#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

# Force reload BEFORE importing
import importlib
if 'src.agent.ea_assistant' in sys.modules:
    del sys.modules['src.agent.ea_assistant']
    print("Cleared old ea_assistant module")

# Now import fresh
from src.agent.ea_assistant import ProductionEAAgent
import asyncio

async def test():
    print("Testing agent as web app would use it...")
    agent = ProductionEAAgent()
    await asyncio.sleep(5)
    
    response = await agent.process_query("What is an asset in Alliander terms?")
    print(f"\nResponse: {response.response[:200]}...")
    
    if "Entity of value" in response.response:
        print("✅ Agent is using the fix")
    elif "BusinessObject" in response.response:
        print("❌ Agent is NOT using the fix - still using LLM")

asyncio.run(test())