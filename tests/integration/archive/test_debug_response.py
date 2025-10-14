# debug_response.py
import asyncio
from src.agent.ea_assistant import ProductionEAAgent

async def debug_response():
    agent = ProductionEAAgent(llm_provider=None)
    response = await agent.process_query("test", "debug")
    
    print(f"Response type: {type(response)}")
    print(f"Response repr: {repr(response)}")
    
    # Check all attributes
    attrs = dir(response)
    print(f"\nAvailable attributes: {[a for a in attrs if not a.startswith('_')]}")
    
    # Try to access each
    for attr in ['answer', 'response', 'text', 'content', 'message', 'result']:
        if hasattr(response, attr):
            value = getattr(response, attr)
            print(f"{attr}: {str(value)[:100]}")

asyncio.run(debug_response())