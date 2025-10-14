# Fixed test_llm_integration.py
import asyncio
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.agent.ea_assistant import ProductionEAAgent

async def test_real_llm():
    print("Testing EA Assistant with LLM Integration\n")
    
    # Test WITHOUT LLM (template fallback)
    print("1. Testing template fallback (no LLM)...")
    try:
        agent_no_llm = ProductionEAAgent(llm_provider=None)
        response1 = await agent_no_llm.process_query(
            "What capability for grid congestion?",
            "test-no-llm"
        )
        # Access response attributes directly (not .get())
        print(f"Template response: {response1.answer[:200] if hasattr(response1, 'answer') else 'No answer'}...")
        print(f"Citations: {response1.citations if hasattr(response1, 'citations') else []}")
        print(f"Confidence: {response1.confidence if hasattr(response1, 'confidence') else 'N/A'}\n")
    except Exception as e:
        print(f"Template test failed: {e}\n")
    
    # Test WITH Groq (only if configured)
    print("2. Testing with Groq LLM...")
    try:
        import os
        if not os.getenv("GROQ_API_KEY"):
            print("No GROQ_API_KEY in environment. Set it first:")
            print("export GROQ_API_KEY='your-key-here'")
            return
            
        agent_with_llm = ProductionEAAgent(llm_provider="groq")
        response2 = await agent_with_llm.process_query(
            "What capability for grid congestion management?",
            "test-with-llm"
        )
        print(f"LLM response: {response2.answer[:200] if hasattr(response2, 'answer') else 'No answer'}...")
        print(f"Citations: {response2.citations if hasattr(response2, 'citations') else []}")
        print(f"Confidence: {response2.confidence if hasattr(response2, 'confidence') else 'N/A'}")
    except Exception as e:
        print(f"LLM test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_real_llm())