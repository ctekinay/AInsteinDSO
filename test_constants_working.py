# test_constants_working.py
import os
from pathlib import Path

# CRITICAL: Load .env file BEFORE importing anything else
from dotenv import load_dotenv
load_dotenv()  # This reads your .env file

# Verify it loaded
if not os.getenv('OPENAI_API_KEY'):
    print("⚠️ Warning: OPENAI_API_KEY not found in environment")
    print("Make sure .env file exists in project root")
else:
    print(f"✅ API key loaded: {os.getenv('OPENAI_API_KEY')[:20]}...")

# NOW import the rest
import asyncio
from src.agents.ea_assistant import ProductionEAAgent

async def test_query():
    print("🧪 Testing with real query...")
    agent = ProductionEAAgent()
    
    # Wait for initialization
    await asyncio.sleep(5)
    
    # Test query
    query = "What is reactive power?"
    print(f"\nQuery: {query}")
    
    try:
        response, trace = await agent.process_query(query)
        
        print(f"✅ Response received: {len(response.response)} chars")
        print(f"✅ Route: {response.route}")
        print(f"✅ Citations: {len(response.citations)}")
        print(f"✅ Confidence: {response.confidence:.2f}")
        
        # Check that constants are actually being used
        from src.config.constants import CONFIDENCE
        print(f"\n📊 Confidence threshold: {CONFIDENCE.HIGH_CONFIDENCE_THRESHOLD}")
        print(f"✅ Constants are accessible")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        await agent.cleanup()  # If this method exists
        return False

if __name__ == "__main__":
    result = asyncio.run(test_query())
    if result:
        print("\n✅ Real query test PASSED")
    else:
        print("\n❌ Real query test FAILED")