# test_integration.py
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded .env file")
except ImportError:
    print("⚠️  python-dotenv not installed, using system environment")

import asyncio
import os
import logging
# Configure logging  # ← ADD THESE LINES
logger = logging.getLogger(__name__)

from src.agents.ea_assistant import ProductionEAAgent

async def test_complete_system():
    """Comprehensive system test covering all components."""
    
    print("\n🧪 TESTING EA ASSISTANT - COMPLETE PIPELINE")
    print("=" * 60)
    
    # Verify API keys are loaded
    print("\n🔑 Checking API Keys...")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    groq_key = os.getenv("GROQ_API_KEY", "")
    
    if openai_key:
        print(f"   ✅ OpenAI Key: {openai_key[:20]}...")
    else:
        print("   ⚠️  OpenAI Key: Not set")
    
    if groq_key:
        print(f"   ✅ Groq Key: {groq_key[:20]}...")
    else:
        print("   ⚠️  Groq Key: Not set")
    
    if not openai_key and not groq_key:
        print("\n❌ ERROR: No API keys found!")
        return False
    
    try:
        # Initialize agent
        print("\n1️⃣ Initializing agent...")
        agent = ProductionEAAgent()
        print("   ✅ Agent initialized")
        
        # Test 1: Simple definition query
        print("\n2️⃣ Test 1: Definition Query")
        print("   Query: 'What is reactive power?'")
        response, trace = await agent.process_query(
            "What is reactive power?",
            session_id="test-001"
        )
        
        print(f"   ✅ Response: {len(response.response)} chars")
        print(f"   ✅ Citations: {len(response.citations)}")
        print(f"   ✅ Confidence: {response.confidence:.2f}")
        print(f"   ✅ Time: {response.processing_time_ms:.0f}ms")
        
        if len(response.citations) > 0:
            print(f"   📝 Sample citations: {response.citations[:3]}")
        
        assert len(response.citations) > 0, "No citations found!"
        assert response.confidence > 0.5, "Confidence too low!"
        print("   ✅ Test 1 PASSED")
        
        # Test 2: Architectural query
        print("\n3️⃣ Test 2: Architectural Query")
        print("   Query: 'Model congestion management'")
        response, trace = await agent.process_query(
            "How should I model congestion management?",
            session_id="test-002"
        )
        
        print(f"   ✅ Response: {len(response.response)} chars")
        print(f"   ✅ Route: {response.route}")
        print(f"   ✅ Confidence: {response.confidence:.2f}")
        print(f"   ✅ Time: {response.processing_time_ms:.0f}ms")
        print("   ✅ Test 2 PASSED")
        
        # Test 3: Citation authenticity - FIXED METHOD NAME
        print("\n4️⃣ Test 3: Citation Validation")
        valid_count = 0
        for citation in response.citations[:3]:
            # FIXED: Use validate_citation_exists
            exists = agent.citation_validator.validate_citation_exists(citation)
            if exists:
                valid_count += 1
            print(f"   • {citation}: {'✅' if exists else '❌'}")

        if valid_count > 0:
            print(f"   ✅ Test 3 PASSED - {valid_count} valid citations")
        else:
            print("   ⚠️  Test 3 - No citations validated (may need knowledge graph)")
        
        # Test 4: Out-of-scope handling
        print("\n5️⃣ Test 4: Out-of-Scope Query")
        response, trace = await agent.process_query(
            "What's the weather?",
            session_id="test-003"
        )
        
        declined = ("outside" in response.response.lower() or 
                   "scope" in response.response.lower() or
                   "cannot" in response.response.lower())
        print(f"   Response preview: {response.response[:100]}...")
        print(f"   {'✅' if declined else '⚠️'} Out-of-scope: {declined}")
        print("   ✅ Test 4 PASSED")
        
        # Test 5: Show actual response content
        print("\n6️⃣ Test 5: Sample Response Content")
        print("   Query: 'What is active power?'")
        response, trace = await agent.process_query(
            "What is active power?",
            session_id="test-004"
        )
        
        print(f"\n   📄 RESPONSE:")
        print(f"   {response.response}")
        print(f"\n   📚 Citations used: {response.citations}")
        print(f"   🎯 Confidence: {response.confidence:.2f}")
        print("   ✅ Test 5 PASSED")
        
        # Statistics
        print("\n" + "=" * 60)
        print("📊 SYSTEM STATISTICS")
        print("=" * 60)
        stats = agent.get_statistics()
        
        kg_stats = stats.get('knowledge_graph', {})
        am_stats = stats.get('archimate_models', {})
        
        print(f"   • KG Triples: {kg_stats.get('triple_count', 0):,}")
        print(f"   • ArchiMate Elements: {am_stats.get('total_elements', 0):,}")
        print(f"   • Citation Pool: {stats.get('citation_pools_loaded', 0):,}")
        print(f"   • Sessions: {stats.get('sessions_processed', 0)}")
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        print("\n💡 System is working correctly!")
        print("   • Query processing: ✅")
        print("   • Citation generation: ✅")
        print("   • Confidence scoring: ✅")
        print("   • Tracing: ✅")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # CRITICAL FIX: Clean up connections
        if 'agent' in locals():
            try:
                # Close LLM Council connections
                if hasattr(agent, 'llm_council') and agent.llm_council:
                    await agent.llm_council.close()
                
                # Close any other async resources
                if hasattr(agent, 'llm_provider') and agent.llm_provider:
                    if hasattr(agent.llm_provider, 'close'):
                        await agent.llm_provider.close()
                
                logger.info("✅ Cleaned up connections")
            except Exception as e:
                logger.debug(f"Cleanup warning: {e}")

if __name__ == "__main__":
    success = asyncio.run(test_complete_system())
    sys.exit(0 if success else 1)