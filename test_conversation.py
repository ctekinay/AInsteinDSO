"""
Test conversation memory with the full pipeline.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

import asyncio
from src.agents.ea_assistant import ProductionEAAgent


async def test_conversation():
    """Test conversational capabilities."""
    
    print("\n🎭 TESTING CONVERSATIONAL AI WITH FULL PIPELINE")
    print("=" * 70)
    
    agent = ProductionEAAgent()
    session_id = "conversation-test-001"
    
    # Test conversation sequence
    conversation = [
        ("What is reactive power?", "Define reactive power"),
        ("What is active power?", "Define active power"),
        ("What is the difference between active and reactive power?", "Compare both"),
        ("How are they used in power systems?", "Application context"),
    ]
    
    print(f"\n📍 Session: {session_id}")
    print(f"📊 Testing {len(conversation)} conversation turns\n")
    
    for i, (query, description) in enumerate(conversation, 1):
        print(f"\n{'='*70}")
        print(f"TURN {i}: {description}")
        print(f"Query: \"{query}\"")
        print('='*70)
        
        response, trace = await agent.process_query(
            query,
            session_id=session_id,
            use_conversation_context=True
        )
        
        print(f"\n📄 RESPONSE:")
        print("-" * 70)
        # Print first 600 chars
        print(response.response[:600])
        if len(response.response) > 600:
            print("...")
        print("-" * 70)
        
        print(f"\n📊 METRICS:")
        print(f"   • Route: {response.route}")
        print(f"   • Citations: {response.citations[:5]}")
        print(f"   • Confidence: {response.confidence:.2f}")
        print(f"   • Time: {response.processing_time_ms:.0f}ms")
        print(f"   • Review needed: {response.requires_human_review}")
        
        # Show session stats
        stats = agent.session_manager.get_session_stats(session_id)
        print(f"\n💬 SESSION STATS:")
        print(f"   • Total turns: {stats['turns']}")
        print(f"   • Avg confidence: {stats['avg_confidence']}")
        print(f"   • Duration: {stats['duration_minutes']:.1f} minutes")
        print(f"   • Key concepts: {', '.join(stats['key_concepts_discussed'][:5])}")
        
        # Brief pause
        await asyncio.sleep(0.5)
    
    print("\n" + "="*70)
    print("✅ CONVERSATION TEST COMPLETE")
    print("="*70)
    
    # Final session summary
    final_stats = agent.session_manager.get_session_stats(session_id)
    print(f"\n📈 FINAL SESSION SUMMARY:")
    print(f"   • Total turns: {final_stats['turns']}")
    print(f"   • Average confidence: {final_stats['avg_confidence']}")
    print(f"   • Total citations: {final_stats['total_citations']}")
    print(f"   • Session duration: {final_stats['duration_minutes']:.1f} minutes")
    print(f"   • Concepts discussed: {', '.join(final_stats['key_concepts_discussed'])}")


if __name__ == "__main__":
    asyncio.run(test_conversation())