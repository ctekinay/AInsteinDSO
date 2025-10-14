#!/usr/bin/env python3
"""
Alliander EA Assistant CLI

Usage:
    python cli.py "What capability for grid congestion?"
    python cli.py --interactive
"""

import asyncio
import sys
import json
import time
from pathlib import Path

# Force load environment
from dotenv import load_dotenv
load_dotenv(override=True)

from src.agent.ea_assistant import ProductionEAAgent

def print_progress(message, done=False):
    """Print progress with timing."""
    if done:
        print(f"✅ {message}")
    else:
        print(f"⏳ {message}...", flush=True)

async def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    
    # Initialize agent with progress tracking
    print("🚀 Initializing Alliander EA Assistant\n")
    
    start_time = time.time()
    print_progress("Loading Knowledge Graph (39K+ triples, may take ~90s)")
    
    agent = ProductionEAAgent()
    
    kg_time = time.time() - start_time
    print_progress(f"Knowledge Graph loaded in {kg_time:.1f}s", done=True)
    
    # Initialize LLM
    print_progress("Initializing LLM provider")
    await agent._initialize_llm()
    
    llm_time = time.time() - start_time - kg_time
    if agent.llm_provider:
        print_progress(f"LLM provider ready in {llm_time:.1f}s", done=True)
    else:
        print_progress("LLM unavailable, using template fallback", done=True)
    
    total_time = time.time() - start_time
    print(f"\n✅ System ready in {total_time:.1f}s\n")
    
    if sys.argv[1] == "--interactive":
        # Interactive mode
        print("💬 Interactive mode - type 'exit' to quit\n")
        session_id = "interactive"
        
        while True:
            try:
                query = input("❓ Query: ").strip()
                if query.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Goodbye!\n")
                    break
                
                if not query:
                    continue
                
                # Process with timing
                query_start = time.time()
                response = await agent.process_query(query, session_id)
                query_time = (time.time() - query_start) * 1000
                
                print(f"\n💡 {response.response}\n")
                
                if response.citations:
                    print(f"📚 Citations: {', '.join(response.citations)}")
                
                print(f"🎯 Confidence: {response.confidence:.0%}")
                print(f"⚡ Processing: {query_time:.0f}ms")
                
                if response.requires_human_review:
                    print("⚠️  Human review recommended")
                
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!\n")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
    else:
        # Single query mode
        query = sys.argv[1]
        
        print(f"Processing query: {query}\n")
        
        query_start = time.time()
        response = await agent.process_query(query, "cli-session")
        query_time = (time.time() - query_start) * 1000
        
        # Output format for CLI interface
        print("="*70)
        print(f"QUERY: {query}")
        print("="*70)
        print(f"\nRESPONSE:\n{response.response}")
        print(f"\n{'='*70}")
        print(f"CITATIONS: {', '.join(response.citations) if response.citations else 'None'}")
        print(f"CONFIDENCE: {response.confidence:.0%}")
        print(f"ROUTE: {response.route}")
        print(f"PROCESSING TIME: {query_time:.0f}ms")
        if response.togaf_phase:
            print(f"TOGAF PHASE: {response.togaf_phase}")
        if response.requires_human_review:
            print("⚠️  HUMAN REVIEW RECOMMENDED")
        print("="*70)
        print()

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))