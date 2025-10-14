#!/usr/bin/env python3
import os
import logging

# ENABLE DEBUG LOGGING
logging.basicConfig(level=logging.DEBUG)

# Force load .env explicitly
from dotenv import load_dotenv
load_dotenv(override=True, verbose=True)

# Verify variables BEFORE importing agent
print("\n" + "="*70)
print("PRE-FLIGHT CHECK")
print("="*70)
print(f"LLM_PROVIDER: {os.getenv('LLM_PROVIDER')}")
print(f"GROQ_API_KEY: {os.getenv('GROQ_API_KEY', 'NOT SET')[:20]}...")
print(f"GROQ_MODEL: {os.getenv('GROQ_MODEL')}")
print("="*70 + "\n")

# NOW import and test
import asyncio
from src.agent.ea_assistant import ProductionEAAgent

async def test():
    print("Initializing agent...")
    agent = ProductionEAAgent()  # Default should be groq
    
    print(f"\nAgent LLM Provider Name: {agent.llm_provider_name}")
    print(f"Agent LLM Provider Object: {agent.llm_provider}")
    
    # Try to initialize LLM explicitly
    await agent._initialize_llm()
    
    print(f"After init - LLM Provider: {agent.llm_provider}")
    
    if agent.llm_provider:
        print(f"✅ LLM Provider Type: {type(agent.llm_provider).__name__}")
        print(f"✅ LLM Model: {agent.llm_provider.model}")
    else:
        print("❌ LLM Provider is None - using template fallback")
    
    # Now try a query
    response = await agent.process_query("What is reactive power?", "debug-test")
    print(f"\nResponse citations: {response.citations}")

asyncio.run(test())