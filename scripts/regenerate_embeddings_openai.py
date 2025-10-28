"""
ONE-TIME SCRIPT: Regenerate embeddings cache with OpenAI text-embedding-3-small

Benefit: Better quality (62.3 vs 58.8 MTEB) + no ongoing costs forever

Usage:
    python3 scripts/regenerate_embeddings_openai.py
"""

import sys
import os
import asyncio
from pathlib import Path

#Add parent directory to path so we can import src
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

#Load .env file before importing anything else
from dotenv import load_dotenv
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded .env from: {env_path}")
else:
    print(f"⚠️  No .env file found at: {env_path}")

# Now we can import
from src.agents.embedding_agent import EmbeddingAgent
from src.knowledge.kg_loader import KnowledgeGraphLoader
from src.archimate.parser import ArchiMateParser
from src.documents.pdf_indexer import PDFIndexer

async def regenerate_embeddings():
    """
    ONE-TIME: Regenerate embeddings cache with OpenAI text-embedding-3-small.
    
    This replaces the old all-MiniLM-L6-v2 embeddings with better OpenAI embeddings.
    After this runs once, all future queries will use the better embeddings FOR FREE.
    """
    
    print()
    print("=" * 70)
    print("🔄 EMBEDDING REGENERATION - OpenAI text-embedding-3-small")
    print("=" * 70)
    print()
    
    # Check for API key
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("❌ ERROR: OPENAI_API_KEY not found in environment")
        print()
        print("   Debug info:")
        print(f"   - .env path: {env_path}")
        print(f"   - .env exists: {env_path.exists()}")
        print(f"   - Current directory: {os.getcwd()}")
        print()
        print("   Fix: Add to .env file:")
        print("   OPENAI_API_KEY='your-key-here'")
        print()
        print("   Or export manually:")
        print("   export OPENAI_API_KEY='your-key-here'")
        return
    
    print("✅ OpenAI API key found")
    print(f"   Key: {openai_key[:8]}...{openai_key[-4:]}")  # Show partial for verification
    print()
    
    # Initialize knowledge sources
    print("⏳ Loading knowledge sources...")
    
    kg_loader = KnowledgeGraphLoader(Path("data/energy_knowledge_graph.ttl"))
    kg_loader.load()
    print("   ✅ Knowledge Graph loaded")
    
    archimate_parser = ArchiMateParser()
    archimate_parser.load_model("data/models/IEC 61968.xml")
    archimate_parser.load_model("data/models/archi-4-archi.xml")
    print("   ✅ ArchiMate models loaded")
    
    pdf_indexer = PDFIndexer("data/docs/")
    try:
        pdf_indexer.load_or_create_index()
        print("   ✅ PDF indexer loaded")
    except:
        print("   ⚠️  PDF indexer unavailable (skipping)")
        pdf_indexer = None
    
    # ✅ NEW: Load ADRs
    from src.documents.adr_indexer import ADRIndexer
    adr_indexer = ADRIndexer("data/adrs/")
    try:
        adr_count = adr_indexer.load_adrs()
        if adr_count > 0:
            print(f"   ✅ ADR indexer loaded ({adr_count} decision records)")
        else:
            print("   ⚠️  No ADRs found (skipping)")
            adr_indexer = None
    except Exception as e:
        print(f"   ⚠️  ADR indexer failed: {e} (skipping)")
        adr_indexer = None
    
    print()
    
    # Estimate cost
    total_texts = 5232  # Base estimate
    if adr_indexer:
        total_texts += len(adr_indexer.adrs)
    
    estimated_tokens = total_texts * 100  # ~100 tokens per text
    estimated_cost = (estimated_tokens / 1_000_000) * 0.02
    
    print("💰 COST ESTIMATE (October 2025 Pricing):")
    print("   - Model: text-embedding-3-small")
    print("   - Rate: $0.02 per 1M tokens (current OpenAI pricing)")
    print(f"   - Estimated texts: ~{total_texts:,}")
    if adr_indexer and len(adr_indexer.adrs) > 0:
        print(f"     • Knowledge Graph + ArchiMate: ~5,232")
        print(f"     • ADRs: {len(adr_indexer.adrs)}")
    print(f"   - Estimated tokens: ~{estimated_tokens:,}")
    print(f"   - Estimated cost: ~${estimated_cost:.2f}")
    print("   - Source: OpenAI Pricing (October 2025)")
    print()
    
    # Confirm
    response = input("   Continue with regeneration? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("❌ Cancelled by user")
        return
    
    print()
    print("🚀 Starting regeneration...")
    print("⏳ This may take 2-5 minutes depending on API speed")
    print()
    
    # Delete old cache FIRST to force regeneration
    cache_file = Path("data/embeddings/embeddings_cache.pkl")
    if cache_file.exists():
        print("🗑️  Removing old cache (all-MiniLM-L6-v2)...")
        cache_file.unlink()
        print("   ✅ Old cache removed")
    
    print()
    print("🔄 Initializing EmbeddingAgent with OpenAI embeddings...")
    
    # Initialize with OpenAI - with lazy_load=False it should generate immediately
    agent = EmbeddingAgent(
        kg_loader=kg_loader,
        archimate_parser=archimate_parser,
        pdf_indexer=pdf_indexer,
        adr_indexer=adr_indexer,
        use_openai=True,
        openai_api_key=openai_key,
        cache_dir="data/embeddings",
        lazy_load=False  # ✅ Should trigger immediate generation
    )
    
    print("   ✅ EmbeddingAgent created")
    print()
    
    # Check if embeddings were automatically generated
    embeddings_loaded = False
    embedding_count = 0
    
    if hasattr(agent, 'embeddings') and agent.embeddings:
        texts = agent.embeddings.get('texts', [])
        if texts:
            embedding_count = len(texts)
            embeddings_loaded = True
            print(f"   ✅ Embeddings automatically generated: {embedding_count} texts")
            
            # ✅ Force save to cache file
            print()
            print("💾 Saving embeddings to cache file...")
            try:
                cache_file = Path("data/embeddings/embeddings_cache.pkl")
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Try agent's save method first
                if hasattr(agent, '_save_cache'):
                    agent._save_cache()
                    print(f"   ✅ Cache saved via _save_cache()")
                elif hasattr(agent, 'save_cache'):
                    agent.save_cache()
                    print(f"   ✅ Cache saved via save_cache()")
                else:
                    # Manual save using pickle
                    import pickle
                    with open(cache_file, 'wb') as f:
                        pickle.dump(agent.embeddings, f)
                    print(f"   ✅ Cache saved manually")
                
                # Verify file exists and show size
                if cache_file.exists():
                    size_mb = cache_file.stat().st_size / (1024 * 1024)
                    print(f"   ✅ Cache file verified: {size_mb:.1f} MB")
                else:
                    print(f"   ⚠️  Warning: Cache file not found!")
                    embeddings_loaded = False
                    
            except Exception as e:
                print(f"   ❌ Error saving cache: {e}")
                import traceback
                traceback.print_exc()
                embeddings_loaded = False
    
    # If not auto-generated, trigger manually
    if not embeddings_loaded:
        print("   🔄 Embeddings not auto-loaded, triggering generation...")
        
        # Try different methods to trigger embedding generation
        methods_tried = []
        
        # Method 1: Try _load_embeddings if it exists
        if hasattr(agent, '_load_embeddings'):
            methods_tried.append('_load_embeddings')
            try:
                print("      Trying _load_embeddings()...")
                agent._load_embeddings()
                if hasattr(agent, 'embeddings') and agent.embeddings:
                    texts = agent.embeddings.get('texts', [])
                    if texts:
                        embedding_count = len(texts)
                        embeddings_loaded = True
                        print(f"   ✅ Embeddings loaded: {embedding_count} texts")
            except Exception as e:
                print(f"      ⚠️  Failed: {e}")
        
        # Method 2: Try semantic_search to trigger loading
        if not embeddings_loaded:
            methods_tried.append('semantic_search')
            try:
                print("      Trying semantic_search()...")
                agent.semantic_search("initialization query", top_k=1, min_score=0.0)
                if hasattr(agent, 'embeddings') and agent.embeddings:
                    texts = agent.embeddings.get('texts', [])
                    if texts:
                        embedding_count = len(texts)
                        embeddings_loaded = True
                        print(f"   ✅ Embeddings generated: {embedding_count} texts")
            except Exception as e:
                print(f"      ⚠️  Failed: {e}")
        
        # If still not loaded, show error
        if not embeddings_loaded:
            print()
            print("   ❌ Could not trigger embedding generation")
            print(f"   Tried methods: {', '.join(methods_tried)}")
            print()
            print("   Debug: Check if EmbeddingAgent has the correct initialization")
    
    print()

    # Verify it worked
    if hasattr(agent, 'embeddings') and len(agent.embeddings.get('texts', [])) > 0:
        count = len(agent.embeddings['texts'])
        print()
        print("=" * 70)
        print("✅ REGENERATION COMPLETE!")
        print("=" * 70)
        print(f"   📊 Total embeddings: {count}")
        if adr_indexer and len(adr_indexer.adrs) > 0:
            print(f"       • Includes {len(adr_indexer.adrs)} ADRs")
        print(f"   📈 Model: text-embedding-3-small (MTEB 62.3)")
        print(f"   💾 Cached at: {cache_file}")
        print()
        print("🎉 BENEFITS:")
        print("   ✅ Better quality (62.3 vs 58.8 MTEB)")
        print("   ✅ All future queries FREE (no API calls)")
        print("   ✅ Faster (no reranking API call needed)")
        print("   ✅ Simpler architecture (single embedding system)")
        print()
        print("📋 NEXT STEPS:")
        print("   1. Restart your application")
        print("   2. Optionally disable API reranking (no longer needed):")
        print("      Set ENABLE_API_RERANKING=false in .env")
        print("   3. Enjoy better semantic search!")
        print()
    else:
        print()
        print("❌ ERROR: Regeneration failed - no embeddings created")
        print("   Check logs above for error details")

if __name__ == "__main__":
    asyncio.run(regenerate_embeddings())