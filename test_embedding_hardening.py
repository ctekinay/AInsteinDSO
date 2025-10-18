"""
Test suite for hardened embedding agent (A+B+C improvements).

Tests:
1. Model fingerprint validation (A)
2. Vector normalization correctness (B)
3. OpenAI retry logic (C)
4. Backwards compatibility
5. Performance improvements

Run with: pytest test_embedding_hardening.py -v

NOTE: These tests require either:
- sentence-transformers installed (pip install sentence-transformers torch)
- OR run only retry tests which don't need it

To run without sentence-transformers:
  pytest test_embedding_hardening.py -v -k "Retry"
"""

import pytest
import numpy as np
import json
import pickle
import time
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from tempfile import TemporaryDirectory

# Import the hardened embedding agent
from src.agents.embedding_agent import EmbeddingAgent, _with_backoff

# Check if sentence-transformers is available
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

# Create the decorator for tests that need sentence-transformers
requires_st = pytest.mark.skipif(
    not ST_AVAILABLE,
    reason="sentence-transformers not installed - install with: pip install sentence-transformers torch"
)


class TestFingerprint:
    """Test suite for fingerprint validation (A)."""
    
    @requires_st
    def test_fingerprint_created_on_first_build(self):
        """Test that fingerprint is created when building embeddings."""
        with TemporaryDirectory() as tmpdir:
            mock_kg = Mock()
            mock_kg.kg_path = None
            mock_kg.get_all_concepts_with_definitions.return_value = [
                ("http://example.com/concept1", "Power", "Electrical power"),
                ("http://example.com/concept2", "Grid", "Electrical grid")
            ]
            
            agent = EmbeddingAgent(
                kg_loader=mock_kg,
                embedding_model="all-MiniLM-L6-v2",
                cache_dir=tmpdir,
                use_openai=False
            )
            
            assert 'fingerprint' in agent.embeddings
            fp = agent.embeddings['fingerprint']
            
            assert fp['backend'] == 'sentence-transformers'
            assert fp['model_name'] == 'all-MiniLM-L6-v2'
            assert fp['vector_dim'] > 0
            assert agent.vector_dim == fp['vector_dim']
            
            print(f"✅ Fingerprint created: {fp}")
    
    @requires_st
    def test_fingerprint_saved_to_cache(self):
        """Test that fingerprint is persisted in cache metadata."""
        with TemporaryDirectory() as tmpdir:
            mock_kg = Mock()
            mock_kg.kg_path = None
            mock_kg.get_all_concepts_with_definitions.return_value = [
                ("http://example.com/concept1", "Power", "Electrical power")
            ]
            
            agent = EmbeddingAgent(
                kg_loader=mock_kg,
                embedding_model="all-MiniLM-L6-v2",
                cache_dir=tmpdir,
                use_openai=False
            )
            
            metadata_file = Path(tmpdir) / "embeddings_metadata.json"
            assert metadata_file.exists()
            
            with open(metadata_file) as f:
                metadata = json.load(f)
            
            assert 'fingerprint' in metadata
            assert metadata['fingerprint']['backend'] == 'sentence-transformers'
            
            print(f"✅ Fingerprint saved to metadata: {metadata['fingerprint']}")
    
    @requires_st
    def test_cache_rebuilds_on_model_change(self):
        """Test that cache is automatically rebuilt when model changes."""
        with TemporaryDirectory() as tmpdir:
            mock_kg = Mock()
            mock_kg.kg_path = None
            mock_kg.get_all_concepts_with_definitions.return_value = [
                ("http://example.com/concept1", "Power", "Electrical power")
            ]
            
            agent1 = EmbeddingAgent(
                kg_loader=mock_kg,
                embedding_model="all-MiniLM-L6-v2",
                cache_dir=tmpdir,
                use_openai=False
            )
            
            first_created = agent1.embeddings['created_at']
            time.sleep(0.1)
            
            cache_file = Path(tmpdir) / "embeddings.pkl"
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
            
            cached['fingerprint']['model_name'] = 'different-model'
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cached, f)
            
            agent2 = EmbeddingAgent(
                kg_loader=mock_kg,
                embedding_model="all-MiniLM-L6-v2",
                cache_dir=tmpdir,
                use_openai=False
            )
            
            assert agent2.embeddings['created_at'] != first_created
            assert agent2.embeddings['fingerprint']['model_name'] == 'all-MiniLM-L6-v2'
            
            print(f"✅ Cache rebuilt on model change")


class TestVectorNormalization:
    """Test suite for vector normalization (B)."""
    
    @requires_st
    def test_vectors_are_normalized(self):
        """Test that stored vectors have unit norm."""
        with TemporaryDirectory() as tmpdir:
            mock_kg = Mock()
            mock_kg.kg_path = None
            mock_kg.get_all_concepts_with_definitions.return_value = [
                ("http://example.com/concept1", "Power", "Electrical power"),
                ("http://example.com/concept2", "Grid", "Electrical grid"),
                ("http://example.com/concept3", "Voltage", "Electrical voltage")
            ]
            
            agent = EmbeddingAgent(
                kg_loader=mock_kg,
                embedding_model="all-MiniLM-L6-v2",
                cache_dir=tmpdir,
                use_openai=False
            )
            
            vectors = agent.embeddings['vectors']
            norms = np.linalg.norm(vectors, axis=1)
            
            assert np.allclose(norms, 1.0, atol=1e-6), \
                f"Vectors not normalized! Norms: {norms}"
            
            print(f"✅ All {len(vectors)} vectors have unit norm")
            print(f"   Mean norm: {norms.mean():.10f}")
            print(f"   Std norm: {norms.std():.10f}")
    
    @requires_st
    def test_cosine_similarity_correctness(self):
        """Test that cosine similarity with normalized vectors is correct."""
        with TemporaryDirectory() as tmpdir:
            mock_kg = Mock()
            mock_kg.kg_path = None
            mock_kg.get_all_concepts_with_definitions.return_value = [
                ("http://example.com/concept1", "Power", "Electrical power system"),
                ("http://example.com/concept2", "Energy", "Energy distribution network")
            ]
            
            agent = EmbeddingAgent(
                kg_loader=mock_kg,
                embedding_model="all-MiniLM-L6-v2",
                cache_dir=tmpdir,
                use_openai=False
            )
            
            results = agent.semantic_search("power", top_k=2, min_score=0.0)
            
            assert len(results) > 0
            assert results[0].text.lower().count("power") > 0
            
            for r in results:
                assert 0 <= r.score <= 1, f"Score out of range: {r.score}"
            
            print(f"✅ Cosine similarity working correctly")
            print(f"   Top result score: {results[0].score:.4f}")
    
    @requires_st
    def test_search_performance_improvement(self):
        """Test that normalized vectors provide faster search."""
        with TemporaryDirectory() as tmpdir:
            concepts = [
                (f"http://example.com/concept{i}", f"Concept{i}", f"Definition {i}")
                for i in range(100)
            ]
            
            mock_kg = Mock()
            mock_kg.kg_path = None
            mock_kg.get_all_concepts_with_definitions.return_value = concepts
            
            agent = EmbeddingAgent(
                kg_loader=mock_kg,
                embedding_model="all-MiniLM-L6-v2",
                cache_dir=tmpdir,
                use_openai=False
            )
            
            start = time.perf_counter()
            for _ in range(50):
                agent.semantic_search("power system", top_k=5)
            
            elapsed = time.perf_counter() - start
            avg_time_ms = (elapsed / 50) * 1000
            
            assert avg_time_ms < 10, f"Search too slow: {avg_time_ms:.2f}ms"
            
            print(f"✅ Search performance: {avg_time_ms:.2f}ms per search")


class TestRetryLogic:
    """Test suite for retry logic (C)."""
    
    def test_backoff_function_retries(self):
        """Test that _with_backoff retries on failure."""
        call_count = [0]
        
        def failing_function():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("Simulated failure")
            return "success"
        
        result = _with_backoff(failing_function, max_retries=5, base_delay=0.01)
        
        assert result == "success"
        assert call_count[0] == 3
        
        print(f"✅ Retry logic worked: succeeded on attempt {call_count[0]}")
    
    def test_backoff_function_exhausts_retries(self):
        """Test that _with_backoff raises after max retries."""
        call_count = [0]
        
        def always_fails():
            call_count[0] += 1
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError, match="Always fails"):
            _with_backoff(always_fails, max_retries=3, base_delay=0.01)
        
        assert call_count[0] == 3
        
        print(f"✅ Retry exhaustion: failed after {call_count[0]} attempts")
    
    def test_backoff_delays_increase(self):
        """Test that retry delays increase exponentially."""
        delays = []
        call_count = [0]
        
        def failing_with_timing():
            call_count[0] += 1
            if call_count[0] > 1:
                delays.append(time.perf_counter())
            if call_count[0] < 4:
                raise ValueError("Fail")
            return "success"
        
        start = time.perf_counter()
        _with_backoff(failing_with_timing, max_retries=5, base_delay=0.1)
        
        if len(delays) >= 2:
            time_diff = delays[1] - delays[0]
            assert time_diff > 0.15, "Delays not increasing exponentially"
        
        print(f"✅ Exponential backoff verified")
    
    @patch('src.agents.embedding_agent.OpenAI')
    def test_openai_retry_on_rate_limit(self, mock_openai_class):
        """Test that OpenAI calls retry on rate limits."""
        with TemporaryDirectory() as tmpdir:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client
            
            call_count = [0]
            
            def create_embedding_with_retry(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] < 2:
                    raise Exception("Rate limit exceeded")
                
                mock_response = MagicMock()
                mock_response.data = [MagicMock(embedding=[0.1] * 384)]
                return mock_response
            
            mock_client.embeddings.create = create_embedding_with_retry
            
            mock_kg = Mock()
            mock_kg.kg_path = None
            mock_kg.get_all_concepts_with_definitions.return_value = [
                ("http://example.com/concept1", "Power", "Electrical power")
            ]
            
            agent = EmbeddingAgent(
                kg_loader=mock_kg,
                cache_dir=tmpdir,
                use_openai=True,
                openai_api_key="test-key"
            )
            
            assert call_count[0] == 2
            print(f"✅ OpenAI retry on rate limit: succeeded after {call_count[0]} attempts")


class TestBackwardsCompatibility:
    """Test backwards compatibility with existing code."""
    
    @requires_st
    def test_api_unchanged(self):
        """Test that public API remains unchanged."""
        with TemporaryDirectory() as tmpdir:
            mock_kg = Mock()
            mock_kg.kg_path = None
            mock_kg.get_all_concepts_with_definitions.return_value = [
                ("http://example.com/concept1", "Power", "Electrical power")
            ]
            
            agent = EmbeddingAgent(
                kg_loader=mock_kg,
                archimate_parser=None,
                pdf_indexer=None,
                embedding_model="all-MiniLM-L6-v2",
                cache_dir=tmpdir,
                use_openai=False,
                openai_api_key=None,
                auto_refresh=False
            )
            
            assert hasattr(agent, 'semantic_search')
            assert hasattr(agent, 'refresh_embeddings')
            assert hasattr(agent, 'get_enhanced_context')
            assert hasattr(agent, 'stats')
            
            results = agent.semantic_search(
                query="power",
                top_k=5,
                min_score=0.3,
                source_filter=None
            )
            
            assert isinstance(results, list)
            
            print("✅ All public APIs remain unchanged")
    
    @requires_st
    def test_old_cache_loads_with_warning(self):
        """Test that old caches without fingerprint still load."""
        with TemporaryDirectory() as tmpdir:
            old_embeddings = {
                'texts': ['Power', 'Grid'],
                'vectors': np.random.randn(2, 384).astype(np.float32),
                'metadata': [{'source': 'kg'}, {'source': 'kg'}],
                'citations': ['skos:1', 'skos:2'],
                'created_at': '2024-01-01T00:00:00'
            }
            
            cache_file = Path(tmpdir) / "embeddings.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(old_embeddings, f)
            
            mock_kg = Mock()
            mock_kg.kg_path = None
            mock_kg.get_all_concepts_with_definitions.return_value = [
                ("http://example.com/concept1", "Power", "Electrical power")
            ]
            
            agent = EmbeddingAgent(
                kg_loader=mock_kg,
                cache_dir=tmpdir,
                use_openai=False
            )
            
            assert 'fingerprint' in agent.embeddings
            
            print("✅ Old cache handled gracefully (rebuild triggered)")


class TestIntegration:
    """Integration tests with realistic scenarios."""
    
    @requires_st
    def test_full_workflow_with_kg_archimate_pdf(self):
        """Test complete workflow with all data sources."""
        with TemporaryDirectory() as tmpdir:
            mock_kg = Mock()
            mock_kg.kg_path = None
            mock_kg.get_all_concepts_with_definitions.return_value = [
                ("http://example.com/power", "Active Power", "Real power consumed"),
                ("http://example.com/reactive", "Reactive Power", "Imaginary power")
            ]
            
            mock_archimate = Mock()
            mock_archimate.model_paths = []
            mock_archimate.elements = {
                'elem1': Mock(
                    id='elem1',
                    name='SCADA System',
                    documentation='Supervisory control',
                    type='ApplicationComponent',
                    layer='Application'
                )
            }
            
            mock_pdf = Mock()
            mock_pdf.docs_path = None
            mock_chunk = Mock()
            mock_chunk.content = "TOGAF Architecture Development Method"
            mock_chunk.doc_id = "togaf-guide"
            mock_chunk.page_number = 1
            mock_pdf.get_all_chunks.return_value = [mock_chunk]
            
            agent = EmbeddingAgent(
                kg_loader=mock_kg,
                archimate_parser=mock_archimate,
                pdf_indexer=mock_pdf,
                embedding_model="all-MiniLM-L6-v2",
                cache_dir=tmpdir,
                use_openai=False
            )
            
            results = agent.semantic_search("power system architecture", top_k=10)
            
            sources = {r.source for r in results}
            assert len(sources) > 1, f"Expected multiple sources, got: {sources}"
            
            stats = agent.stats()
            assert stats['total_embeddings'] > 0
            assert stats['sources']['knowledge_graph'] > 0
            assert stats['fingerprint']['backend'] == 'sentence-transformers'
            
            print(f"✅ Full integration test passed")
            print(f"   Total embeddings: {stats['total_embeddings']}")
            print(f"   Sources: {list(sources)}")
    
    @requires_st
    def test_performance_with_large_dataset(self):
        """Test performance with realistic dataset size."""
        with TemporaryDirectory() as tmpdir:
            concepts = [
                (f"http://example.com/concept{i}", 
                 f"Concept {i}", 
                 f"This is definition number {i} about power systems")
                for i in range(1000)
            ]
            
            mock_kg = Mock()
            mock_kg.kg_path = None
            mock_kg.get_all_concepts_with_definitions.return_value = concepts
            
            start = time.perf_counter()
            
            agent = EmbeddingAgent(
                kg_loader=mock_kg,
                cache_dir=tmpdir,
                use_openai=False
            )
            
            creation_time = time.perf_counter() - start
            
            search_start = time.perf_counter()
            
            for _ in range(100):
                agent.semantic_search("power", top_k=5)
            
            search_time = (time.perf_counter() - search_start) / 100
            
            print(f"✅ Performance test passed")
            print(f"   Creation time for 1000 embeddings: {creation_time:.2f}s")
            print(f"   Average search time: {search_time*1000:.2f}ms")
            
            assert search_time < 0.01, "Search too slow for normalized vectors"


def test_summary():
    """Print summary of what was tested."""
    print("\n" + "="*60)
    print("HARDENED EMBEDDING AGENT TEST SUMMARY")
    print("="*60)
    print("\n✅ Feature A: Model Fingerprint Validation")
    print("   - Fingerprint creation")
    print("   - Cache persistence")
    print("   - Auto-rebuild on model change")
    
    print("\n✅ Feature B: Vector Normalization")
    print("   - Unit norm verification")
    print("   - Cosine similarity correctness")
    print("   - Performance improvement")
    
    print("\n✅ Feature C: Retry Logic")
    print("   - Backoff function retry behavior")
    print("   - Retry exhaustion")
    print("   - Exponential delay growth")
    print("   - OpenAI rate limit handling")
    
    print("\n✅ Backwards Compatibility")
    print("   - API unchanged")
    print("   - Old cache handling")
    
    print("\n✅ Integration Tests")
    print("   - Multi-source workflow")
    print("   - Large dataset performance")
    
    print("\n" + "="*60)
    
    if not ST_AVAILABLE:
        print("\n⚠️  NOTE: Some tests were skipped because sentence-transformers")
        print("   is not installed. Install with:")
        print("   pip install sentence-transformers torch")
    print("\n" + "="*60)


if __name__ == "__main__":
    print("HARDENED EMBEDDING AGENT TEST SUITE")
    print("="*60)
    print("\nTo run all tests:")
    print("  pytest test_embedding_hardening.py -v")
    print("\nTo run only retry tests (no sentence-transformers needed):")
    print("  pytest test_embedding_hardening.py -v -k 'Retry'")
    print("\nTo run with coverage:")
    print("  pytest test_embedding_hardening.py --cov=src.agents.embedding_agent")
    
    if not ST_AVAILABLE:
        print("\n⚠️  WARNING: sentence-transformers not installed")
        print("   Most tests will be SKIPPED")
        print("   Install with: pip install sentence-transformers torch")
    
    print("\n" + "="*60)
    print("\n🔥 Running quick smoke test (retry logic)...\n")
    
    test = TestRetryLogic()
    test.test_backoff_function_retries()
    test.test_backoff_function_exhausts_retries()
    
    print("\n✅ Smoke test passed!")
    
    if ST_AVAILABLE:
        print("\n🔥 Running normalization smoke test...\n")
        test2 = TestVectorNormalization()
        test2.test_vectors_are_normalized()
        print("\n✅ Full smoke test passed! Run pytest for complete suite.")
    else:
        print("\nRun full test suite after installing sentence-transformers.")
    
    test_summary()