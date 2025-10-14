#!/usr/bin/env python3
from pathlib import Path
import json

def check_pdf_indexing():
    """Check PDF indexing status."""

    print("=== PDF DOCUMENT STATUS ===\n")

    # Check PDF files
    docs_dir = Path("data/docs")
    print(f"Docs directory: {docs_dir.absolute()}")
    print(f"Directory exists: {docs_dir.exists()}")

    if docs_dir.exists():
        pdf_files = list(docs_dir.glob("*.pdf"))
        print(f"PDF files found: {len(pdf_files)}")
        for pdf in pdf_files:
            print(f"  - {pdf.name}: {pdf.stat().st_size / 1024:.1f} KB")

    # Check document index
    index_path = Path("data/document_index.json")
    print(f"\nDocument index: {index_path}")
    print(f"Index exists: {index_path.exists()}")

    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        print(f"Indexed chunks: {len(index.get('chunks', []))}")

        # Show sample chunks
        for i, chunk in enumerate(index.get('chunks', [])[:3]):
            print(f"\n  Chunk {i+1}:")
            print(f"    Doc: {chunk.get('doc_id', 'unknown')}")
            print(f"    Title: {chunk.get('title', 'untitled')}")
            print(f"    Page: {chunk.get('page_number', '?')}")
            print(f"    Content preview: {chunk.get('content', '')[:100]}...")
    else:
        print("\n⚠️ No document index found - PDFs not indexed!")

        # Try to create the index
        print("\nAttempting to create document index...")
        try:
            from src.documents.pdf_indexer import PDFIndexer
            indexer = PDFIndexer("data/docs")
            indexer.load_or_create_index()
            print("✅ Index created successfully!")
        except Exception as e:
            print(f"❌ Failed to create index: {e}")

if __name__ == "__main__":
    check_pdf_indexing()