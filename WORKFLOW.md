# EA Assistant Workflow

## Query Processing Pipeline

1. **REFLECT**: Analyze query intent
   - Identify energy terms (IEC, ENTSOE)
   - Detect ArchiMate elements
   - Determine TOGAF phase

2. **ROUTE**: Direct to appropriate source
   - structured_model → KG + ArchiMate
   - togaf_method → TOGAF patterns
   - unstructured_docs → PDFs

3. **RETRIEVE**: Gather relevant information
   - SPARQL for definitions
   - XPath for ArchiMate elements
   - Vector search for documents

4. **REFINE**: Enhance with LLM if available
   - Synthesize multiple sources
   - Add domain context
   - Generate recommendations

5. **GROUND**: Ensure citations
   - Validate all claims have sources
   - Format citations properly
   - Raise UngroundedReplyError if needed

6. **CRITIC**: Assess confidence
   - Calculate relevance scores
   - Check for contradictions
   - Trigger human review if <75%

7. **VALIDATE**: TOGAF alignment
   - Check phase appropriateness
   - Validate layer consistency