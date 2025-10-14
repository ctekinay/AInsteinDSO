# Claude Coding Agent Instructions - Alliander EA Assistant

## Project Context
You are building an Energy Systems Enterprise Architecture AI Assistant for Alliander, a Dutch Distribution System Operator (DSO). This system helps Enterprise Architects model energy infrastructure using ArchiMate while ensuring compliance with IEC standards, TOGAF methodology, and Dutch energy regulations.

## Critical Requirements

### 1. GROUNDING IS MANDATORY
- **NEVER** generate a response without citations
- Every response MUST include at least one of: `archi:id-`, `skos:`, `iec:`, `togaf:adm:`, `entsoe:`, `lido:`
- If you cannot ground a response, raise `UngroundedReplyError` - do not guess

### 2. Knowledge Sources (in priority order)
1. **energy_knowledge_graph.ttl** - 39,100+ triples with IEC, ENTSOE, EUR-LEX terms
2. **ArchiMate models** - XML files in data/models/
3. **TOGAF patterns** - Indexed separately in togaf_core RAG
4. **Only use vectors as last resort** - Structured data first, always

### 3. Architecture Principles
- **4R+G+C Pipeline**: Reflect → Retrieve → Refine → Ground → Critic → Validate
- **Router-based retrieval**: Check for domain terms BEFORE using embeddings
- **PR-only changes**: Never modify models directly, always generate PR drafts
- **Human review required**: When confidence < 0.75

## File Structure to Maintain
```
alliander-ea-assistant/
├── src/
│   ├── knowledge/          # Knowledge graph and SPARQL
│   ├── safety/             # Grounding and validation
│   ├── routing/            # Query router
│   ├── archimate/          # ArchiMate parsing
│   ├── validation/         # Critic and TOGAF compliance
│   ├── agent/              # Main pipeline
│   ├── api/                # FastAPI interface
│   └── evaluation/         # Quality metrics
├── data/
│   ├── energy_knowledge_graph.ttl  # DO NOT MODIFY - source of truth
│   ├── models/             # ArchiMate XML files
│   └── test_cases.json    # Evaluation test cases
├── config/
│   └── vocabularies.json   # Extracted terms for routing
├── tests/
│   ├── unit/              # Component tests
│   └── integration/       # Full pipeline tests
└── scripts/
    ├── validate_kg.py     # Knowledge graph validation
    └── run_evaluation.py  # Quality gate checks
```

## Code Standards

### Python Requirements
- Python 3.11+
- Type hints on ALL functions
- Docstrings with parameter descriptions
- Async/await for I/O operations
- Comprehensive error handling with specific exceptions

### Testing Requirements
- Every module must have corresponding tests
- Integration tests for full pipeline
- Mock external dependencies in unit tests
- Test coverage > 80%

### Critical Classes to Implement

1. **GroundingCheck** (src/safety/grounding.py)
   - Must validate ALL responses have citations
   - Suggest citations if missing
   - Raise exception if cannot ground

2. **QueryRouter** (src/routing/query_router.py)
   - Check domain terms FIRST
   - Route to: structured_model | togaf_method | unstructured_docs
   - Load vocabularies from config/

3. **Critic** (src/validation/critic.py)
   - Rank top-3 suggestions
   - Mark irrelevant items (max 18%)
   - Force human review if confidence < 0.75

4. **ProductionEAAgent** (src/agent/ea_assistant.py)
   - Implement full pipeline
   - Store audit trail in context_store
   - Generate PR drafts for changes

## Quality Gates (MUST PASS)
- **Grounding failures**: 0 (absolutely no ungrounded responses)
- **Top-1 accuracy**: ≥ 80%
- **Abstention rate**: ≥ 15% (good to abstain when uncertain)
- **Response time**: < 3 seconds
- **All model changes**: via PR only

## Domain Vocabulary Samples

### IEC Terms to Recognize
- ActivePower, ReactivePower, Equipment
- Conductor, Breaker, Transformer
- IEC 61968, IEC 61970, CIM

### ArchiMate Elements
- Business: Actor, Role, Process, Service, Capability
- Application: Component, Service, Interface
- Technology: Node, Device, System Software

### TOGAF ADM Phases
- Phase A: Architecture Vision
- Phase B: Business Architecture → Business Layer elements
- Phase C: Information Systems → Application/Data elements
- Phase D: Technology Architecture → Technology Layer elements

## Example Patterns

### Good Citation Example
```python
response = "Based on TOGAF Phase B (togaf:adm:B), use Business Capability (archi:id-cap-001) for modeling grid congestion per IEC 61968 (iec:GridCongestion)."
# Has multiple citations ✓
```

### Bad Response Example
```python
response = "You should use a Business Capability for this."
# NO CITATIONS - Must raise UngroundedReplyError ✗
```

## Special Instructions

### When Creating Tests
Always include these test cases:
1. Electronic Court Filing → Capability (from research paper)
2. Grid congestion management (energy domain specific)
3. Query without possible grounding (should abstain)

### When Implementing Router
Priority order:
1. Check for IEC/energy terms → structured_model
2. Check for TOGAF/ArchiMate terms → togaf_method  
3. Default → unstructured_docs

### When Generating PR Drafts
Include:
- Branch name: `ea-assist-{session_id[:8]}`
- jArchi script for changes
- Diff preview
- Citations for why changes are needed

## Performance Targets
- Knowledge graph load: < 3000ms (realistic for 39K triples)
- SPARQL query: < 1500ms P95 (with 35,000x cache speedup)
- Full pipeline: < 3s P50
- Router decision: < 50ms

## Error Messages
- UngroundedReplyError: "Response lacks required citations. Must include at least one of: {prefixes}"
- LowConfidenceError: "Confidence {score} below threshold. Human review required."
- InvalidModelChangeError: "Model changes must be submitted via PR, not direct modification."

## Remember
1. **Grounding is not optional** - it's the #1 requirement
2. **Structured data beats embeddings** - always check KG first
3. **When in doubt, abstain** - better to request human review than give wrong answer
4. **Every decision is audited** - context_store tracks everything

## Git Workflow Instructions

### Commit Strategy
- **Commit after EVERY completed component** (not just at the end)
- **Run tests before committing** - never commit broken code
- **Use conventional commits** format (see below)
- **Push to feature branch**, not main directly

### Commit Message Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature implementation
- `fix`: Bug fix
- `test`: Adding tests
- `docs`: Documentation only
- `refactor`: Code change that neither fixes nor adds feature
- `chore`: Changes to build process or auxiliary tools

Examples:
```
feat(grounding): implement citation validation with UngroundedReplyError

- Add GroundingCheck class with assert_citations method
- Implement citation pattern matching for all required prefixes
- Add auto-suggestion for missing citations
- Comprehensive test coverage for grounding scenarios

Refs: claude.md requirements section
```

### Git Commands to Execute

After creating each module:
```bash
# Run tests first
pytest tests/test_<module>.py -v

# If tests pass, stage changes
git add .

# Commit with descriptive message
git commit -m "feat(<module>): <what you implemented>"

# Push to feature branch
git push origin feat/ea-assistant-implementation
```

### Branch Strategy
- Main branch: `main` (protected)
- Feature branch: `feat/ea-assistant-implementation`
- PR will be created after all quality gates pass

### Automated Commit Points
1. After project structure creation → `chore(setup): initialize project structure with Poetry`
2. After KG loader → `feat(knowledge): implement knowledge graph loader with validation`
3. After grounding → `feat(safety): add grounding check with citation enforcement`
4. After router → `feat(routing): implement query router with domain term detection`
5. After critic → `feat(validation): add critic module with confidence assessment`
6. After ArchiMate parser → `feat(archimate): implement model parser with TOGAF mapping`
7. After main pipeline → `feat(agent): integrate full 4R+G+C pipeline`
8. After API → `feat(api): add FastAPI endpoints with async handling`
9. After evaluation → `feat(evaluation): implement quality gate harness`
10. After documentation → `docs: complete README and ADRs`

## Current Implementation Status
- [x] Project structure created
- [x] Knowledge graph loader implemented
- [x] Grounding check operational (FULLY TESTED AND WORKING)
- [x] Router configured (FULLY IMPLEMENTED AND TESTED)
- [x] Critic module complete (FULLY IMPLEMENTED AND WORKING)
- [x] Main pipeline integrated (FULLY IMPLEMENTED - ProductionEAAgent)
- [ ] API endpoints active
- [ ] Quality gates passing
- [ ] Documentation complete

Update this checklist as you complete each component.