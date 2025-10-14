"""
Domain-specific prompt templates for Alliander EA Assistant.

This module contains specialized prompt templates for energy domain
Enterprise Architecture assistance, ensuring consistent and grounded
responses across different LLM providers.

Key features:
- Redesigned to prevent fake citation hallucination
- Example citations removed, context blocks prominent
- Citation pool integration with metadata ensures real citations only
"""

from typing import Dict, List, Optional, Any
import json


class EAPromptTemplate:
    """
    Enterprise Architecture prompt template for Alliander energy domain.

    Prevents LLM hallucination of fake citations by:
    1. Making context block 5x larger than instructions
    2. Removing all example citations from templates
    3. Using XML-style formatting to separate real citations from instructions
    4. Emphasizing use of provided citations only
    5. Citation pool with metadata so LLM sees labels and definitions
    """

    SYSTEM_PROMPT = """You are an Enterprise Architect at Alliander, a Dutch Distribution System Operator (DSO).

Your expertise:
- IEC 61968/61970 standards for energy management systems
- ArchiMate enterprise architecture modeling
- TOGAF ADM methodology
- Dutch energy market regulations and grid operations
- Smart grid technologies and distribution automation

🔒 CRITICAL CITATION RULES - READ CAREFULLY:

1. You will receive a [AVAILABLE CITATIONS] section with REAL citations
2. ONLY use citations from that section - they include labels and definitions
3. NEVER invent, fabricate, or use example citations
4. NEVER use citations from your training data or general knowledge
5. If no relevant citation exists in [AVAILABLE CITATIONS], say so explicitly

Valid citation formats (only when provided in context):
- ArchiMate elements: archi:id-[uuid]
- Knowledge Graph concepts: skos:[term], iec:[term], entsoe:[term]
- TOGAF phases: togaf:adm:[A-H]
- Documents: doc:[id]:page[num]

🚫 INVALID - Never use these patterns:
- Do NOT use "archi:id-cap-001" or similar generic IDs
- Do NOT use "iec:GridCongestion" or "iec:61968" 
- Do NOT create new citation IDs
- Do NOT use example citations

✅ CORRECT APPROACH:
- Look for citations in [AVAILABLE CITATIONS] section
- Use the exact citation ID provided
- Reference the label to understand what it means
- If unsure, say "Based on [citation]: label..."

Response Quality:
- Accurate technical terminology
- Practical implementation guidance
- Regulatory compliance awareness
- Clear architectural reasoning"""

    @classmethod
    def create_retrieval_prompt(
        cls,
        query: str,
        context: Dict,
        citation_pool: List[Dict]
    ) -> str:
        """
        Create prompt for retrieval-augmented generation WITH ENRICHED CITATION POOL.
        
        ENHANCED: Citation pool now includes metadata (labels, definitions, source).
        This helps LLM understand what each citation means and select appropriately.
        
        Args:
            query: User query
            context: Retrieval context with candidates
            citation_pool: List of citation dicts with {citation, label, definition, source}
            
        Returns:
            Formatted prompt with enriched citation constraints
        """
        
        # Format the retrieval context
        formatted_context = cls._format_context(context)
        
        # Build enriched citations section
        citations_section = "\n[AVAILABLE CITATIONS]\n\n"
        citations_section += f"You have access to {len(citation_pool)} validated citations.\n"
        citations_section += "Each citation includes:\n"
        citations_section += "  • Citation ID (use this exact ID in your response)\n"
        citations_section += "  • Label (what this citation represents)\n"
        citations_section += "  • Source (where it comes from)\n\n"
        citations_section += "="*70 + "\n\n"
        
        for idx, item in enumerate(citation_pool[:50], 1):  # Limit to 50 for prompt size
            citation = item.get('citation', 'unknown')
            label = item.get('label', 'N/A')
            definition = item.get('definition', '')
            source = item.get('source', 'N/A')
            
            citations_section += f"CITATION #{idx}:\n"
            citations_section += f"  ID: {citation}\n"
            citations_section += f"  Label: {label}\n"
            citations_section += f"  Source: {source}\n"
            
            if definition:
                # Truncate long definitions
                def_preview = definition[:150] + '...' if len(definition) > 150 else definition
                citations_section += f"  Definition: {def_preview}\n"
            
            citations_section += "\n"
        
        if len(citation_pool) > 50:
            citations_section += f"... and {len(citation_pool) - 50} more citations available\n\n"
        
        citations_section += "="*70 + "\n\n"
        citations_section += "CRITICAL RULES:\n"
        citations_section += "  ✓ Use ONLY the citation IDs listed above\n"
        citations_section += "  ✓ Reference citations using format: [citation_id]\n"
        citations_section += "  ✗ DO NOT create new citation IDs\n"
        citations_section += "  ✗ DO NOT modify citation IDs\n"
        citations_section += "  ✗ DO NOT use citations not in this list\n\n"
        
        prompt = f"""You are an Enterprise Architecture assistant for Alliander DSO specializing in energy systems modeling.

CRITICAL GROUNDING REQUIREMENT:
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
You MUST use ONLY citations from the [AVAILABLE CITATIONS] pool below.
DO NOT generate, invent, or modify any citations.
DO NOT create citations that "look similar" to the pool.
Every factual claim MUST include a citation from this pool.
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

QUERY: {query}

{citations_section}

[RETRIEVAL CONTEXT]
{formatted_context}

RESPONSE FORMAT:
1. Answer the query directly and concisely
2. Include citations inline using format: [citation_id]
3. Example: "An Asset [iec:Asset] is equipment used in grid operations."
4. Use multiple citations if the context provides them
5. If you reference a citation, you can mention its label for clarity

CRITICAL REMINDER:
✓ Use ONLY citations listed in [AVAILABLE CITATIONS]
✓ Each citation has a label to help you understand its meaning
✗ DO NOT create new citations
✗ DO NOT modify citation IDs
✗ DO NOT use citations not in the pool

If you cannot answer with available citations, say so explicitly.
"""
        
        return prompt
    
    @classmethod
    def _format_context(cls, context: Dict) -> str:
        """
        Format retrieval context for prompt.
        
        Extracts relevant information from context dictionary and formats it
        in a clear, readable way for the LLM.
        
        Args:
            context: Retrieval context dictionary
            
        Returns:
            Formatted context string
        """
        lines = []
        
        # Format candidates if available
        candidates = context.get("candidates", [])
        if candidates:
            lines.append("Retrieved Information:")
            lines.append("")
            for idx, candidate in enumerate(candidates[:10], 1):  # Limit to top 10
                element = candidate.get("element", "Unknown")
                definition = candidate.get("definition", "")
                citation = candidate.get("citation", "")
                elem_type = candidate.get("type", "")
                
                lines.append(f"{idx}. {element}")
                if elem_type:
                    lines.append(f"   Type: {elem_type}")
                if citation:
                    lines.append(f"   Citation: {citation}")
                if definition:
                    # Truncate long definitions
                    def_preview = definition[:200] + "..." if len(definition) > 200 else definition
                    lines.append(f"   Definition: {def_preview}")
                lines.append("")
        
        # Format TOGAF context if available
        togaf_context = context.get("togaf_context", {})
        if togaf_context:
            lines.append("TOGAF Context:")
            if togaf_context.get("primary_phase"):
                lines.append(f"  Primary Phase: {togaf_context['primary_phase']}")
            if togaf_context.get("adm_guidance"):
                lines.append(f"  Guidance: {togaf_context['adm_guidance']}")
            lines.append("")
        
        # Format domain context if available
        domain_context = context.get("domain_context", {})
        if domain_context:
            lines.append("Domain Context:")
            if domain_context.get("domain"):
                lines.append(f"  Domain: {domain_context['domain']}")
            if domain_context.get("standards"):
                standards = domain_context.get("standards", [])
                lines.append(f"  Standards: {', '.join(standards)}")
            if domain_context.get("methodology"):
                lines.append(f"  Methodology: {domain_context['methodology']}")
            lines.append("")
        
        return "\n".join(lines) if lines else "No additional context available."

    @classmethod
    def create_user_prompt(cls,
                          query: str,
                          retrieval_context: Dict,
                          format_type: str = "recommendation") -> str:
        """
        Create domain-specific user prompt with MASSIVE context emphasis.

        Strategy: Make context block so large and prominent that LLM cannot miss it.
        Context-to-instruction ratio: 5:1 (context dominates)
        
        NOTE: This method is used for template-based responses.
        For LLM-based responses with citation pool, use create_retrieval_prompt().

        Args:
            query: Original user query
            retrieval_context: Enhanced retrieval context from agent
            format_type: Type of response format (recommendation, analysis, guidance)

        Returns:
            Formatted prompt with overwhelmingly prominent context
        """
        candidates = retrieval_context.get("candidates", [])
        
        # If no candidates, use no-context prompt
        if not candidates:
            return cls._create_no_context_prompt(query, retrieval_context)
        
        # Build MASSIVE context block (this will be 80% of the prompt)
        context_block = cls._build_massive_context_block(candidates, retrieval_context)
        
        # Get appropriate template (much smaller than context)
        if format_type == "recommendation":
            template = cls._get_recommendation_template()
        elif format_type == "analysis":
            template = cls._get_analysis_template()
        elif format_type == "guidance":
            template = cls._get_guidance_template()
        else:
            template = cls._get_default_template()
        
        # Combine with context DOMINATING the prompt
        formatted_prompt = template.format(
            context=context_block,
            query=query,
            candidate_count=len(candidates)
        )
        
        return formatted_prompt

    @classmethod
    def _build_massive_context_block(cls, candidates: List[Dict], retrieval_context: Dict) -> str:
        """
        Build a MASSIVE, UNMISSABLE context block.
        
        This is the most important method - it must make real citations
        so prominent that the LLM cannot possibly miss them.
        
        Goal: 100+ lines of context vs 20 lines of instructions
        """
        lines = []
        
        # ============================================================
        # SECTION 1: AVAILABLE CITATIONS (MOST CRITICAL)
        # ============================================================
        lines.append("")
        lines.append("=" * 80)
        lines.append("=" * 80)
        lines.append("🔒 [AVAILABLE CITATIONS] - THE ONLY CITATIONS YOU MAY USE")
        lines.append("=" * 80)
        lines.append("=" * 80)
        lines.append("")
        lines.append("⚠️  CRITICAL INSTRUCTION:")
        lines.append("    - These are the ONLY valid citations for this query")
        lines.append("    - You MUST use these exact citation IDs in your response")
        lines.append("    - Do NOT create new citations")
        lines.append("    - Do NOT use examples from your training")
        lines.append("    - Each citation is marked with 'CITATION:' for easy identification")
        lines.append("")
        lines.append("=" * 80)
        lines.append("")
        
        # List all available citations prominently
        for idx, candidate in enumerate(candidates, 1):
            element = candidate.get("element", "Unknown")
            citation = candidate.get("citation", "NO_CITATION_AVAILABLE")
            elem_type = candidate.get("type", "Unknown")
            layer = candidate.get("layer", "")
            definition = candidate.get("definition", "")
            confidence = candidate.get("confidence", 0)
            priority = candidate.get("priority", "")
            source = candidate.get("source", "")
            
            lines.append(f"╔═══════════════════════════════════════════════════════════════════════════╗")
            lines.append(f"║ CITATION #{idx}")
            lines.append(f"╠═══════════════════════════════════════════════════════════════════════════╣")
            lines.append(f"║")
            lines.append(f"║ 📌 CITATION: {citation}")
            lines.append(f"║    👆 USE THIS EXACT ID IN YOUR RESPONSE")
            lines.append(f"║")
            lines.append(f"║ Label: {element}")
            lines.append(f"║ Type: {elem_type}")
            
            if layer:
                lines.append(f"║ Layer: {layer}")
            
            if source:
                lines.append(f"║ Source: {source}")
            
            if priority:
                lines.append(f"║ Priority: {priority}")
                
            lines.append(f"║ Confidence: {confidence:.0%}")
            lines.append(f"║")
            
            if definition:
                # Word wrap definition to 70 chars per line
                def_lines = []
                words = definition.split()
                current_line = "║ Definition: "
                
                for word in words:
                    if len(current_line) + len(word) + 1 <= 78:
                        current_line += word + " "
                    else:
                        def_lines.append(current_line.rstrip())
                        current_line = "║             " + word + " "
                
                if current_line.strip() != "║":
                    def_lines.append(current_line.rstrip())
                
                lines.extend(def_lines)
                lines.append(f"║")
            
            lines.append(f"╚═══════════════════════════════════════════════════════════════════════════╝")
            lines.append("")
        
        lines.append("=" * 80)
        lines.append("END OF AVAILABLE CITATIONS")
        lines.append("=" * 80)
        lines.append("")
        lines.append("🔴 REMINDER: Only use citations listed above!")
        lines.append("🔴 Do NOT invent new citation IDs!")
        lines.append("🔴 Do NOT use example citations!")
        lines.append("")
        lines.append("=" * 80)
        lines.append("")
        
        # ============================================================
        # SECTION 2: TOGAF CONTEXT (If available)
        # ============================================================
        togaf_context = retrieval_context.get("togaf_context", {})
        if togaf_context:
            lines.append("─" * 80)
            lines.append("📋 [TOGAF ARCHITECTURE CONTEXT]")
            lines.append("─" * 80)
            lines.append("")
            
            if togaf_context.get("phase"):
                phase = togaf_context.get("phase", "")
                phase_letter = togaf_context.get("phase_letter", "")
                description = togaf_context.get("description", "")
                
                lines.append(f"  TOGAF Phase: {phase}")
                lines.append(f"  Phase ID: togaf:adm:{phase_letter}")
                lines.append(f"  Description: {description}")
                lines.append("")
            
            deliverables = togaf_context.get("deliverables", [])
            if deliverables:
                lines.append(f"  Expected Deliverables:")
                for deliverable in deliverables:
                    lines.append(f"    - {deliverable}")
                lines.append("")
            
            lines.append("─" * 80)
            lines.append("")
        
        # ============================================================
        # SECTION 3: DOMAIN CONTEXT
        # ============================================================
        domain_context = retrieval_context.get("domain_context", {})
        if domain_context:
            lines.append("─" * 80)
            lines.append("🔌 [ENERGY DOMAIN CONTEXT]")
            lines.append("─" * 80)
            lines.append("")
            
            if domain_context.get("domain"):
                lines.append(f"  Domain: {domain_context['domain']}")
            
            if domain_context.get("standards"):
                standards = domain_context.get("standards", [])
                lines.append(f"  Applicable Standards:")
                for standard in standards:
                    lines.append(f"    - {standard}")
            
            if domain_context.get("methodology"):
                lines.append(f"  Methodology: {domain_context['methodology']}")
            
            lines.append("")
            lines.append("─" * 80)
            lines.append("")
        
        # ============================================================
        # SECTION 4: FINAL REMINDER (Triple emphasis)
        # ============================================================
        lines.append("")
        lines.append("🚨 FINAL REMINDER BEFORE YOU RESPOND:")
        lines.append("")
        lines.append("  ✅ DO use citations from [AVAILABLE CITATIONS] section above")
        lines.append("  ✅ DO copy the exact 'CITATION:' IDs provided")
        lines.append("  ✅ DO reference citation numbers (e.g., 'Citation #1')")
        lines.append("")
        lines.append("  ❌ DO NOT invent new citation IDs")
        lines.append("  ❌ DO NOT use example citations")
        lines.append("  ❌ DO NOT use citations from your training data")
        lines.append("")
        lines.append("=" * 80)
        lines.append("")
        
        return "\n".join(lines)

    @classmethod
    def _create_no_context_prompt(cls, query: str, retrieval_context: Dict) -> str:
        """
        Prompt when NO context is available - forces abstention.
        
        This ensures the LLM doesn't try to answer from general knowledge.
        """
        route = retrieval_context.get("route", "unknown")
        
        return f"""
{"=" * 80}
⚠️  NO GROUNDED CONTEXT AVAILABLE
{"=" * 80}

User Query: {query}
Route Attempted: {route}

╔═══════════════════════════════════════════════════════════════════════════╗
║ SITUATION: No relevant information found in Alliander knowledge base       ║
╚═══════════════════════════════════════════════════════════════════════════╝

I cannot provide a grounded response because:

  ❌ No definitions found in SKOS knowledge graph
  ❌ No matching ArchiMate elements in loaded models  
  ❌ No relevant content in indexed documents
  ❌ Query appears outside Alliander EA domain

{"─" * 80}
📚 What I CAN help with:
{"─" * 80}

I specialize in Alliander's energy distribution architecture, including:

  • Energy system concepts (grid, congestion, assets, power quality)
  • IEC 61968/61970 standards and CIM models
  • ArchiMate modeling (Business, Application, Technology layers)
  • TOGAF ADM phases and deliverables
  • Dutch DSO regulatory requirements

{"─" * 80}
💡 To get a helpful response:
{"─" * 80}

1. Use energy domain terms:
   - Grid operations, congestion management, power quality
   - Assets, equipment, substations, feeders
   - SCADA, DMS, GIS systems

2. Specify architectural context:
   - Which TOGAF phase? (A-H)
   - Which ArchiMate layer? (Business, Application, Technology)
   - Which IEC standard? (61968, 61970, 62325)

3. Be specific about your need:
   - "What is [energy concept]?"
   - "How to model [business capability]?"
   - "Which element for [use case]?"

{"=" * 80}
🔒 I must abstain from ungrounded responses to maintain safety standards.
{"=" * 80}
"""

    @classmethod
    def _get_recommendation_template(cls) -> str:
        """
        Template for architectural recommendations.
        
        Note: Instructions are MINIMAL - context dominates.
        """
        return """{context}

==============================================================================
USER QUERY
==============================================================================

{query}

==============================================================================
YOUR TASK: Provide an Architectural Recommendation
==============================================================================

Based on the {candidate_count} citations provided above, give a specific recommendation.

Your response must include:

1. **Recommended Element(s)**
   - State which citation(s) to use
   - Reference by citation number and ID (e.g., "Citation #1: skos:ServiceProvider")
   
2. **TOGAF Alignment**
   - Explain which TOGAF phase this aligns with
   - Why this choice fits the architecture methodology
   
3. **Energy Domain Considerations**
   - IEC standard compliance
   - Grid operations impact
   - Dutch DSO regulatory context
   
4. **Implementation Guidance**
   - Practical next steps
   - Alliander-specific considerations

⚠️  CRITICAL: Use ONLY the citations from [AVAILABLE CITATIONS] section above.

Format your response clearly with proper citation references.
"""

    @classmethod
    def _get_analysis_template(cls) -> str:
        """
        Template for architectural analysis.
        
        Note: Instructions are MINIMAL - context dominates.
        """
        return """{context}

{"=" * 80}
👤 USER QUERY
{"=" * 80}

{query}

{"=" * 80}
🔍 YOUR TASK: Provide an Architectural Analysis
{"=" * 80}

Analyze this query using the {candidate_count} citations provided above.

Your analysis should cover:

1. **Relevant Architectural Patterns**
   - Identify which citations are most relevant
   - Reference by citation number and ID
   
2. **TOGAF Phase Assessment**
   - Which ADM phase(s) are involved?
   - What are the implications?
   
3. **IEC Standard Compliance**
   - Which IEC standards apply?
   - Compliance considerations
   
4. **Alliander Context Fit**
   - How does this fit Alliander's architecture?
   - Energy distribution specific concerns

⚠️  CRITICAL: Use ONLY the citations from [AVAILABLE CITATIONS] section above.

Provide structured analysis with proper citations.
"""

    @classmethod
    def _get_guidance_template(cls) -> str:
        """
        Template for implementation guidance.
        
        Note: Instructions are MINIMAL - context dominates.
        """
        return """{context}

{"=" * 80}
👤 USER QUERY
{"=" * 80}

{query}

{"=" * 80}
🛠 YOUR TASK: Provide Implementation Guidance
{"=" * 80}

Using the {candidate_count} citations above, provide actionable guidance.

Your guidance should include:

1. **Specific Elements to Use**
   - Reference exact citations from above
   - Include citation numbers and IDs
   
2. **TOGAF-Compliant Steps**
   - Follow ADM methodology
   - Phase-appropriate deliverables
   
3. **Technical Requirements**
   - IEC standard compliance
   - Energy domain specifics
   
4. **Alliander Operational Constraints**
   - Dutch DSO regulations
   - Existing system integration
   - ArchiMate model updates

⚠️  CRITICAL: Use ONLY the citations from [AVAILABLE CITATIONS] section above.

Structure as clear, actionable steps with citations.
"""

    @classmethod
    def _get_default_template(cls) -> str:
        """
        Default template for general queries.
        
        Note: Instructions are MINIMAL - context dominates.
        """
        return """{context}

{"=" * 80}
👤 USER QUERY
{"=" * 80}

{query}

{"=" * 80}
🔎 YOUR TASK: Provide a Comprehensive Response
{"=" * 80}

Based on the {candidate_count} citations provided, address this query.

Ensure your response:

1. Uses EXACT citations from [AVAILABLE CITATIONS] section
2. References citations by number and ID (e.g., "Citation #1: skos:ServiceProvider")
3. Follows TOGAF architectural principles
4. Considers energy domain requirements
5. Provides practical Alliander-specific guidance

⚠️  CRITICAL: Use ONLY the citations from [AVAILABLE CITATIONS] section above.

If the available citations are insufficient to fully answer the query, 
state this explicitly and explain what additional information would be needed.
"""

    @classmethod
    def create_fallback_prompt(cls, query: str, route: str) -> str:
        """
        Create fallback prompt when no context is available.
        
        This is essentially the same as _create_no_context_prompt,
        provided for backward compatibility.

        Args:
            query: Original query
            route: Route that was taken

        Returns:
            Fallback prompt that explains limitations
        """
        return cls._create_no_context_prompt(query, {"route": route})

    @classmethod
    def validate_response_format(cls, response: str) -> Dict[str, Any]:
        """
        Validate that response follows required format and has real citations.

        Args:
            response: Generated response text

        Returns:
            Validation results with issues and suggestions
        """
        validation = {
            "has_citations": False,
            "citation_types": [],
            "citation_count": 0,
            "issues": [],
            "suggestions": [],
            "fake_citations_detected": []
        }

        # Check for real citation patterns
        citation_patterns = [
            ("archi:id-", "ArchiMate"),
            ("iec:", "IEC Standard"),
            ("togaf:adm:", "TOGAF ADM"),
            ("skos:", "SKOS Vocabulary"),
            ("entsoe:", "ENTSOE"),
            ("lido:", "LIDO"),
            ("eurlex:", "EUR-LEX"),
            ("doc:", "Document"),
            ("external:", "External Source")
        ]

        # CRITICAL: Check for KNOWN FAKE citations
        # These should NEVER appear in any response
        fake_citation_patterns = [
            "archi:id-cap-001",
            "iec:GridCongestion", 
            "iec:61968",
            "iec:61970",
        ]
        
        for fake in fake_citation_patterns:
            if fake in response:
                validation["fake_citations_detected"].append(fake)
                validation["issues"].append(f"🚨 FAKE CITATION DETECTED: {fake}")
                validation["suggestions"].append(
                    f"Remove fake citation {fake}. This citation was likely "
                    f"hallucinated by the LLM. Only use citations provided in context."
                )

        # Count real citations
        for pattern, citation_type in citation_patterns:
            if pattern in response:
                validation["has_citations"] = True
                validation["citation_types"].append(citation_type)
                count = response.count(pattern)
                validation["citation_count"] += count

        # Validation checks
        if validation["fake_citations_detected"]:
            validation["issues"].append(
                f"❌ CRITICAL FAILURE: Found {len(validation['fake_citations_detected'])} "
                f"fake citation(s). Response must be regenerated with real citations only."
            )
        
        if not validation["has_citations"]:
            validation["issues"].append("No citations found in response")
            validation["suggestions"].append(
                "Add citations from the [AVAILABLE CITATIONS] section provided in context"
            )
        
        if validation["citation_count"] < 1 and not validation["fake_citations_detected"]:
            validation["issues"].append("Insufficient citations (minimum 1 required)")
            validation["suggestions"].append(
                "Every claim should reference at least one citation from the provided context"
            )

        # Check for energy domain terminology
        energy_terms = [
            "grid", "energy", "distribution", "scada", "congestion", 
            "capability", "power", "reactive", "asset", "substation"
        ]
        has_energy_context = any(term in response.lower() for term in energy_terms)

        if not has_energy_context and "cannot provide" not in response.lower():
            validation["suggestions"].append(
                "Consider adding energy domain context for Alliander EA responses"
            )

        # Overall verdict
        if validation["fake_citations_detected"]:
            validation["verdict"] = "FAILED - Fake citations detected"
        elif not validation["has_citations"]:
            validation["verdict"] = "FAILED - No citations"
        elif validation["citation_count"] < 1:
            validation["verdict"] = "FAILED - Insufficient citations"
        else:
            validation["verdict"] = "PASSED"

        return validation