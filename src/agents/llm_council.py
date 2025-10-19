"""
LLM Council - Dual LLM validation system for response quality assurance.

This module implements a two-LLM validation pattern where:
1. Primary LLM generates responses
2. Validator LLM checks for hallucinations and quality
3. Reconciliation process when disagreements occur

Designed for use with OpenAI GPT-5 + Groq Llama-3.3 for optimal quality.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from src.llm.base import LLMConfig

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Status of LLM validation."""
    VALID = "valid"
    HALLUCINATION_DETECTED = "hallucination_detected"
    OFF_TOPIC = "off_topic"
    INSUFFICIENT_GROUNDING = "insufficient_grounding"
    NEEDS_RECONCILIATION = "needs_reconciliation"


@dataclass
class ValidationResult:
    """Result of LLM validation."""
    status: ValidationStatus
    confidence: float
    issues: List[str]
    suggestions: List[str]
    metadata: Dict[str, Any]


@dataclass
class CouncilResponse:
    """Final response from LLM Council."""
    content: str
    primary_response: str
    validation: ValidationResult
    reconciled: bool
    total_tokens: int
    response_time_ms: float


class LLMCouncil:
    """
    LLM Council for dual LLM validation.
    
    Primary: OpenAI GPT-5 (intelligence, accuracy)
    Validator: Groq Llama-3.3-70b (fast validation)
    """

    def __init__(
        self, 
        primary_api_key: str, 
        validator_api_key: str, 
        use_openai_primary: bool = True,
        primary_model: str = "gpt-5",
        validator_model: str = "llama-3.3-70b-versatile",
        reconciliation_rounds: int = 1
    ):
        """Initialize LLM Council with dual LLM validation."""
        self.reconciliation_rounds = reconciliation_rounds
        
        # Primary LLM (OpenAI GPT-5)
        if use_openai_primary:
            from src.llm.openai_provider import OpenAIProvider
            self.primary_llm = OpenAIProvider(LLMConfig(
                api_key=primary_api_key,
                model=primary_model,
                provider="openai",
                max_tokens=4096,
                temperature=0.3,
                timeout=60
            ))
            self.backend = "openai_groq"
            logger.info(f"✅ Primary LLM = OpenAI {primary_model}")
        else:
            from src.llm.groq_provider import GroqProvider
            self.primary_llm = GroqProvider(LLMConfig(
                api_key=primary_api_key,
                model=primary_model,
                provider="groq",
                max_tokens=4096,
                temperature=0.3,
                timeout=60
            ))
            self.backend = "groq"
            logger.info(f"✅ Primary LLM = Groq {primary_model}")
        
        # Validator LLM (Groq Llama 3.3)
        from src.llm.groq_provider import GroqProvider
        self.validator_llm = GroqProvider(LLMConfig(
            api_key=validator_api_key,
            model=validator_model,
            provider="groq",
            max_tokens=4096,
            temperature=0.2,
            timeout=60
        ))
        logger.info(f"✅ Validator LLM = Groq {validator_model}")
        
        self.primary_model = primary_model
        self.validator_model = validator_model
    
    async def get_validated_response(
        self,
        query: str,
        context: Dict,
        citation_pool: List[Dict],
        temperature: float = 0.3
    ) -> CouncilResponse:
        """
        Get validated response through dual LLM process.
        
        Args:
            query: User query
            context: Retrieval context with candidates
            citation_pool: Valid citations that can be used
            temperature: Generation temperature
            
        Returns:
            CouncilResponse with validated content
        """
        start_time = time.perf_counter()
        
        # Step 1: Generate primary response
        primary_response, primary_tokens = await self._generate_primary_response(
            query, context, citation_pool, temperature
        )
        
        # Step 2: Validate response
        validation = await self._validate_response(
            query, context, primary_response, citation_pool
        )
        
        # Step 3: Decide if reconciliation needed
        final_response = primary_response
        reconciled = False
        
        if validation.confidence < 0.5 or validation.status == ValidationStatus.HALLUCINATION_DETECTED:
            logger.info(f"Validation failed (confidence: {validation.confidence}), reconciling...")
            
            # Step 4: Reconciliation process
            final_response = await self._reconcile_responses(
                query, context, primary_response, validation, citation_pool
            )
            reconciled = True
        
        response_time = (time.perf_counter() - start_time) * 1000
        
        return CouncilResponse(
            content=final_response,
            primary_response=primary_response,
            validation=validation,
            reconciled=reconciled,
            total_tokens=primary_tokens,
            response_time_ms=response_time
        )
    
    # ← FIXED: Added missing method
    async def _generate_primary_response(
        self,
        query: str,
        context: Dict,
        citation_pool: List[Dict],
        temperature: float
    ) -> Tuple[str, int]:
        """
        Generate primary response with citations.
        
        Returns:
            Tuple of (response_text, token_count)
        """
        # Build context summary
        context_summary = self._build_context_summary(context)
        
        # Build citation instruction
        citation_instruction = self._build_citation_instruction(citation_pool)
        
        system_prompt = """You are an Enterprise Architecture expert at Alliander, a Dutch DSO.
Your expertise covers TOGAF methodology, ArchiMate modeling, and IEC energy standards.
You MUST ground all responses in the provided context and citations."""
        
        user_prompt = f"""Query: {query}

Context:
{context_summary}

Available Citations (USE ONLY THESE):
{citation_instruction}

Instructions:
1. Answer the query using ONLY information from the context
2. Include relevant citations in [brackets] format
3. Be specific and technical where appropriate
4. Acknowledge if information is incomplete

Response:"""
        
        try:
            # Use the LLMProvider's generate method
            response = await self.primary_llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=1024
            )
            
            return response.content, response.tokens_used
                
        except Exception as e:
            logger.error(f"Primary LLM generation failed: {e}")
            raise
    
    async def _validate_response(
        self,
        query: str,
        context: Dict,
        response: str,
        citation_pool: List[Dict]
    ) -> ValidationResult:
        """
        Validate response for hallucinations and quality.
        
        Returns:
            ValidationResult with status and confidence
        """
        # Extract citations used in response
        used_citations = self._extract_citations_from_response(response)
        valid_citations = {c['citation'] for c in citation_pool}
        
        validation_prompt = f"""As a validation expert, analyze this response for accuracy and grounding.

Original Query: {query}

Response to Validate:
{response}

Context Provided (summary):
- {len(context.get('candidates', []))} candidates found
- Valid citations available: {len(valid_citations)}
- Citations used in response: {used_citations}

Validation Criteria:
1. Does the response use ONLY information from the provided context?
2. Are all citations valid and from the approved pool?
3. Does it answer the actual question asked?
4. Is it technically accurate for energy systems architecture?

Respond in JSON format:
{{
    "is_valid": boolean,
    "confidence": float (0-1),
    "issues": ["list of specific issues found"],
    "suggestions": ["list of improvement suggestions"],
    "hallucination_detected": boolean,
    "off_topic": boolean
}}"""
        
        try:
            # Use validator LLM's generate method
            response_obj = await self.validator_llm.generate(
                prompt=validation_prompt,
                system_prompt="You are a strict validation expert. Be critical and thorough.",
                temperature=0.1,
                max_tokens=512
            )
            
            # Parse JSON from response
            try:
                # Strip markdown code blocks if present
                content = response_obj.content.strip()
                
                # Remove markdown JSON code blocks
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                
                content = content.strip()
                
                # Parse JSON
                validation_data = json.loads(content)
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse validation JSON: {e}")
                logger.error(f"Raw content: {response_obj.content[:200]}")
                
                # Return safe fallback
                return ValidationResult(
                    status=ValidationStatus.NEEDS_RECONCILIATION,
                    confidence=0.3,
                    issues=["Validation response was not valid JSON"],
                    suggestions=["Manual review recommended"],
                    metadata={"parse_error": str(e)}
                )
            
            # Determine status
            if validation_data.get("hallucination_detected"):
                status = ValidationStatus.HALLUCINATION_DETECTED
            elif validation_data.get("off_topic"):
                status = ValidationStatus.OFF_TOPIC
            elif validation_data.get("confidence", 0) < 0.5:
                status = ValidationStatus.INSUFFICIENT_GROUNDING
            elif validation_data.get("is_valid"):
                status = ValidationStatus.VALID
            else:
                status = ValidationStatus.NEEDS_RECONCILIATION
            
            return ValidationResult(
                status=status,
                confidence=validation_data.get("confidence", 0),
                issues=validation_data.get("issues", []),
                suggestions=validation_data.get("suggestions", []),
                metadata=validation_data
            )
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return ValidationResult(
                status=ValidationStatus.NEEDS_RECONCILIATION,
                confidence=0.3,
                issues=["Validation process failed"],
                suggestions=["Manual review recommended"],
                metadata={"error": str(e)}
            )
    
    async def _reconcile_responses(
        self,
        query: str,
        context: Dict,
        primary_response: str,
        validation: ValidationResult,
        citation_pool: List[Dict]
    ) -> str:
        """
        Reconcile disagreements between primary and validator.
        
        Returns:
            Reconciled response text
        """
        context_summary = self._build_context_summary(context)
        reconciliation_prompt = f"""The original response needs improvement. Create a better version.

        Query: {query}

        Original Response (with issues):
        {primary_response}

        Problems Found:
        {chr(10).join(f"- {issue}" for issue in validation.issues)}

        Improvements Needed:
        {chr(10).join(f"- {suggestion}" for suggestion in validation.suggestions)}

        Context Available:
        {context_summary}

        Valid Citations You Can Use:
        {self._build_citation_instruction(citation_pool[:10])}

        IMPORTANT INSTRUCTIONS:
        1. Keep the good parts of the original response
        2. Fix only the specific issues mentioned
        3. Use ONLY citations from the approved list above
        4. Format citations as [citation_id]
        5. Maintain technical accuracy
        6. Answer the original query completely

        Provide ONLY the improved response text below (no meta-commentary):
        """
        
        if len(primary_response.strip()) > 100 and validation.confidence > 0.3:
            logger.info("Primary response acceptable despite validation issues, using it")
            return primary_response
        
        try:
            response = await self.primary_llm.generate(
                prompt=reconciliation_prompt,
                system_prompt="You must create an accurate, well-grounded response.",
                temperature=0.2,
                max_tokens=1024
            )
            
            reconciled = response.content
            
            # ✅ ADD: Debug logging
            logger.info(f"🔄 Reconciliation response length: {len(reconciled)}")
            logger.info(f"🔄 Reconciliation preview: {reconciled[:200]}")
            
            # ✅ ADD: Safety check for empty responses
            if not reconciled or len(reconciled.strip()) < 50:
                logger.error("❌ Reconciliation produced empty/short response, using primary")
                return primary_response
            
            # Quick re-validation (optional)
            if self.reconciliation_rounds > 1:
                revalidation = await self._validate_response(
                    query, context, reconciled, citation_pool
                )
                
                if revalidation.confidence > validation.confidence:
                    logger.info(f"Reconciliation improved confidence: {validation.confidence:.2f} -> {revalidation.confidence:.2f}")
                    return reconciled
                else:
                    logger.warning("Reconciliation did not improve confidence, using original")
                    return primary_response
            
            return reconciled
            
        except Exception as e:
            logger.error(f"Reconciliation failed: {e}, using primary response")
            return primary_response
    
    def _build_context_summary(self, context: Dict) -> str:
        """Build concise context summary for LLM."""
        lines = []
        
        # Add candidates
        candidates = context.get("candidates", [])[:5]
        if candidates:
            lines.append("Top Candidates:")
            for i, c in enumerate(candidates, 1):
                element = c.get("element", "Unknown")
                citation = c.get("citation", "unknown")
                confidence = c.get("confidence", 0)
                lines.append(f"{i}. {element} [{citation}] (confidence: {confidence:.2f})")
        
        return "\n".join(lines)
    
    def _build_citation_instruction(self, citation_pool: List[Dict]) -> str:
        """Build citation instruction for LLM."""
        lines = []
        
        for citation in citation_pool[:20]:
            cit_id = citation.get("citation", "unknown")
            label = citation.get("label", "")
            definition = citation.get("definition", "")
            
            if definition and len(definition) > 100:
                definition = definition[:100] + "..."
            
            lines.append(f"[{cit_id}] - {label}")
            if definition:
                lines.append(f"  Definition: {definition}")
        
        return "\n".join(lines)
    
    def _extract_citations_from_response(self, response: str) -> List[str]:
        """Extract citations from response text."""
        import re
        
        # Pattern to match [citation] format
        pattern = r'\[([^\]]+)\]'
        matches = re.findall(pattern, response)
        
        # Filter out non-citation brackets
        citations = []
        for match in matches:
            # Check if it looks like a citation
            if any(match.startswith(prefix) for prefix in ["skos:", "iec:", "archi:", "togaf:", "doc:", "eurlex:", "entsoe:"]):
                citations.append(match)
        
        return citations
    
    async def evaluate_response_pair(
        self,
        query: str,
        response1: str,
        response2: str,
        context: Dict
    ) -> Dict[str, Any]:
        """Compare and evaluate two responses (for A/B testing)."""
        eval_prompt = f"""Compare these two responses for quality and accuracy.

Query: {query}

Response A:
{response1}

Response B:
{response2}

Context available: {len(context.get('candidates', []))} candidates

Evaluate based on:
1. Factual accuracy
2. Proper use of citations
3. Completeness of answer
4. Technical correctness
5. Clarity and structure

Return JSON:
{{
    "preferred": "A" or "B",
    "confidence": float (0-1),
    "response_a_score": float (0-1),
    "response_b_score": float (0-1),
    "reasoning": "explanation"
}}"""
        
        try:
            response = await self.validator_llm.generate(
                prompt=eval_prompt,
                system_prompt="You are an expert evaluator.",
                temperature=0.1,
                max_tokens=512
            )
            
            return json.loads(response.content)
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {
                "preferred": "unknown",
                "confidence": 0,
                "error": str(e)
            }
    
    async def close(self):
        """Close connections for both LLMs."""
        if hasattr(self.primary_llm, 'close'):
            await self.primary_llm.close()
        if hasattr(self.validator_llm, 'close'):
            await self.validator_llm.close()