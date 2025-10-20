"""
ADR (Architectural Decision Records) Indexer

Indexes markdown-based ADRs for semantic search and retrieval.
Handles YAML frontmatter format used by Jekyll/Just the Docs.

Citation format: adr:0025 or adr:unify-demand-response-interfaces
"""

import logging
import re
import yaml
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ADR:
    """Represents an Architectural Decision Record."""
    number: str          # e.g., "0025"
    title: str           # e.g., "Provide a unified Demand/Response product interface..."
    status: str          # e.g., "proposed", "accepted", "deprecated"
    date: Optional[str]  # e.g., "2024-03-15"
    driver: Optional[str]  # e.g., "Robert-Jan Peters"
    contributors: Optional[str]  # Contributors list
    context: str         # Context and Problem Statement section
    decision: str        # Decision Outcome section
    consequences: str    # Consequences section
    options: str         # Considered Options section
    more_info: str       # More Information section
    full_text: str       # Complete markdown content (without frontmatter)
    file_path: Path      # Path to .md file
    slug: str            # URL-friendly slug
    
    def get_citation_id(self) -> str:
        """Get citation ID for this ADR."""
        return f"adr:{self.number}"
    
    def get_searchable_text(self) -> str:
        """Get combined text for semantic search."""
        parts = [
            self.title,
            self.context,
            self.decision,
            self.consequences,
            self.options,
            self.more_info
        ]
        return "\n\n".join(p for p in parts if p)


class ADRIndexer:
    """
    Indexes Architectural Decision Records for retrieval.
    
    Supports markdown-based ADRs with YAML frontmatter:
    ---
    title: ADR Title
    status: proposed | accepted | deprecated
    date: YYYY-MM-DD
    driver: Person Name
    ---
    
    # Title
    ## Context and Problem Statement
    ## Decision Outcome
    ## Consequences
    """
    
    def __init__(self, adrs_dir: str = "data/adrs/"):
        """
        Initialize ADR indexer.
        
        Args:
            adrs_dir: Directory containing ADR markdown files
        """
        self.adrs_dir = Path(adrs_dir)
        self.adrs: List[ADR] = []
        self.adr_by_number: Dict[str, ADR] = {}
        self.adr_by_slug: Dict[str, ADR] = {}
        
        logger.info(f"Successfully loaded {len(self.adrs)} ADRs")
        for adr in self.adrs:
            logger.info(f"  - ADR-{adr.number}: {adr.title} (status: {adr.status})")
        
        logger.info(f"ADRIndexer initialized for directory: {adrs_dir}")
    
    def load_adrs(self) -> int:
        """
        Load all ADRs from directory.
        
        Returns:
            Number of ADRs loaded
        """
        if not self.adrs_dir.exists():
            logger.warning(f"ADR directory not found: {self.adrs_dir}")
            return 0
        
        # Find all .md files except index.md and template
        md_files = [
            f for f in self.adrs_dir.glob("*.md")
            if f.name not in ["index.md", "adr-template.md"]
        ]
        
        logger.info(f"Found {len(md_files)} ADR files")
        
        for md_file in md_files:
            try:
                adr = self._parse_adr_file(md_file)
                if adr:
                    self.adrs.append(adr)
                    self.adr_by_number[adr.number] = adr
                    self.adr_by_slug[adr.slug] = adr
                    logger.debug(f"Loaded ADR-{adr.number}: {adr.title[:50]}...")
            except Exception as e:
                logger.error(f"Failed to parse ADR {md_file.name}: {e}")
        
        logger.info(f"Successfully loaded {len(self.adrs)} ADRs")
        return len(self.adrs)
    
    def _parse_adr_file(self, file_path: Path) -> Optional[ADR]:
        """
        Parse ADR markdown file with YAML frontmatter.
        
        Expected format:
        ---
        title: ADR Title
        status: proposed
        date: 2024-03-15
        driver: Person Name <email@domain.com>
        ---
        
        # Main Title
        ## Context and Problem Statement
        ...
        ## Decision Outcome
        ...
        """
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Extract number from filename (e.g., 0025-title.md -> 0025)
            number_match = re.match(r'(\d{4})', file_path.stem)
            if not number_match:
                logger.warning(f"Could not extract number from {file_path.name}")
                return None
            
            number = number_match.group(1)
            
            # Parse YAML frontmatter
            frontmatter = {}
            markdown_content = content
            
            if content.startswith('---'):
                # Split frontmatter and content
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    try:
                        frontmatter = yaml.safe_load(parts[1]) or {}
                        markdown_content = parts[2].strip()
                    except yaml.YAMLError as e:
                        logger.warning(f"Could not parse YAML frontmatter in {file_path.name}: {e}")
            
            # Extract title (from frontmatter or first heading)
            title = frontmatter.get('title', '')
            if not title:
                title_match = re.search(r'^#\s+(.+)$', markdown_content, re.MULTILINE)
                title = title_match.group(1) if title_match else file_path.stem
            
            # Extract metadata from frontmatter
            status = frontmatter.get('status', 'unknown')
            # Handle status with multiple options (e.g., "proposed | rejected | accepted")
            if isinstance(status, str) and '|' in status:
                status = status.split('|')[0].strip()
            
            date = frontmatter.get('date')
            driver = frontmatter.get('driver')
            contributors = frontmatter.get('contributors')
            
            # Extract sections from markdown
            context = self._extract_section(markdown_content, "Context and Problem Statement")
            decision = self._extract_section(markdown_content, "Decision Outcome")
            consequences = self._extract_section(markdown_content, "Consequences")
            options = self._extract_section(markdown_content, "Considered Options")
            more_info = self._extract_section(markdown_content, "More Information")
            
            # Create slug (URL-friendly)
            slug = re.sub(r'[^\w\s-]', '', title.lower())
            slug = re.sub(r'[-\s]+', '-', slug)[:60]  # Limit length
            
            return ADR(
                number=number,
                title=title,
                status=status,
                date=date,
                driver=driver,
                contributors=contributors,
                context=context,
                decision=decision,
                consequences=consequences,
                options=options,
                more_info=more_info,
                full_text=markdown_content,
                file_path=file_path,
                slug=slug
            )
            
        except Exception as e:
            logger.error(f"Error parsing {file_path.name}: {e}")
            return None
    
    def _extract_section(self, content: str, section_name: str) -> str:
        """
        Extract content between section headers.
        
        Matches ## Section until next ## or end of document.
        Handles nested ### subsections correctly.
        """
        # Match ## Section until next ## (same level) or end
        pattern = rf'##\s+{re.escape(section_name)}\s*\n(.*?)(?=\n##\s+[^#]|\Z)'
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        
        if match:
            section_content = match.group(1).strip()
            # Remove markdown comments and excessive whitespace
            section_content = re.sub(r'<!--.*?-->', '', section_content, flags=re.DOTALL)
            section_content = re.sub(r'\n{3,}', '\n\n', section_content)
            return section_content.strip()
        
        return ""
    
    def search_adrs(self, query_terms: List[str], max_results: int = 5) -> List[Dict]:
        """
        Search ADRs by query terms with PRIORITY for direct number matches.
        
        Args:
            query_terms: List of search terms
            max_results: Maximum results to return
            
        Returns:
            List of matching ADR dictionaries with scores
        """
        if not self.adrs:
            return []
        
        # ✅ PRIORITY 1: Extract ADR numbers from query
        adr_numbers = []
        import re
        
        # Combine all query terms into one string for easier pattern matching
        query_string = ' '.join(query_terms).lower()
        
        # Pattern 1: "adr-0025", "adr 0025", "adr:0025"
        matches = re.findall(r'adr[:\s-]?(\d{4})', query_string)
        adr_numbers.extend(matches)
        
        # Pattern 2: Just "0025" by itself
        standalone_numbers = re.findall(r'\b(\d{4})\b', query_string)
        adr_numbers.extend(standalone_numbers)
        
        # Pattern 3: "adr 25" (without leading zeros)
        short_numbers = re.findall(r'adr[:\s-]?(\d{1,3})\b', query_string)
        # Pad with zeros
        adr_numbers.extend([n.zfill(4) for n in short_numbers])
        
        # Remove duplicates
        adr_numbers = list(set(adr_numbers))
        
        logger.info(f"🔍 ADR search: extracted numbers {adr_numbers} from query: {query_string}")
        
        results = []
        
        # ✅ DIRECT NUMBER MATCH (Highest Priority)
        if adr_numbers:
            for adr in self.adrs:
                if adr.number in adr_numbers:
                    logger.info(f"✅ Direct ADR number match: {adr.number}")
                    results.append({
                        "adr_number": adr.number,
                        "title": adr.title,
                        "status": adr.status,
                        "citation": adr.get_citation_id(),
                        "citation_id": adr.get_citation_id(),
                        "decision": adr.decision,
                        "context": adr.context,
                        "consequences": adr.consequences,
                        "options": adr.options,
                        "more_info": adr.more_info,
                        "driver": adr.driver or "Unknown",
                        "score": 1000,  # Highest priority
                        "confidence": 0.99,
                        "type": "Architectural Decision Record",
                        "source": "ADR"
                    })
            
            # If we found direct matches, return them immediately
            if results:
                logger.info(f"Returning {len(results)} direct ADR matches")
                return results[:max_results]
        
        # ✅ FALLBACK: Content-based search (only if no direct match)
        logger.info("No direct ADR number match, trying content search...")
        
        for adr in self.adrs:
            # Calculate relevance score
            score = 0
            searchable = adr.get_searchable_text().lower()
            
            # Title match (highest weight)
            for term in query_terms:
                term_lower = term.lower()
                if term_lower in adr.title.lower():
                    score += 10
                if term_lower in adr.decision.lower():
                    score += 5
                if term_lower in searchable:
                    score += searchable.count(term_lower)
            
            if score > 0:
                results.append({
                    "adr_number": adr.number,
                    "title": adr.title,
                    "status": adr.status,
                    "citation": adr.get_citation_id(),
                    "citation_id": adr.get_citation_id(),
                    "decision": adr.decision,
                    "context": adr.context,
                    "consequences": adr.consequences,
                    "options": adr.options,
                    "more_info": adr.more_info,
                    "driver": adr.driver or "Unknown",
                    "score": score,
                    "confidence": min(0.9, score / 20.0),
                    "type": "Architectural Decision Record",
                    "source": "ADR"
                })
        
        # Sort by score (highest first)
        results.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"Content search found {len(results)} ADRs")
        
        return results[:max_results]
    
    def get_adr_by_number(self, number: str) -> Optional[ADR]:
        """Get ADR by number (e.g., '0025')."""
        return self.adr_by_number.get(number)
    
    def get_adr_by_citation(self, citation: str) -> Optional[ADR]:
        """Get ADR by citation (e.g., 'adr:0025')."""
        if citation.startswith("adr:"):
            number = citation.replace("adr:", "")
            return self.adr_by_number.get(number)
        return None
    
    def get_all_citations(self) -> List[str]:
        """Get all valid ADR citations."""
        return [adr.get_citation_id() for adr in self.adrs]
    
    def validate_citation(self, citation: str) -> bool:
        """Check if citation is a valid ADR."""
        return self.get_adr_by_citation(citation) is not None
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        status_counts = {}
        for adr in self.adrs:
            status = adr.status.lower()
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_adrs": len(self.adrs),
            "by_status": status_counts,
            "directory": str(self.adrs_dir)
        }