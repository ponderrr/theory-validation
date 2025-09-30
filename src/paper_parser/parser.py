"""
Main paper parsing logic
"""
import json
from pathlib import Path
from datetime import datetime
from typing import List
from .models import (
    ParsedPaper, PaperMetadata, ProblemFormulation,
    Algorithm, TheoreticalClaim, ExperimentalSetup, PaperSource
)
from .llm_extractor import LLMExtractor

class PaperParser:
    """Parse research papers into structured format"""

    def __init__(self):
        self.extractor = LLMExtractor()

    def parse_paper(self, paper_path: Path, source: PaperSource) -> ParsedPaper:
        """Parse a single paper"""

        # Read paper with encoding error handling
        try:
            text = paper_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            # Try with error handling
            text = paper_path.read_text(encoding='utf-8', errors='ignore')

        print(f"\nParsing {paper_path.name}...")

        # Extract metadata
        metadata = self._extract_metadata(text, source)
        print("  ✓ Extracted metadata")

        # Extract problem formulation
        problem = self._extract_problem(text)
        print("  ✓ Extracted problem formulation")

        # Extract algorithms
        algorithms = self._extract_algorithms(text)
        print(f"  ✓ Found {len(algorithms)} algorithms")

        # Extract claims
        claims = self._extract_claims(text, algorithms)
        print(f"  ✓ Found {len(claims)} theoretical claims")

        # Extract experiments
        experiments = self._extract_experiments(text)
        print("  ✓ Extracted experimental setup")

        return ParsedPaper(
            metadata=metadata,
            problem=problem,
            algorithms=algorithms,
            claims=claims,
            experiments=experiments,
            raw_sections=self._split_sections(text)
        )

    def _extract_metadata(self, text: str, source: PaperSource) -> PaperMetadata:
        """Extract paper metadata"""

        # Try to extract title from first heading
        lines = text.split('\n')
        title = "Untitled"
        for line in lines:
            if line.startswith('#'):
                title = line.lstrip('#').strip()
                break

        return PaperMetadata(
            source=source,
            title=title,
            date_parsed=datetime.now().isoformat(),
            word_count=len(text.split())
        )

    def _extract_problem(self, text: str) -> ProblemFormulation:
        """Extract problem formulation using LLM or fallback"""

        prompt = """Extract the problem formulation from this research paper.

Return JSON with:
{
  "problem_statement": "concise problem description",
  "input_description": "what the system takes as input",
  "output_description": "what the system produces",
  "constraints": ["constraint1", "constraint2"],
  "success_metrics": ["metric1", "metric2"]
}

Focus on the product matching problem: matching products at 4 hierarchical levels (L0-L3) without universal IDs."""

        try:
            data = self.extractor.extract_json(text[:8000], prompt)  # First 8k chars for context
            return ProblemFormulation(**data)
        except Exception as e:
            print(f"LLM extraction failed for problem, using fallback: {e}")
            # Use fallback extraction
            fallback_data = self.extractor._fallback_extraction(text)
            # Extract problem-specific data from fallback
            problem_data = {
                "problem_statement": fallback_data.get("problem_statement", "Product matching problem"),
                "input_description": fallback_data.get("input_description", "Product records"),
                "output_description": fallback_data.get("output_description", "Matched pairs"),
                "constraints": fallback_data.get("constraints", []),
                "success_metrics": fallback_data.get("success_metrics", [])
            }
            return ProblemFormulation(**problem_data)

    def _extract_algorithms(self, text: str) -> List[Algorithm]:
        """Extract all algorithms mentioned"""

        prompt = """Extract ALL algorithms from this paper on hierarchical product matching.

Expected algorithms:
1. Constrained Sinkhorn (Optimal Transport with brand penalties)
2. MDL Edit Distance (Minimum Description Length similarity)
3. Multi-Pass Blocking (LSH/MinHash for candidate generation)
4. Nested Clustering (Hierarchical matching across L0-L3)

Return JSON array:
[
  {
    "name": "Algorithm Name",
    "description": "what it does",
    "pseudocode": "step-by-step algorithm (if present)",
    "input_parameters": {"param1": "description", "param2": "description"},
    "output_format": "what it returns",
    "time_complexity": "Big-O notation",
    "space_complexity": "Big-O notation",
    "key_properties": ["property1", "property2"]
  }
]"""

        try:
            data = self.extractor.extract_json(text, prompt)

            # Handle both dict with 'algorithms' key or direct array
            if isinstance(data, dict) and 'algorithms' in data:
                algorithms_data = data['algorithms']
            elif isinstance(data, list):
                algorithms_data = data
            else:
                algorithms_data = []

            # If no algorithms found in LLM response, use fallback
            if not algorithms_data:
                print("No algorithms found in LLM response, using fallback")
                algorithms_data = self.extractor._extract_algorithms_fallback(text)

            return [Algorithm(**algo) for algo in algorithms_data]
        except Exception as e:
            print(f"LLM extraction failed for algorithms, using fallback: {e}")
            # Use fallback extraction
            algorithms_data = self.extractor._extract_algorithms_fallback(text)
            return [Algorithm(**algo) for algo in algorithms_data]

    def _extract_claims(self, text: str, algorithms: List[Algorithm]) -> List[TheoreticalClaim]:
        """Extract theoretical claims"""

        algo_names = [a.name for a in algorithms]

        prompt = f"""Extract ALL theoretical claims from this paper.

Algorithms in paper: {', '.join(algo_names)}

Look for:
- Convergence guarantees (e.g., "Sinkhorn converges in O(n²/ε) iterations")
- Complexity bounds (e.g., "MDL distance computable in O(nm) time")
- Approximation guarantees (e.g., "Blocking achieves ≥99.5% recall")
- Mathematical properties (e.g., "MDL satisfies quasi-metric property")

Return JSON array:
[
  {{
    "claim_id": "claim_1",
    "claim_type": "convergence|complexity|approximation|property",
    "statement": "precise claim statement",
    "assumptions": ["assumption1", "assumption2"],
    "proof_sketch": "brief proof outline if given",
    "relates_to_algorithm": "algorithm name this claim is about"
  }}
]"""

        try:
            data = self.extractor.extract_json(text, prompt)

            if isinstance(data, dict) and 'claims' in data:
                claims_data = data['claims']
            elif isinstance(data, list):
                claims_data = data
            else:
                claims_data = []

            # If no claims found in LLM response, use fallback
            if not claims_data:
                print("No claims found in LLM response, using fallback")
                claims_data = self.extractor._extract_claims_fallback(text)

            return [TheoreticalClaim(**claim) for claim in claims_data]
        except Exception as e:
            print(f"LLM extraction failed for claims, using fallback: {e}")
            # Use fallback extraction
            claims_data = self.extractor._extract_claims_fallback(text)
            return [TheoreticalClaim(**claim) for claim in claims_data]

    def _extract_experiments(self, text: str) -> ExperimentalSetup:
        """Extract experimental setup"""

        prompt = """Extract experimental setup from this paper.

Return JSON:
{
  "datasets": ["dataset1", "dataset2"],
  "evaluation_metrics": ["precision", "recall", "F1", etc],
  "hyperparameters": {
    "alpha": "value or range",
    "beta": "value or range",
    "epsilon": "value or range"
  },
  "expected_results": "summary of claimed results"
}"""

        try:
            data = self.extractor.extract_json(text[-6000:], prompt)  # Last 6k chars often have experiments
            return ExperimentalSetup(**data)
        except Exception as e:
            print(f"LLM extraction failed for experiments, using fallback: {e}")
            # Use fallback extraction
            experiments_data = self.extractor._extract_experiments_fallback(text)
            return ExperimentalSetup(**experiments_data)

    def _split_sections(self, text: str) -> dict:
        """Split paper into major sections"""
        sections = {}
        current_section = "introduction"
        current_content = []

        for line in text.split('\n'):
            if line.startswith('#'):
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content)

                # Start new section
                current_section = line.lstrip('#').strip().lower().replace(' ', '_')
                current_content = []
            else:
                current_content.append(line)

        # Save last section
        if current_content:
            sections[current_section] = '\n'.join(current_content)

        return sections
