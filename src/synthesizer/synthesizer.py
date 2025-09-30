"""
Results Synthesizer

Synthesizes findings across papers and validation results to generate
comprehensive insights and unified conclusions.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from rich.console import Console
from rich.table import Table

console = Console()

class Synthesizer:
    """Synthesizes validation results and generates unified insights"""
    
    def __init__(self):
        self.papers = []
        self.unified_claims = []
        self.conflicts = []
        self.validation_results = {}
        self.synthesis = {}
    
    def load_data(self, data_dir: Path) -> None:
        """Load all necessary data for synthesis"""
        console.print("Loading data for synthesis...")
        
        # Load parsed papers
        papers_dir = data_dir / "extracted_claims"
        for paper_file in papers_dir.glob("*_parsed.json"):
            with open(paper_file, 'r') as f:
                paper_data = json.load(f)
                self.papers.append(paper_data)
        
        # Load unified claims
        with open(papers_dir / "unified_claims.json", 'r') as f:
            self.unified_claims = json.load(f)
        
        # Load conflicts
        with open(papers_dir / "conflicts.json", 'r') as f:
            self.conflicts = json.load(f)
        
        # Load validation results
        validation_dir = data_dir / "validation_results"
        if validation_dir.exists():
            for result_file in validation_dir.glob("*_results.json"):
                with open(result_file, 'r') as f:
                    result_data = json.load(f)
                    algorithm_name = result_file.stem.replace("_results", "")
                    self.validation_results[algorithm_name] = result_data
        
        console.print(f"Loaded {len(self.papers)} papers, {len(self.unified_claims)} claims, {len(self.conflicts)} conflicts")
    
    def synthesize(self) -> Dict[str, Any]:
        """Perform comprehensive synthesis"""
        console.print("\n[bold blue]Synthesizing Results[/bold blue]")
        console.print("=" * 50)
        
        synthesis = {
            "problem_formulation": self._synthesize_problem_formulation(),
            "algorithms": self._synthesize_algorithms(),
            "validated_claims": self._synthesize_validated_claims(),
            "refuted_claims": self._synthesize_refuted_claims(),
            "resolved_conflicts": self._synthesize_resolved_conflicts(),
            "methodology": self._synthesize_methodology(),
            "results": self._synthesize_results(),
            "limitations": self._synthesize_limitations(),
            "future_work": self._synthesize_future_work()
        }
        
        self.synthesis = synthesis
        return synthesis
    
    def _synthesize_problem_formulation(self) -> Dict[str, Any]:
        """Synthesize the problem formulation across papers"""
        problem_statements = []
        for paper in self.papers:
            if 'problem_statement' in paper:
                problem_statements.append(paper['problem_statement'])
        
        # Find common themes
        common_keywords = self._extract_common_keywords(problem_statements)
        
        return {
            "unified_problem": "Product matching across heterogeneous datasets using multiple algorithmic approaches",
            "key_challenges": [
                "Scalability to large datasets",
                "Handling noisy and incomplete data",
                "Ensuring matching accuracy",
                "Computational efficiency"
            ],
            "common_keywords": common_keywords,
            "paper_count": len(self.papers)
        }
    
    def _synthesize_algorithms(self) -> List[Dict[str, Any]]:
        """Synthesize algorithms across papers"""
        algorithms = []
        
        # Extract algorithms from papers
        for paper in self.papers:
            if 'algorithms' in paper:
                for algo in paper['algorithms']:
                    algorithms.append({
                        "name": algo.get('name', 'Unknown'),
                        "description": algo.get('description', ''),
                        "complexity": algo.get('complexity', 'Unknown'),
                        "source_paper": paper.get('title', 'Unknown'),
                        "validated": algo.get('name', '').lower() in self.validation_results
                    })
        
        return algorithms
    
    def _synthesize_validated_claims(self) -> List[Dict[str, Any]]:
        """Synthesize claims that were successfully validated"""
        validated_claims = []
        
        for claim in self.unified_claims:
            # Check if claim was validated
            is_validated = self._is_claim_validated(claim)
            
            if is_validated:
                validated_claims.append({
                    "text": claim.get('text', ''),
                    "algorithm": claim.get('related_algorithm', 'Unknown'),
                    "confidence": claim.get('confidence', 0.0),
                    "validation_evidence": self._get_validation_evidence(claim)
                })
        
        return validated_claims
    
    def _synthesize_refuted_claims(self) -> List[Dict[str, Any]]:
        """Synthesize claims that were refuted by validation"""
        refuted_claims = []
        
        for claim in self.unified_claims:
            is_validated = self._is_claim_validated(claim)
            
            if not is_validated and claim.get('confidence', 0) > 0.5:
                refuted_claims.append({
                    "text": claim.get('text', ''),
                    "algorithm": claim.get('related_algorithm', 'Unknown'),
                    "original_confidence": claim.get('confidence', 0.0),
                    "refutation_reason": "Failed validation experiments"
                })
        
        return refuted_claims
    
    def _synthesize_resolved_conflicts(self) -> List[Dict[str, Any]]:
        """Synthesize conflicts that were resolved through validation"""
        resolved_conflicts = []
        
        for conflict in self.conflicts:
            resolution = self._resolve_conflict(conflict)
            if resolution:
                resolved_conflicts.append({
                    "conflict_description": conflict.get('description', ''),
                    "resolution": resolution,
                    "resolution_method": "Experimental validation"
                })
        
        return resolved_conflicts
    
    def _synthesize_methodology(self) -> Dict[str, Any]:
        """Synthesize the validation methodology"""
        return {
            "approach": "Multi-phase validation pipeline",
            "phases": [
                "Paper parsing and claim extraction",
                "Claim comparison and conflict identification", 
                "Algorithm implementation generation",
                "Experimental validation",
                "Results synthesis",
                "Master paper generation"
            ],
            "validation_methods": [
                "Synthetic data generation",
                "Performance benchmarking",
                "Accuracy testing",
                "Scalability analysis"
            ],
            "tools_used": [
                "Python implementations",
                "Unit testing framework",
                "Statistical analysis",
                "LLM-based code generation"
            ]
        }
    
    def _synthesize_results(self) -> Dict[str, Any]:
        """Synthesize validation results"""
        total_algorithms = len(self.validation_results)
        successful_validations = sum(1 for result in self.validation_results.values() 
                                   if result.get('validated_claims', 0) > 0)
        
        return {
            "total_algorithms_tested": total_algorithms,
            "successful_validations": successful_validations,
            "validation_success_rate": successful_validations / total_algorithms if total_algorithms > 0 else 0,
            "total_claims_validated": sum(result.get('validated_claims', 0) for result in self.validation_results.values()),
            "average_execution_time": self._calculate_average_execution_time(),
            "performance_metrics": self._extract_performance_metrics()
        }
    
    def _synthesize_limitations(self) -> List[str]:
        """Synthesize limitations identified"""
        return [
            "Limited to synthetic data validation",
            "Real-world data complexity not fully captured",
            "Performance metrics may not reflect production conditions",
            "Limited scalability testing on very large datasets",
            "Human evaluation not included in validation process"
        ]
    
    def _synthesize_future_work(self) -> List[str]:
        """Synthesize future work recommendations"""
        return [
            "Validate on real-world product datasets",
            "Implement more sophisticated similarity metrics",
            "Add human evaluation components",
            "Scale testing to larger datasets",
            "Integrate with production systems",
            "Develop automated claim extraction improvements"
        ]
    
    def _extract_common_keywords(self, texts: List[str]) -> List[str]:
        """Extract common keywords from text list"""
        # Simple keyword extraction (in practice, would use more sophisticated NLP)
        all_words = []
        for text in texts:
            words = text.lower().split()
            all_words.extend(words)
        
        # Count word frequencies
        word_counts = {}
        for word in all_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Return most common words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:10] if len(word) > 3]
    
    def _is_claim_validated(self, claim: Dict[str, Any]) -> bool:
        """Check if a claim was validated"""
        algorithm = claim.get('related_algorithm', '').lower()
        if algorithm in self.validation_results:
            result = self.validation_results[algorithm]
            return result.get('validated_claims', 0) > 0
        return False
    
    def _get_validation_evidence(self, claim: Dict[str, Any]) -> str:
        """Get validation evidence for a claim"""
        algorithm = claim.get('related_algorithm', '').lower()
        if algorithm in self.validation_results:
            result = self.validation_results[algorithm]
            return f"Validated through {result.get('total_claims', 0)} experimental tests"
        return "No validation evidence available"
    
    def _resolve_conflict(self, conflict: Dict[str, Any]) -> Optional[str]:
        """Resolve a conflict through validation results"""
        # Simple conflict resolution logic
        if 'algorithm' in conflict:
            algorithm = conflict['algorithm'].lower()
            if algorithm in self.validation_results:
                result = self.validation_results[algorithm]
                if result.get('validated_claims', 0) > 0:
                    return f"Resolved in favor of {algorithm} based on validation results"
        return None
    
    def _calculate_average_execution_time(self) -> float:
        """Calculate average execution time across validations"""
        times = [result.get('execution_time', 0) for result in self.validation_results.values()]
        return sum(times) / len(times) if times else 0.0
    
    def _extract_performance_metrics(self) -> Dict[str, Any]:
        """Extract performance metrics from validation results"""
        metrics = {
            "total_execution_time": sum(result.get('execution_time', 0) for result in self.validation_results.values()),
            "algorithms_with_errors": sum(1 for result in self.validation_results.values() if result.get('errors', 0) > 0),
            "average_claims_per_algorithm": sum(result.get('total_claims', 0) for result in self.validation_results.values()) / len(self.validation_results) if self.validation_results else 0
        }
        return metrics
    
    def display_summary(self) -> None:
        """Display synthesis summary"""
        console.print("\n[bold green]Synthesis Summary[/bold green]")
        console.print("=" * 50)
        
        # Create summary table
        table = Table(title="Validation Results Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Papers Analyzed", str(len(self.papers)))
        table.add_row("Total Claims", str(len(self.unified_claims)))
        table.add_row("Conflicts Identified", str(len(self.conflicts)))
        table.add_row("Algorithms Tested", str(len(self.validation_results)))
        table.add_row("Validated Claims", str(len(self._synthesize_validated_claims())))
        table.add_row("Refuted Claims", str(len(self._synthesize_refuted_claims())))
        
        console.print(table)
    
    def save_synthesis(self, output_dir: Path) -> None:
        """Save synthesis results"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        synthesis_file = output_dir / "synthesis.json"
        with open(synthesis_file, 'w') as f:
            json.dump(self.synthesis, f, indent=2)
        
        console.print(f"\n[green]Synthesis saved to {synthesis_file}[/green]")
