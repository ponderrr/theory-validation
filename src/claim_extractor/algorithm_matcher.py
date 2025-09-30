"""
Match algorithms across papers
"""
from typing import List, Dict
from difflib import SequenceMatcher
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from paper_parser.models import Algorithm, ParsedPaper
from .models import UnifiedAlgorithm

class AlgorithmMatcher:
    """Match algorithms across multiple papers"""

    # Known algorithm aliases
    ALGORITHM_ALIASES = {
        'sinkhorn': ['constrained sinkhorn', 'sinkhorn algorithm', 'entropic ot'],
        'mdl': ['mdl distance', 'mdl edit distance', 'minimum description length'],
        'blocking': ['multi-pass blocking', 'lsh blocking', 'candidate generation'],
        'clustering': ['nested clustering', 'hierarchical clustering', 'correlation clustering']
    }

    def match_algorithms(self, papers: List[ParsedPaper]) -> List[UnifiedAlgorithm]:
        """Find matching algorithms across papers"""

        # Collect all algorithms
        all_algos = []
        for paper in papers:
            for algo in paper.algorithms:
                all_algos.append((paper.metadata.source.value, algo))

        # Group by similarity
        groups = self._group_similar(all_algos)

        # Create unified algorithms
        unified = []
        for group_id, group in enumerate(groups):
            unified_algo = self._merge_algorithm_group(group_id, group)
            unified.append(unified_algo)

        return unified

    def _group_similar(self, algos: List[tuple]) -> List[List[tuple]]:
        """Group similar algorithms together"""
        groups = []
        used = set()

        for i, (source1, algo1) in enumerate(algos):
            if i in used:
                continue

            # Start new group
            group = [(source1, algo1)]
            used.add(i)

            # Find similar algorithms
            for j, (source2, algo2) in enumerate(algos):
                if j <= i or j in used:
                    continue

                if self._are_similar(algo1, algo2):
                    group.append((source2, algo2))
                    used.add(j)

            groups.append(group)

        return groups

    def _are_similar(self, algo1: Algorithm, algo2: Algorithm) -> bool:
        """Check if two algorithms are the same"""

        name1 = algo1.name.lower()
        name2 = algo2.name.lower()

        # Check aliases
        for canonical, aliases in self.ALGORITHM_ALIASES.items():
            if name1 in aliases and name2 in aliases:
                return True

        # Check name similarity
        similarity = SequenceMatcher(None, name1, name2).ratio()
        if similarity > 0.6:
            return True

        # Check description similarity
        desc1 = algo1.description.lower()
        desc2 = algo2.description.lower()
        desc_similarity = SequenceMatcher(None, desc1, desc2).ratio()
        if desc_similarity > 0.5:
            return True

        return False

    def _merge_algorithm_group(self, group_id: int, group: List[tuple]) -> UnifiedAlgorithm:
        """Merge algorithms from same group"""

        sources = [source for source, _ in group]
        algorithms = [algo for _, algo in group]

        # Pick canonical name (most common or shortest)
        names = [a.name for a in algorithms]
        canonical_name = min(names, key=len)

        # Merge descriptions
        descriptions = {source: algo.description for source, algo in group}
        consensus_desc = algorithms[0].description  # Use first as consensus

        # Merge input parameters
        all_params = {}
        for algo in algorithms:
            all_params.update(algo.input_parameters)

        # Collect complexity claims
        complexity_claims = {}
        for source, algo in group:
            complexity_claims[source] = {
                'time': algo.time_complexity or 'not specified',
                'space': algo.space_complexity or 'not specified'
            }

        return UnifiedAlgorithm(
            algorithm_id=f"algo_{group_id}",
            canonical_name=canonical_name,
            sources=sources,
            descriptions=descriptions,
            consensus_description=consensus_desc,
            input_params_consensus=all_params,
            complexity_claims=complexity_claims
        )
