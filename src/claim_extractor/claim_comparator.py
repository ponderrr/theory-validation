"""
Compare theoretical claims across papers
"""
from typing import List, Dict
from difflib import SequenceMatcher
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from paper_parser.models import TheoreticalClaim, ParsedPaper
from .models import UnifiedClaim, AgreementLevel, ValidationPriority

class ClaimComparator:
    """Compare claims across papers"""

    def compare_claims(self, papers: List[ParsedPaper], unified_algos: List) -> List[UnifiedClaim]:
        """Compare claims across all papers"""

        # Collect all claims
        all_claims = []
        for paper in papers:
            for claim in paper.claims:
                all_claims.append((paper.metadata.source.value, claim))

        # Group similar claims
        groups = self._group_similar_claims(all_claims)

        # Create unified claims
        unified = []
        for group_id, group in enumerate(groups):
            unified_claim = self._merge_claim_group(group_id, group)
            unified.append(unified_claim)

        return unified

    def _group_similar_claims(self, claims: List[tuple]) -> List[List[tuple]]:
        """Group similar claims together"""
        groups = []
        used = set()

        for i, (source1, claim1) in enumerate(claims):
            if i in used:
                continue

            group = [(source1, claim1)]
            used.add(i)

            # Find similar claims
            for j, (source2, claim2) in enumerate(claims):
                if j <= i or j in used:
                    continue

                if self._are_similar_claims(claim1, claim2):
                    group.append((source2, claim2))
                    used.add(j)

            groups.append(group)

        return groups

    def _are_similar_claims(self, claim1: TheoreticalClaim, claim2: TheoreticalClaim) -> bool:
        """Check if two claims are about the same thing"""

        # Same claim type is strong indicator
        if claim1.claim_type != claim2.claim_type:
            return False

        # Check if about same algorithm
        if claim1.relates_to_algorithm and claim2.relates_to_algorithm:
            algo1 = claim1.relates_to_algorithm.lower()
            algo2 = claim2.relates_to_algorithm.lower()
            algo_match = SequenceMatcher(None, algo1, algo2).ratio() > 0.6
            if not algo_match:
                return False

        # Check statement similarity
        stmt1 = claim1.statement.lower()
        stmt2 = claim2.statement.lower()
        similarity = SequenceMatcher(None, stmt1, stmt2).ratio()

        return similarity > 0.5

    def _merge_claim_group(self, group_id: int, group: List[tuple]) -> UnifiedClaim:
        """Merge claims from same group"""

        sources = [source for source, _ in group]
        claims = [claim for _, claim in group]

        # Collect statements
        statements = {source: claim.statement for source, claim in group}

        # Determine agreement level
        agreement = self._determine_agreement(claims)

        # Create consensus statement
        consensus = self._create_consensus_statement(claims, agreement)

        # Detect conflicts
        conflicts = self._detect_conflicts(claims)

        # Determine validation needs
        requires_validation = True  # Always validate
        priority = self._determine_priority(claims, agreement, conflicts)
        validation_method = self._determine_validation_method(claims[0].claim_type)

        return UnifiedClaim(
            claim_id=f"claim_{group_id}",
            claim_type=claims[0].claim_type,
            related_algorithm=claims[0].relates_to_algorithm,
            sources=sources,
            statements=statements,
            agreement_level=agreement,
            consensus_statement=consensus,
            conflicts=conflicts,
            requires_validation=requires_validation,
            validation_priority=priority,
            validation_method=validation_method
        )

    def _determine_agreement(self, claims: List[TheoreticalClaim]) -> AgreementLevel:
        """Determine level of agreement"""

        if len(claims) == 1:
            return AgreementLevel.UNIQUE

        # Check if statements are essentially the same
        statements = [c.statement.lower() for c in claims]

        # Simple heuristic: check pairwise similarity
        similarities = []
        for i in range(len(statements)):
            for j in range(i+1, len(statements)):
                sim = SequenceMatcher(None, statements[i], statements[j]).ratio()
                similarities.append(sim)

        avg_similarity = sum(similarities) / len(similarities) if similarities else 0

        if avg_similarity > 0.8:
            return AgreementLevel.FULL
        elif avg_similarity > 0.5:
            return AgreementLevel.PARTIAL
        else:
            return AgreementLevel.CONFLICT

    def _create_consensus_statement(self, claims: List[TheoreticalClaim], agreement: AgreementLevel) -> str:
        """Create consensus statement"""

        if agreement == AgreementLevel.FULL:
            return claims[0].statement
        elif agreement == AgreementLevel.UNIQUE:
            return claims[0].statement
        else:
            # For partial/conflict, note the variation
            return f"{claims[0].statement} (with variations across sources)"

    def _detect_conflicts(self, claims: List[TheoreticalClaim]) -> List[str]:
        """Detect specific conflicts between claims"""
        conflicts = []

        # Check for numeric conflicts (different O() bounds, etc)
        import re

        for i, claim1 in enumerate(claims):
            for claim2 in claims[i+1:]:
                # Extract numbers/complexities
                nums1 = re.findall(r'O\([^\)]+\)', claim1.statement)
                nums2 = re.findall(r'O\([^\)]+\)', claim2.statement)

                if nums1 and nums2 and nums1 != nums2:
                    conflicts.append(f"Complexity mismatch: {nums1[0]} vs {nums2[0]}")

                # Extract percentages
                pcts1 = re.findall(r'\d+\.?\d*%', claim1.statement)
                pcts2 = re.findall(r'\d+\.?\d*%', claim2.statement)

                if pcts1 and pcts2 and pcts1 != pcts2:
                    conflicts.append(f"Percentage mismatch: {pcts1[0]} vs {pcts2[0]}")

        return conflicts

    def _determine_priority(self, claims: List[TheoreticalClaim],
                          agreement: AgreementLevel, conflicts: List[str]) -> ValidationPriority:
        """Determine validation priority"""

        # Conflicts are critical
        if conflicts or agreement == AgreementLevel.CONFLICT:
            return ValidationPriority.CRITICAL

        # Core claims (convergence, complexity) are high priority
        if claims[0].claim_type in ['convergence', 'complexity']:
            return ValidationPriority.HIGH

        # Guarantees are high priority
        if claims[0].claim_type == 'guarantee':
            return ValidationPriority.HIGH

        # Everything else is medium
        return ValidationPriority.MEDIUM

    def _determine_validation_method(self, claim_type: str) -> str:
        """Determine how to validate this claim"""

        if claim_type == 'convergence':
            return 'empirical'  # Run experiments
        elif claim_type == 'complexity':
            return 'both'  # Theoretical + empirical
        elif claim_type == 'guarantee':
            return 'empirical'  # Test with data
        elif claim_type == 'property':
            return 'theoretical'  # Prove mathematically
        else:
            return 'empirical'
