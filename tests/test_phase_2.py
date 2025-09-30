"""
Test Phase 2: Claim Comparison
"""
import pytest
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from claim_extractor.algorithm_matcher import AlgorithmMatcher
from claim_extractor.claim_comparator import ClaimComparator
from paper_parser.models import ParsedPaper

def test_algorithm_matcher():
    """Test algorithm matching"""
    matcher = AlgorithmMatcher()
    assert matcher is not None

def test_claim_comparator():
    """Test claim comparison"""
    comparator = ClaimComparator()
    assert comparator is not None

def test_similarity_detection():
    """Test that similar algorithm names are detected"""
    from paper_parser.models import Algorithm
    matcher = AlgorithmMatcher()

    algo1 = Algorithm(
        name="Constrained Sinkhorn",
        description="Optimal transport with constraints"
    )
    algo2 = Algorithm(
        name="Sinkhorn Algorithm",
        description="Entropic optimal transport"
    )

    assert matcher._are_similar(algo1, algo2)

def test_load_phase1_outputs():
    """Test loading Phase 1 outputs"""
    claims_dir = Path("output/extracted_claims")

    if not claims_dir.exists():
        pytest.skip("Phase 1 not completed")

    json_files = list(claims_dir.glob("*_parsed.json"))
    assert len(json_files) > 0, "No parsed papers found"

    # Try loading one
    with open(json_files[0]) as f:
        data = json.load(f)
        paper = ParsedPaper(**data)
        assert paper is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
