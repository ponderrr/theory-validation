"""
Test Phase 1: Paper Parsing
"""
import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from paper_parser.parser import PaperParser
from paper_parser.models import PaperSource

def test_parser_initialization():
    """Test parser can be initialized"""
    parser = PaperParser()
    assert parser is not None

def test_parse_sample_paper():
    """Test parsing sample format paper"""
    parser = PaperParser()
    sample_path = Path("input_papers/SAMPLE_FORMAT.md")

    if not sample_path.exists():
        pytest.skip("Sample paper not found")

    parsed = parser.parse_paper(sample_path, PaperSource.CLAUDE)

    # Check basic structure
    assert parsed.metadata.source == PaperSource.CLAUDE
    assert len(parsed.algorithms) >= 3  # Should find at least 3 algorithms
    assert len(parsed.claims) >= 2      # Should find at least 2 claims
    assert parsed.problem.problem_statement != ""

def test_algorithm_extraction():
    """Test that key algorithms are extracted"""
    parser = PaperParser()
    sample_path = Path("input_papers/SAMPLE_FORMAT.md")

    if not sample_path.exists():
        pytest.skip("Sample paper not found")

    parsed = parser.parse_paper(sample_path, PaperSource.CLAUDE)

    algo_names = [a.name.lower() for a in parsed.algorithms]

    # Should find key algorithms
    assert any('sinkhorn' in name for name in algo_names)
    assert any('mdl' in name or 'edit' in name for name in algo_names)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
