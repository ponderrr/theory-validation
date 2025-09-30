"""
Unit tests for Phase 5: Results Synthesis
"""

import pytest
from pathlib import Path
import sys
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class TestSynthesizer:
    """Test cases for Synthesizer"""
    
    def test_synthesizer_init(self):
        """Test Synthesizer initialization"""
        from synthesizer.synthesizer import Synthesizer
        
        synthesizer = Synthesizer()
        assert synthesizer is not None
        assert synthesizer.papers == []
        assert synthesizer.unified_claims == []
    
    def test_extract_common_keywords(self):
        """Test common keyword extraction"""
        from synthesizer.synthesizer import Synthesizer
        
        synthesizer = Synthesizer()
        texts = [
            "product matching algorithm",
            "matching products efficiently", 
            "algorithm for product matching"
        ]
        
        keywords = synthesizer._extract_common_keywords(texts)
        assert isinstance(keywords, list)
        assert len(keywords) > 0
    
    def test_synthesize_problem_formulation(self):
        """Test problem formulation synthesis"""
        from synthesizer.synthesizer import Synthesizer
        
        synthesizer = Synthesizer()
        synthesizer.papers = [
            {"problem_statement": "product matching challenge"},
            {"problem_statement": "matching products across datasets"}
        ]
        
        problem = synthesizer._synthesize_problem_formulation()
        assert isinstance(problem, dict)
        assert "unified_problem" in problem
        assert "key_challenges" in problem

if __name__ == "__main__":
    pytest.main([__file__])
