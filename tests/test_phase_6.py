"""
Unit tests for Phase 6: Master Paper Generation
"""

import pytest
from pathlib import Path
import sys
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class TestPaperGenerator:
    """Test cases for PaperGenerator"""
    
    def test_paper_generator_init(self):
        """Test PaperGenerator initialization"""
        from report_generator.paper_generator import PaperGenerator
        
        generator = PaperGenerator()
        assert generator is not None
        assert generator.synthesis == {}
        assert generator.paper_content == ""
    
    def test_generate_title(self):
        """Test title generation"""
        from report_generator.paper_generator import PaperGenerator
        
        generator = PaperGenerator()
        title = generator._generate_title()
        
        assert isinstance(title, str)
        assert "Validated Product Matching" in title
        assert "#" in title  # Markdown header
    
    def test_generate_abstract(self):
        """Test abstract generation"""
        from report_generator.paper_generator import PaperGenerator
        
        generator = PaperGenerator()
        generator.synthesis = {
            "problem_formulation": {"paper_count": 3},
            "results": {"total_algorithms_tested": 5, "validation_success_rate": 0.8},
            "validated_claims": [{"text": "test claim"}]
        }
        
        abstract = generator._generate_abstract()
        assert isinstance(abstract, str)
        assert "## Abstract" in abstract
        assert "3" in abstract  # paper count
        assert "5" in abstract  # algorithm count
    
    def test_format_list(self):
        """Test list formatting"""
        from report_generator.paper_generator import PaperGenerator
        
        generator = PaperGenerator()
        items = ["item1", "item2", "item3"]
        formatted = generator._format_list(items)
        
        assert "1. item1" in formatted
        assert "2. item2" in formatted
        assert "3. item3" in formatted

if __name__ == "__main__":
    pytest.main([__file__])
