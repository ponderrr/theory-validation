"""
Unit tests for Phase 3: Implementation Generation
"""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from implementation_generator.code_generator import CodeGenerator
from implementation_generator.templates import SINKHORN_TEMPLATE, MDL_TEMPLATE

class TestCodeGenerator:
    """Test cases for CodeGenerator"""
    
    def test_code_generator_init(self):
        """Test CodeGenerator initialization"""
        generator = CodeGenerator()
        assert generator is not None
        assert hasattr(generator, 'llm_extractor')
    
    def test_generate_basic_implementation(self):
        """Test basic implementation generation"""
        generator = CodeGenerator()
        algorithm = {
            'canonical_name': 'Test Algorithm',
            'description': 'A test algorithm',
            'complexity': 'O(n log n)'
        }
        
        impl = generator._generate_basic_implementation(algorithm)
        assert isinstance(impl, str)
        assert 'Test Algorithm' in impl
        assert 'def test_algorithm(' in impl
    
    def test_generate_basic_tests(self):
        """Test basic test generation"""
        generator = CodeGenerator()
        algorithm = {
            'canonical_name': 'Test Algorithm',
            'description': 'A test algorithm'
        }
        implementation = "def test_algorithm(data): return data"
        
        tests = generator._generate_basic_tests(algorithm, implementation)
        assert isinstance(tests, str)
        assert 'def test_' in tests
        assert 'pytest' in tests
    
    def test_generate_basic_validation(self):
        """Test basic validation generation"""
        generator = CodeGenerator()
        algorithm = {
            'canonical_name': 'Test Algorithm',
            'description': 'A test algorithm'
        }
        claims = [{'id': '1', 'text': 'Test claim'}]
        implementation = "def test_algorithm(data): return data"
        
        validation = generator._generate_basic_validation(algorithm, claims, implementation)
        assert isinstance(validation, str)
        assert 'def run_validation_experiment(' in validation
        assert 'json' in validation

class TestTemplates:
    """Test cases for algorithm templates"""
    
    def test_sinkhorn_template(self):
        """Test Sinkhorn template"""
        assert isinstance(SINKHORN_TEMPLATE, str)
        assert 'constrained_sinkhorn' in SINKHORN_TEMPLATE
        assert 'SinkhornResult' in SINKHORN_TEMPLATE
    
    def test_mdl_template(self):
        """Test MDL template"""
        assert isinstance(MDL_TEMPLATE, str)
        assert 'mdl_distance' in MDL_TEMPLATE
        assert 'MDLResult' in MDL_TEMPLATE

if __name__ == "__main__":
    pytest.main([__file__])
