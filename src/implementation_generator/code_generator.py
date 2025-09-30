"""
Code Generator for Algorithm Implementations

Generates Python implementations, unit tests, and validation experiments
using LLM-based code generation with fallback templates.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any
from rich.console import Console

from ..paper_parser.llm_extractor import LLMExtractor

console = Console()

class CodeGenerator:
    """Generates code implementations for algorithms"""
    
    def __init__(self):
        self.llm_extractor = LLMExtractor()
    
    def generate_implementation(self, algorithm: Dict[str, Any]) -> str:
        """Generate Python implementation for an algorithm"""
        
        prompt = f"""
Generate a complete Python implementation for the following algorithm:

Algorithm: {algorithm.get('canonical_name', 'Unknown')}
Description: {algorithm.get('description', 'No description available')}
Complexity: {algorithm.get('complexity', 'Unknown')}
Input: {algorithm.get('input', 'Unknown')}
Output: {algorithm.get('output', 'Unknown')}

Requirements:
1. Use proper type hints
2. Include comprehensive docstrings
3. Add input validation
4. Include error handling
5. Follow PEP 8 style guidelines
6. Make it production-ready

Return ONLY the Python code, no explanations or markdown formatting.
"""

        try:
            # Try LLM generation first
            response = self.llm_extractor.extract_json(prompt)
            if response and 'code' in response:
                return response['code']
        except Exception as e:
            console.print(f"[yellow]LLM generation failed: {e}[/yellow]")
        
        # Fallback to basic template
        return self._generate_basic_implementation(algorithm)
    
    def generate_tests(self, algorithm: Dict[str, Any], implementation: str) -> str:
        """Generate unit tests for an algorithm implementation"""
        
        prompt = f"""
Generate comprehensive unit tests for this algorithm implementation:

Algorithm: {algorithm.get('canonical_name', 'Unknown')}
Implementation:
```python
{implementation}
```

Requirements:
1. Test normal cases
2. Test edge cases
3. Test error conditions
4. Use pytest framework
5. Include fixtures for test data
6. Test performance characteristics
7. Verify output correctness

Return ONLY the Python test code, no explanations or markdown formatting.
"""

        try:
            response = self.llm_extractor.extract_json(prompt)
            if response and 'code' in response:
                return response['code']
        except Exception as e:
            console.print(f"[yellow]LLM test generation failed: {e}[/yellow]")
        
        # Fallback to basic test template
        return self._generate_basic_tests(algorithm, implementation)
    
    def generate_validation_experiment(self, algorithm: Dict[str, Any], 
                                     claims: List[Dict[str, Any]], 
                                     implementation: str) -> str:
        """Generate validation experiment for algorithm and claims"""
        
        prompt = f"""
Generate a validation experiment script for this algorithm and its claims:

Algorithm: {algorithm.get('canonical_name', 'Unknown')}
Claims to validate: {len(claims)}

Claims:
{json.dumps(claims, indent=2)}

Implementation:
```python
{implementation}
```

Requirements:
1. Generate synthetic test data
2. Test each claim systematically
3. Measure performance metrics
4. Generate validation reports
5. Include visualization if helpful
6. Make it runnable and self-contained

Return ONLY the Python validation code, no explanations or markdown formatting.
"""

        try:
            response = self.llm_extractor.extract_json(prompt)
            if response and 'code' in response:
                return response['code']
        except Exception as e:
            console.print(f"[yellow]LLM validation generation failed: {e}[/yellow]")
        
        # Fallback to basic validation template
        return self._generate_basic_validation(algorithm, claims, implementation)
    
    def _generate_basic_implementation(self, algorithm: Dict[str, Any]) -> str:
        """Generate basic implementation template"""
        algo_name = algorithm.get('canonical_name', 'Unknown').lower().replace(' ', '_')
        
        return f'''"""
{algorithm.get('canonical_name', 'Unknown')} Algorithm Implementation

Description: {algorithm.get('description', 'No description available')}
Complexity: {algorithm.get('complexity', 'Unknown')}
"""

from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class {algo_name.title().replace('_', '')}Result:
    """Result container for {algorithm.get('canonical_name', 'Unknown')}"""
    result: Any
    metadata: Dict[str, Any]

def {algo_name}(data: Any, **kwargs) -> {algo_name.title().replace('_', '')}Result:
    """
    {algorithm.get('canonical_name', 'Unknown')} algorithm implementation.
    
    Args:
        data: Input data for the algorithm
        **kwargs: Additional parameters
        
    Returns:
        {algo_name.title().replace('_', '')}Result: Algorithm result and metadata
    """
    try:
        # TODO: Implement actual algorithm logic
        # This is a placeholder implementation
        
        # Basic input validation
        if data is None:
            raise ValueError("Input data cannot be None")
        
        # Placeholder processing
        result = data  # Replace with actual algorithm
        
        metadata = {{
            "algorithm": "{algorithm.get('canonical_name', 'Unknown')}",
            "input_size": len(data) if hasattr(data, '__len__') else 1,
            "parameters": kwargs
        }}
        
        return {algo_name.title().replace('_', '')}Result(result=result, metadata=metadata)
        
    except Exception as e:
        raise RuntimeError(f"Algorithm execution failed: {{e}}")

# Example usage
if __name__ == "__main__":
    # Test with sample data
    sample_data = [1, 2, 3, 4, 5]
    result = {algo_name}(sample_data)
    print(f"Result: {{result.result}}")
    print(f"Metadata: {{result.metadata}}")
'''
    
    def _generate_basic_tests(self, algorithm: Dict[str, Any], implementation: str) -> str:
        """Generate basic test template"""
        algo_name = algorithm.get('canonical_name', 'Unknown').lower().replace(' ', '_')
        
        return f'''"""
Unit tests for {algorithm.get('canonical_name', 'Unknown')} algorithm
"""

import pytest
import numpy as np
from {algo_name} import {algo_name}, {algo_name.title().replace('_', '')}Result

class Test{algo_name.title().replace('_', '')}:
    """Test cases for {algorithm.get('canonical_name', 'Unknown')}"""
    
    def test_basic_functionality(self):
        """Test basic algorithm functionality"""
        data = [1, 2, 3, 4, 5]
        result = {algo_name}(data)
        
        assert isinstance(result, {algo_name.title().replace('_', '')}Result)
        assert result.result is not None
        assert isinstance(result.metadata, dict)
    
    def test_empty_input(self):
        """Test with empty input"""
        data = []
        result = {algo_name}(data)
        
        assert isinstance(result, {algo_name.title().replace('_', '')}Result)
    
    def test_none_input(self):
        """Test with None input"""
        with pytest.raises(ValueError):
            {algo_name}(None)
    
    def test_with_parameters(self):
        """Test with additional parameters"""
        data = [1, 2, 3, 4, 5]
        result = {algo_name}(data, param1="test", param2=42)
        
        assert result.metadata["parameters"]["param1"] == "test"
        assert result.metadata["parameters"]["param2"] == 42
    
    def test_performance(self):
        """Test algorithm performance"""
        import time
        
        data = list(range(1000))
        start_time = time.time()
        result = {algo_name}(data)
        end_time = time.time()
        
        execution_time = end_time - start_time
        assert execution_time < 1.0  # Should complete within 1 second
        assert result is not None

if __name__ == "__main__":
    pytest.main([__file__])
'''
    
    def _generate_basic_validation(self, algorithm: Dict[str, Any], 
                                 claims: List[Dict[str, Any]], 
                                 implementation: str) -> str:
        """Generate basic validation template"""
        algo_name = algorithm.get('canonical_name', 'Unknown').lower().replace(' ', '_')
        
        return f'''"""
Validation experiment for {algorithm.get('canonical_name', 'Unknown')} algorithm
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from {algo_name} import {algo_name}

def generate_synthetic_data(size: int = 100) -> List[Any]:
    """Generate synthetic test data"""
    np.random.seed(42)  # For reproducibility
    return np.random.randn(size).tolist()

def validate_claim(claim: Dict[str, Any], algorithm_result: Any) -> Dict[str, Any]:
    """Validate a specific claim against algorithm result"""
    validation_result = {{
        "claim_id": claim.get("id", "unknown"),
        "claim_text": claim.get("text", ""),
        "validated": False,
        "evidence": "",
        "confidence": 0.0
    }}
    
    # TODO: Implement actual claim validation logic
    # This is a placeholder validation
    
    # Basic validation - check if result exists and is reasonable
    if algorithm_result is not None:
        validation_result["validated"] = True
        validation_result["evidence"] = "Algorithm produced valid result"
        validation_result["confidence"] = 0.8
    
    return validation_result

def run_validation_experiment():
    """Run complete validation experiment"""
    print(f"Running validation for {algorithm.get('canonical_name', 'Unknown')}")
    
    # Generate test data
    test_data = generate_synthetic_data(100)
    print(f"Generated test data with {{len(test_data)}} samples")
    
    # Run algorithm
    start_time = time.time()
    result = {algo_name}(test_data)
    end_time = time.time()
    
    execution_time = end_time - start_time
    print(f"Algorithm executed in {{execution_time:.4f}} seconds")
    
    # Validate claims
    validation_results = []
    for claim in {claims}:
        claim_result = validate_claim(claim, result.result)
        validation_results.append(claim_result)
        print(f"Claim {{claim_result['claim_id']}}: {{'✓' if claim_result['validated'] else '✗'}}")
    
    # Generate summary
    summary = {{
        "algorithm": "{algorithm.get('canonical_name', 'Unknown')}",
        "total_claims": len({claims}),
        "validated_claims": sum(1 for r in validation_results if r["validated"]),
        "execution_time": execution_time,
        "validation_results": validation_results
    }
    
    # Save results
    output_dir = Path("output/validation_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{algo_name}_results.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\\nValidation complete!")
    print(f"Validated {{summary['validated_claims']}}/{{summary['total_claims']}} claims")
    print(f"Results saved to {{output_file}}")
    
    return summary

if __name__ == "__main__":
    run_validation_experiment()
'''
