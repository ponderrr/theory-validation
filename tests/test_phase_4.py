"""
Unit tests for Phase 4: Validation Execution
"""

import pytest
from pathlib import Path
import sys
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class TestPhase4:
    """Test cases for Phase 4 validation execution"""
    
    def test_find_validation_scripts(self):
        """Test finding validation scripts"""
        from phase_4 import find_validation_scripts
        
        # This will test the function exists and can be called
        scripts = find_validation_scripts()
        assert isinstance(scripts, list)
    
    def test_create_validation_summary(self):
        """Test validation summary creation"""
        from phase_4 import create_validation_summary
        
        results = [
            {"algorithm": "test1", "status": "completed"},
            {"algorithm": "test2", "status": "failed"},
            {"algorithm": "test3", "status": "error"}
        ]
        
        summary = create_validation_summary(results)
        
        assert summary["total_scripts"] == 3
        assert summary["successful"] == 1
        assert summary["failed"] == 1
        assert summary["errors"] == 1
        assert summary["success_rate"] == 1/3

if __name__ == "__main__":
    pytest.main([__file__])
