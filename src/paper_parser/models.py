"""
Data models for parsed paper structure
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum

class PaperSource(str, Enum):
    CLAUDE = "claude"
    CHATGPT = "chatgpt"
    GEMINI = "gemini"

class PaperMetadata(BaseModel):
    source: PaperSource
    title: str
    date_parsed: str
    word_count: int

class ProblemFormulation(BaseModel):
    problem_statement: str
    input_description: str
    output_description: str
    constraints: List[str] = []
    success_metrics: List[str] = []

class Algorithm(BaseModel):
    name: str
    description: str
    pseudocode: Optional[str] = None
    input_parameters: Dict[str, str] = {}
    output_format: str = ""
    time_complexity: Optional[str] = None
    space_complexity: Optional[str] = None
    key_properties: List[str] = []

class TheoreticalClaim(BaseModel):
    claim_id: str
    claim_type: str  # theorem, lemma, property, guarantee
    statement: str
    assumptions: List[str] = []
    proof_sketch: Optional[str] = None
    relates_to_algorithm: Optional[str] = None

class ExperimentalSetup(BaseModel):
    datasets: List[str] = []
    evaluation_metrics: List[str] = []
    hyperparameters: Dict[str, Any] = {}
    expected_results: Optional[str] = None

class ParsedPaper(BaseModel):
    metadata: PaperMetadata
    problem: ProblemFormulation
    algorithms: List[Algorithm] = []
    claims: List[TheoreticalClaim] = []
    experiments: ExperimentalSetup
    raw_sections: Dict[str, str] = {}  # Store original text by section

    def get_algorithm(self, name: str) -> Optional[Algorithm]:
        """Find algorithm by name"""
        for algo in self.algorithms:
            if algo.name.lower() in name.lower() or name.lower() in algo.name.lower():
                return algo
        return None
