"""
Models for unified claims across papers
"""
from typing import List, Dict, Optional
from pydantic import BaseModel
from enum import Enum

class AgreementLevel(str, Enum):
    FULL = "full"           # All papers agree
    PARTIAL = "partial"     # Some papers agree
    CONFLICT = "conflict"   # Papers contradict
    UNIQUE = "unique"       # Only one paper mentions

class ValidationPriority(str, Enum):
    CRITICAL = "critical"   # Core claims, conflicts
    HIGH = "high"          # Important claims
    MEDIUM = "medium"      # Supporting claims
    LOW = "low"            # Minor details

class UnifiedAlgorithm(BaseModel):
    algorithm_id: str
    canonical_name: str
    sources: List[str]  # claude, chatgpt, gemini
    descriptions: Dict[str, str]  # per source
    consensus_description: str
    input_params_consensus: Dict[str, str]
    complexity_claims: Dict[str, Dict[str, str]]  # source -> {time, space}

class UnifiedClaim(BaseModel):
    claim_id: str
    claim_type: str  # convergence, complexity, guarantee, property
    related_algorithm: Optional[str] = None
    sources: List[str]
    statements: Dict[str, str]  # source -> statement
    agreement_level: AgreementLevel
    consensus_statement: str
    conflicts: List[str] = []  # Descriptions of conflicts
    requires_validation: bool
    validation_priority: ValidationPriority
    validation_method: str  # theoretical, empirical, both

class ClaimComparison(BaseModel):
    total_papers: int
    unique_algorithms: List[UnifiedAlgorithm]
    unified_claims: List[UnifiedClaim]
    full_agreements: List[str]  # claim_ids with full agreement
    conflicts: List[str]  # claim_ids with conflicts
    unique_insights: List[str]  # claim_ids from single paper

class ValidationPlan(BaseModel):
    critical_claims: List[str]  # Must validate
    high_priority_claims: List[str]
    medium_priority_claims: List[str]
    conflicts_to_resolve: List[str]
    recommended_order: List[str]  # Order of validation
