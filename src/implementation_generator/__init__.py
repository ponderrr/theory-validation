"""
Implementation Generator Module

Generates Python implementations, unit tests, and validation experiments
for algorithms extracted from research papers.
"""

from .code_generator import CodeGenerator
from .templates import (
    SINKHORN_TEMPLATE,
    MDL_TEMPLATE, 
    BLOCKING_TEMPLATE,
    CLUSTERING_TEMPLATE
)

__all__ = [
    'CodeGenerator',
    'SINKHORN_TEMPLATE',
    'MDL_TEMPLATE',
    'BLOCKING_TEMPLATE', 
    'CLUSTERING_TEMPLATE'
]
