#!/usr/bin/env python3
"""
Phase 2: Extract and compare claims across papers
"""
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
import sys
sys.path.insert(0, str(Path(__file__).parent))

from paper_parser.models import ParsedPaper
from claim_extractor.algorithm_matcher import AlgorithmMatcher
from claim_extractor.claim_comparator import ClaimComparator
from claim_extractor.models import ClaimComparison, ValidationPlan, AgreementLevel

console = Console()

def load_parsed_papers() -> list:
    """Load all parsed papers from Phase 1"""
    papers = []
    claims_dir = Path("output/extracted_claims")

    for json_file in claims_dir.glob("*_parsed.json"):
        with open(json_file) as f:
            data = json.load(f)
            paper = ParsedPaper(**data)
            papers.append(paper)

    return papers

def main():
    console.print("\n[bold blue]Phase 2: Claim Extraction & Comparison[/bold blue]\n")
    console.print("=" * 60)

    # Load parsed papers
    console.print("\nLoading parsed papers...")
    papers = load_parsed_papers()

    if not papers:
        console.print("[red]No parsed papers found! Run Phase 1 first.[/red]")
        return

    for paper in papers:
        console.print(f"  ✓ Loaded {paper.metadata.source.value}_parsed.json")

    # Match algorithms
    console.print("\nMatching algorithms across papers...")
    matcher = AlgorithmMatcher()
    unified_algos = matcher.match_algorithms(papers)
    console.print(f"  ✓ Found {len(unified_algos)} unique algorithms")

    algo_mentions = sum(len(a.sources) for a in unified_algos)
    console.print(f"  ✓ Matched {algo_mentions} algorithm mentions")

    # Compare claims
    console.print("\nComparing theoretical claims...")
    comparator = ClaimComparator()
    unified_claims = comparator.compare_claims(papers, unified_algos)
    console.print(f"  ✓ Grouped into {len(unified_claims)} unique claims")

    # Analyze agreement
    full_agree = [c for c in unified_claims if c.agreement_level == AgreementLevel.FULL]
    conflicts = [c for c in unified_claims if c.agreement_level == AgreementLevel.CONFLICT]
    unique = [c for c in unified_claims if c.agreement_level == AgreementLevel.UNIQUE]

    console.print(f"  ✓ Found {len(full_agree)} agreements")
    console.print(f"  ✓ Found {len(conflicts)} conflicts")
    console.print(f"  ✓ Found {len(unique)} unique insights")

    # Display agreement analysis
    console.print("\n[bold]Agreement Analysis[/bold]")
    console.print("-" * 60)

    for claim in full_agree[:5]:  # Show top 5
        num_sources = len(claim.sources)
        total_sources = len(papers)
        console.print(f"✓ AGREE: {claim.consensus_statement[:60]}... [{num_sources}/{total_sources} papers]")

    for claim in conflicts:
        console.print(f"⚠ CONFLICT: {claim.consensus_statement[:60]}...")
        for conflict_desc in claim.conflicts:
            console.print(f"    {conflict_desc}")

    for claim in unique[:3]:  # Show top 3
        console.print(f"ℹ UNIQUE: {claim.consensus_statement[:60]}... [{claim.sources[0]}]")

    # Create comparison object
    comparison = ClaimComparison(
        total_papers=len(papers),
        unique_algorithms=unified_algos,
        unified_claims=unified_claims,
        full_agreements=[c.claim_id for c in full_agree],
        conflicts=[c.claim_id for c in conflicts],
        unique_insights=[c.claim_id for c in unique]
    )

    # Create validation plan
    critical = [c.claim_id for c in unified_claims if c.validation_priority.value == 'critical']
    high = [c.claim_id for c in unified_claims if c.validation_priority.value == 'high']
    medium = [c.claim_id for c in unified_claims if c.validation_priority.value == 'medium']

    # Order: conflicts first, then high priority, then rest
    validation_order = critical + high + medium

    validation_plan = ValidationPlan(
        critical_claims=critical,
        high_priority_claims=high,
        medium_priority_claims=medium,
        conflicts_to_resolve=[c.claim_id for c in conflicts],
        recommended_order=validation_order
    )

    # Save outputs
    output_dir = Path("output/extracted_claims")

    # Save unified claims
    with open(output_dir / "unified_claims.json", 'w') as f:
        json.dump([c.model_dump() for c in unified_claims], f, indent=2)
    console.print(f"\n[green]✓ Saved unified claims[/green]")

    # Save comparison
    with open(output_dir / "claim_comparison.json", 'w') as f:
        json.dump(comparison.model_dump(), f, indent=2)
    console.print(f"[green]✓ Saved comparison analysis[/green]")

    # Save conflicts
    with open(output_dir / "conflicts.json", 'w') as f:
        json.dump([c.model_dump() for c in conflicts], f, indent=2)
    console.print(f"[green]✓ Saved conflicts[/green]")

    # Save validation plan
    with open(output_dir / "validation_priorities.json", 'w') as f:
        json.dump(validation_plan.model_dump(), f, indent=2)
    console.print(f"[green]✓ Saved validation plan[/green]")

    # Summary table
    console.print("\n[bold]Validation Priority Summary[/bold]")
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Priority")
    table.add_column("Count", justify="center")
    table.add_column("Examples")

    table.add_row(
        "Critical",
        str(len(critical)),
        ", ".join([c.consensus_statement[:30] for c in conflicts[:2]])
    )
    table.add_row(
        "High",
        str(len(high)),
        f"{len(high)} convergence/complexity claims"
    )
    table.add_row(
        "Medium",
        str(len(medium)),
        f"{len(medium)} supporting claims"
    )

    console.print(table)

    console.print("\n[bold cyan]Phase 2 Complete! Ready for Phase 3.[/bold cyan]\n")

if __name__ == "__main__":
    main()
