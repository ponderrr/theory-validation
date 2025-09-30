#!/usr/bin/env python3
"""
Phase 3: Generate implementations for algorithms
"""
import json
from pathlib import Path
from rich.console import Console
from rich.progress import track
import sys
sys.path.insert(0, str(Path(__file__).parent))

from implementation_generator.code_generator import CodeGenerator
from implementation_generator.templates import SINKHORN_TEMPLATE, MDL_TEMPLATE, BLOCKING_TEMPLATE, CLUSTERING_TEMPLATE

console = Console()

def load_unified_data():
    """Load algorithms and claims from Phase 2"""
    claims_dir = Path("output/extracted_claims")

    with open(claims_dir / "claim_comparison.json") as f:
        comparison = json.load(f)

    with open(claims_dir / "unified_claims.json") as f:
        claims = json.load(f)

    return comparison['unique_algorithms'], claims

def main():
    console.print("\n[bold blue]Phase 3: Implementation Generator[/bold blue]\n")
    console.print("=" * 60)

    # Load data
    console.print("\nLoading unified algorithms and claims...")
    algorithms, claims = load_unified_data()
    console.print(f"   Loaded {len(algorithms)} algorithms")
    console.print(f"   Loaded {len(claims)} claims to validate")

    # Create output directory
    output_dir = Path("output/implementations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize generator
    generator = CodeGenerator()

    console.print("\nGenerating implementations...\n")

    total_lines = 0
    total_tests = 0

    # Use templates for known algorithms
    templates = {
        'sinkhorn': ('constrained_sinkhorn', SINKHORN_TEMPLATE),
        'mdl': ('mdl_distance', MDL_TEMPLATE),
        'blocking': ('multi_pass_blocking', BLOCKING_TEMPLATE),
        'clustering': ('nested_clustering', CLUSTERING_TEMPLATE)
    }

    for i, algo in enumerate(algorithms, 1):
        console.print(f"[bold cyan][{i}/{len(algorithms)}] {algo['canonical_name']}[/bold cyan]")

        algo_name = algo['canonical_name'].lower().replace(' ', '_').replace('-', '_')

        # Check if we have a template
        template_key = None
        for key in templates:
            if key in algo_name:
                template_key = key
                break

        # Generate or use template
        if template_key:
            filename, template_code = templates[template_key]
            impl_code = template_code
            console.print(f"   Using template ({len(impl_code.split(chr(10)))} lines)")
        else:
            impl_code = generator.generate_implementation(algo)
            filename = algo_name
            console.print(f"   Generated implementation ({len(impl_code.split(chr(10)))} lines)")

        # Save implementation
        impl_file = output_dir / f"{filename}.py"
        with open(impl_file, 'w') as f:
            f.write(impl_code)

        total_lines += len(impl_code.split('\n'))

        # Generate tests
        test_code = generator.generate_tests(algo, impl_code)
        test_file = output_dir / f"{filename}_test.py"
        with open(test_file, 'w') as f:
            f.write(test_code)

        num_tests = test_code.count('def test_')
        total_tests += num_tests
        console.print(f"   Generated {num_tests} unit tests")

        # Generate validation experiment
        related_claims = [c for c in claims if c.get('related_algorithm') == algo['canonical_name']]

        if related_claims:
            validation_code = generator.generate_validation_experiment(algo, related_claims, impl_code)
            validation_file = output_dir / f"{filename}_validation.py"
            with open(validation_file, 'w') as f:
                f.write(validation_code)
            console.print(f"   Generated validation for {len(related_claims)} claims")

        console.print()

    # Summary
    console.print("[bold]Summary[/bold]")
    console.print("-" * 60)
    console.print(f"Implementations: {len(algorithms)}/{len(algorithms)}")
    console.print(f"Lines of code: ~{total_lines}")
    console.print(f"Unit tests: {total_tests}")
    console.print(f"\n[green] Output: {output_dir}/[/green]")

    console.print("\n[bold cyan]Phase 3 Complete! Ready for Phase 4.[/bold cyan]\n")

if __name__ == "__main__":
    main()
