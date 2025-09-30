#!/usr/bin/env python3
"""
Phase 4: Validation Execution

Runs validation experiments for all implemented algorithms.
"""

import json
import subprocess
import sys
from pathlib import Path
from rich.console import Console
from rich.progress import track
from rich.table import Table

console = Console()

def find_validation_scripts() -> list:
    """Find all validation scripts in output/implementations"""
    impl_dir = Path("output/implementations")
    if not impl_dir.exists():
        console.print("[red]No implementations directory found![/red]")
        return []
    
    validation_scripts = list(impl_dir.glob("*_validation.py"))
    console.print(f"Found {len(validation_scripts)} validation scripts")
    return validation_scripts

def run_validation_script(script_path: Path) -> dict:
    """Run a single validation script and capture results"""
    console.print(f"Running {script_path.name}...")
    
    try:
        # Run the validation script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        # Parse results
        if result.returncode == 0:
            # Try to find results file
            results_file = Path("output/validation_results") / f"{script_path.stem.replace('_validation', '')}_results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    return json.load(f)
            else:
                return {
                    "algorithm": script_path.stem.replace('_validation', ''),
                    "status": "completed",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode
                }
        else:
            return {
                "algorithm": script_path.stem.replace('_validation', ''),
                "status": "failed",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "error": "Script execution failed"
            }
    
    except subprocess.TimeoutExpired:
        return {
            "algorithm": script_path.stem.replace('_validation', ''),
            "status": "timeout",
            "error": "Script execution timed out"
        }
    except Exception as e:
        return {
            "algorithm": script_path.stem.replace('_validation', ''),
            "status": "error",
            "error": str(e)
        }

def create_validation_summary(results: list) -> dict:
    """Create summary of validation results"""
    total_scripts = len(results)
    successful = sum(1 for r in results if r.get('status') == 'completed')
    failed = sum(1 for r in results if r.get('status') == 'failed')
    errors = sum(1 for r in results if r.get('status') in ['error', 'timeout'])
    
    return {
        "total_scripts": total_scripts,
        "successful": successful,
        "failed": failed,
        "errors": errors,
        "success_rate": successful / total_scripts if total_scripts > 0 else 0,
        "results": results
    }

def display_results_table(summary: dict) -> None:
    """Display results in a table"""
    table = Table(title="Validation Results")
    table.add_column("Algorithm", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="yellow")
    
    for result in summary['results']:
        status = result.get('status', 'unknown')
        status_style = {
            'completed': 'green',
            'failed': 'red',
            'error': 'red',
            'timeout': 'yellow'
        }.get(status, 'white')
        
        details = ""
        if status == 'completed':
            details = f"Validated {result.get('validated_claims', 0)} claims"
        elif status == 'failed':
            details = result.get('error', 'Unknown error')
        else:
            details = result.get('error', 'Unknown issue')
        
        table.add_row(
            result.get('algorithm', 'Unknown'),
            f"[{status_style}]{status}[/{status_style}]",
            details
        )
    
    console.print(table)

def main():
    console.print("\n[bold blue]Phase 4: Validation Execution[/bold blue]")
    console.print("=" * 60)
    
    # Find validation scripts
    validation_scripts = find_validation_scripts()
    if not validation_scripts:
        console.print("[red]No validation scripts found![/red]")
        console.print("Make sure Phase 3 has been completed successfully.")
        return
    
    # Create output directory
    output_dir = Path("output/validation_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run validation scripts
    console.print(f"\nRunning {len(validation_scripts)} validation scripts...\n")
    
    results = []
    for script in track(validation_scripts, description="Validating algorithms..."):
        result = run_validation_script(script)
        results.append(result)
    
    # Create summary
    summary = create_validation_summary(results)
    
    # Display results
    console.print("\n[bold]Validation Summary[/bold]")
    console.print("-" * 60)
    console.print(f"Total scripts: {summary['total_scripts']}")
    console.print(f"Successful: {summary['successful']}")
    console.print(f"Failed: {summary['failed']}")
    console.print(f"Errors: {summary['errors']}")
    console.print(f"Success rate: {summary['success_rate']:.1%}")
    
    # Display detailed results
    display_results_table(summary)
    
    # Save results
    summary_file = output_dir / "validation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    console.print(f"\n[green]Results saved to {summary_file}[/green]")
    
    if summary['successful'] > 0:
        console.print("\n[bold cyan]Phase 4 Complete! Ready for Phase 5.[/bold cyan]")
    else:
        console.print("\n[bold red]Phase 4 Failed! No validations completed successfully.[/bold red]")

if __name__ == "__main__":
    main()
