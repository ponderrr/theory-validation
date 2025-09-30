#!/usr/bin/env python3
"""
Phase 1: Parse input papers and extract structured information
"""
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from paper_parser.parser import PaperParser
from paper_parser.models import PaperSource

console = Console()

def main():
    console.print("\n[bold blue]Phase 1: Paper Parsing & Extraction[/bold blue]\n")
    console.print("=" * 50)

    # Initialize parser
    parser = PaperParser()

    # Define input papers
    input_papers = [
        ("input_papers/claude_draft.md", PaperSource.CLAUDE),
        ("input_papers/chatgpt_draft.md", PaperSource.CHATGPT),
        ("input_papers/gemini_draft.md", PaperSource.GEMINI),
    ]

    # Parse each paper
    parsed_papers = []
    output_dir = Path("output/extracted_claims")
    output_dir.mkdir(parents=True, exist_ok=True)

    for paper_path, source in input_papers:
        path = Path(paper_path)

        if not path.exists():
            console.print(f"[yellow]⚠ Skipping {path.name} (not found)[/yellow]")
            continue

        try:
            # Parse paper
            parsed = parser.parse_paper(path, source)
            parsed_papers.append(parsed)

            # Save parsed output
            output_file = output_dir / f"{source.value}_parsed.json"
            with open(output_file, 'w') as f:
                json.dump(parsed.model_dump(), f, indent=2)

            console.print(f"[green]✓ Saved to {output_file}[/green]")

        except Exception as e:
            console.print(f"[red]✗ Error parsing {path.name}: {e}[/red]")

    # Generate summary
    console.print("\n[bold]Summary[/bold]")
    console.print("-" * 50)

    if not parsed_papers:
        console.print("[red]No papers successfully parsed![/red]")
        return

    # Create summary table
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Source")
    table.add_column("Title")
    table.add_column("Algorithms", justify="center")
    table.add_column("Claims", justify="center")
    table.add_column("Word Count", justify="right")

    total_algos = 0
    total_claims = 0

    for paper in parsed_papers:
        table.add_row(
            paper.metadata.source.value,
            paper.metadata.title[:40] + "..." if len(paper.metadata.title) > 40 else paper.metadata.title,
            str(len(paper.algorithms)),
            str(len(paper.claims)),
            str(paper.metadata.word_count)
        )
        total_algos += len(paper.algorithms)
        total_claims += len(paper.claims)

    console.print(table)
    console.print(f"\n[bold]Total:[/bold] {len(parsed_papers)} papers, {total_algos} algorithms, {total_claims} claims")

    # Save summary
    summary = {
        "papers_parsed": len(parsed_papers),
        "total_algorithms": total_algos,
        "total_claims": total_claims,
        "papers": [
            {
                "source": p.metadata.source.value,
                "algorithms": len(p.algorithms),
                "claims": len(p.claims)
            }
            for p in parsed_papers
        ]
    }

    summary_file = output_dir / "parsing_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    console.print(f"\n[green]✓ Summary saved to {summary_file}[/green]")
    console.print("\n[bold cyan]Phase 1 Complete! Ready for Phase 2.[/bold cyan]\n")

if __name__ == "__main__":
    main()
