#!/usr/bin/env python3
"""
Phase 6: Master Paper Generation

Generates the final comprehensive research paper.
"""

import json
from pathlib import Path
from rich.console import Console
import sys
sys.path.insert(0, str(Path(__file__).parent))

from report_generator.paper_generator import PaperGenerator

console = Console()

def main():
    console.print("\n[bold blue]Phase 6: Master Paper Generation[/bold blue]")
    console.print("=" * 60)
    
    # Initialize paper generator
    generator = PaperGenerator()
    
    # Load synthesis data
    synthesis_file = Path("output/master_paper/synthesis.json")
    if not synthesis_file.exists():
        console.print("[red]No synthesis file found![/red]")
        console.print("Make sure Phase 5 has been completed successfully.")
        return
    
    generator.load_synthesis(synthesis_file)
    
    # Generate paper
    paper_content = generator.generate_paper()
    
    # Display statistics
    generator.display_paper_stats()
    
    # Save paper
    output_file = Path("output/master_paper/VALIDATED_PRODUCT_MATCHING_PAPER.md")
    generator.save_paper(output_file)
    
    console.print("\n[bold green]🎉 ALL PHASES COMPLETE! 🎉[/bold green]")
    console.print("=" * 60)
    console.print("The theory validation pipeline has been successfully executed!")
    console.print(f"Master paper saved to: {output_file}")
    console.print("\n[bold cyan]Theory Validation System - Complete![/bold cyan]")

if __name__ == "__main__":
    main()
