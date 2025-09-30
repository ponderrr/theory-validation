#!/usr/bin/env python3
"""
Phase 5: Results Synthesis

Synthesizes findings across papers and validation results.
"""

import json
from pathlib import Path
from rich.console import Console
import sys
sys.path.insert(0, str(Path(__file__).parent))

from synthesizer.synthesizer import Synthesizer

console = Console()

def main():
    console.print("\n[bold blue]Phase 5: Results Synthesis[/bold blue]")
    console.print("=" * 60)
    
    # Initialize synthesizer
    synthesizer = Synthesizer()
    
    # Load data
    data_dir = Path("output")
    if not data_dir.exists():
        console.print("[red]No output directory found![/red]")
        console.print("Make sure previous phases have been completed.")
        return
    
    synthesizer.load_data(data_dir)
    
    # Perform synthesis
    synthesis = synthesizer.synthesize()
    
    # Display summary
    synthesizer.display_summary()
    
    # Save synthesis
    output_dir = Path("output/master_paper")
    synthesizer.save_synthesis(output_dir)
    
    console.print("\n[bold cyan]Phase 5 Complete! Ready for Phase 6.[/bold cyan]")

if __name__ == "__main__":
    main()
