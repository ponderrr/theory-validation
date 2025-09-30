#!/usr/bin/env python3
"""
Theory Validator - Main Entry Point
Validates research papers on hierarchical product matching
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console

console = Console()

def check_environment():
    """Verify environment is set up correctly"""
    console.print("\n[bold blue]Theory Validator - Environment Check[/bold blue]\n")

    # Check Python version
    py_version = sys.version_info
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 11):
        console.print("[red]❌ Python 3.11+ required[/red]")
        return False
    console.print(f"[green]✓[/green] Python {py_version.major}.{py_version.minor}")

    # Check .env file
    env_path = Path("config/.env")
    if not env_path.exists():
        console.print("[red]❌ config/.env not found[/red]")
        console.print("   Copy config/.env.template and fill in API keys")
        return False
    console.print("[green]✓[/green] config/.env exists")

    # Load environment
    load_dotenv(env_path)

    # Check for at least one API key
    has_api_key = any([
        os.getenv("GOOGLE_API_KEY"),
        os.getenv("OPENAI_API_KEY"),
        os.getenv("ANTHROPIC_API_KEY")
    ])
    if not has_api_key:
        console.print("[red]❌ No API key configured[/red]")
        return False
    console.print("[green]✓[/green] API key configured")

    # Check input papers
    input_dir = Path("input_papers")
    required_papers = ["claude_draft.md", "chatgpt_draft.md", "gemini_draft.md"]
    missing = [p for p in required_papers if not (input_dir / p).exists()]
    if missing:
        console.print(f"[yellow]⚠[/yellow] Missing input papers: {', '.join(missing)}")
        console.print("   (Will proceed with available papers)")
    else:
        console.print("[green]✓[/green] All input papers present")

    console.print("\n[bold green]Environment ready![/bold green]\n")
    return True

def main():
    """Main execution flow"""
    if not check_environment():
        console.print("[red]Setup incomplete. Please fix issues above.[/red]")
        sys.exit(1)

    console.print("[bold]Starting Theory Validator pipeline...[/bold]\n")
    console.print("Phase 0: ✓ Setup complete")
    console.print("Phase 1: Parsing papers (run phase_1.py)")
    console.print("Phase 2: Extracting claims (run phase_2.py)")
    console.print("Phase 3: Generating implementations (run phase_3.py)")
    console.print("Phase 4: Running validations (run phase_4.py)")
    console.print("Phase 5: Synthesizing results (run phase_5.py)")
    console.print("Phase 6: Generating master paper (run phase_6.py)")
    console.print("\n[bold cyan]Ready to proceed to Phase 1![/bold cyan]")

if __name__ == "__main__":
    main()
