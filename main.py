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
    """Enhanced environment checking with better user guidance"""
    console.print("\n[bold blue]Theory Validator - Environment Check[/bold blue]\n")

    # Check Python version
    py_version = sys.version_info
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 11):
        console.print("[red]❌ Python 3.11+ required[/red]")
        console.print("   Current version:", f"{py_version.major}.{py_version.minor}")
        console.print("   Please upgrade Python: https://www.python.org/downloads/")
        return False
    console.print(f"[green]✓[/green] Python {py_version.major}.{py_version.minor}")

    # Check .env file
    env_path = Path("config/.env")
    if not env_path.exists():
        console.print("[red]❌ config/.env not found[/red]")
        console.print("\n[yellow]Quick fix:[/yellow]")
        console.print("1. Copy the template: [cyan]cp config/.env.template config/.env[/cyan]")
        console.print("2. Edit the file: [cyan]notepad config/.env[/cyan] (Windows) or [cyan]nano config/.env[/cyan] (macOS/Linux)")
        console.print("3. Add your API key (Google AI Studio is FREE!)")
        console.print("   Get free key: [cyan]https://aistudio.google.com/app/apikey[/cyan]")
        return False
    console.print("[green]✓[/green] config/.env exists")

    # Load environment
    load_dotenv(env_path)

    # Check for at least one API key
    api_keys = {
        "Google AI Studio": os.getenv("GOOGLE_API_KEY"),
        "OpenAI": os.getenv("OPENAI_API_KEY"),
        "Anthropic": os.getenv("ANTHROPIC_API_KEY")
    }
    
    configured_keys = [name for name, key in api_keys.items() if key and key != "your_api_key_here"]
    
    if not configured_keys:
        console.print("[red]❌ No API key configured[/red]")
        console.print("\n[yellow]Quick fix:[/yellow]")
        console.print("1. Edit config/.env")
        console.print("2. Add at least one API key:")
        console.print("   [cyan]GOOGLE_API_KEY=your_actual_key_here[/cyan]")
        console.print("3. Get FREE Google AI Studio key: [cyan]https://aistudio.google.com/app/apikey[/cyan]")
        return False
    
    console.print(f"[green]✓[/green] API keys configured: {', '.join(configured_keys)}")

    # Check input papers
    input_dir = Path("input_papers")
    if not input_dir.exists():
        console.print("[red]❌ input_papers/ directory not found[/red]")
        console.print("\n[yellow]Quick fix:[/yellow]")
        console.print("1. Create directory: [cyan]mkdir input_papers[/cyan]")
        console.print("2. Add research papers (see docs/RESEARCH_PROMPTS.md)")
        return False

    # Check for papers
    papers = list(input_dir.glob("*.md"))
    required_papers = ["claude_draft.md", "chatgpt_draft.md", "gemini_draft.md"]
    missing = [p for p in required_papers if not (input_dir / p).exists()]
    
    if len(papers) == 0:
        console.print("[red]❌ No papers found in input_papers/[/red]")
        console.print("\n[yellow]Quick fix:[/yellow]")
        console.print("1. Generate research papers using prompts in [cyan]docs/RESEARCH_PROMPTS.md[/cyan]")
        console.print("2. Save as .md files in input_papers/")
        console.print("3. Or copy sample: [cyan]cp input_papers/SAMPLE_FORMAT.md input_papers/claude_draft.md[/cyan]")
        return False
    elif missing:
        console.print(f"[yellow]⚠[/yellow] Missing recommended papers: {', '.join(missing)}")
        console.print(f"   Found {len(papers)} papers, will proceed with available ones")
        console.print("   For best results, use all 3 papers from different AIs")
    else:
        console.print("[green]✓[/green] All input papers present")

    # Check output directory
    output_dir = Path("output")
    if not output_dir.exists():
        console.print("[yellow]⚠[/yellow] Creating output directory...")
        output_dir.mkdir(exist_ok=True)
        console.print("[green]✓[/green] Output directory created")

    console.print("\n[bold green]Environment ready![/bold green]")
    console.print(f"[cyan]Found {len(papers)} papers to process[/cyan]")
    console.print("[cyan]Ready to start validation pipeline...[/cyan]\n")
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
