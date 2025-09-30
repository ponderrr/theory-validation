# Contributing to Theory Validation System

Thank you for your interest in contributing to the Theory Validation System! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Development Workflow](#development-workflow)

## Code of Conduct

This project adheres to a code of conduct that ensures a welcoming environment for all contributors. Please be respectful and constructive in all interactions.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Create a new branch for your feature or bugfix
4. Make your changes
5. Test your changes thoroughly
6. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.11 or higher
- Git
- Virtual environment (venv, conda, or similar)

### Setup Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/theory-validation.git
   cd theory-validation
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp config/.env.template config/.env
   # Edit config/.env with your API keys
   ```

5. **Run tests to verify setup:**
   ```bash
   python -m pytest tests/
   ```

## Contributing Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write clear, descriptive variable and function names
- Add docstrings for all public functions and classes
- Keep functions focused and single-purpose

### Testing

- Write tests for new functionality
- Ensure all existing tests pass
- Aim for good test coverage
- Use descriptive test names

### Documentation

- Update README.md if you add new features
- Add docstrings to new functions and classes
- Update API documentation if applicable
- Include examples for new features

## Pull Request Process

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes:**
   - Write code following the style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes:**
   ```bash
   python -m pytest tests/
   python main.py  # Test the main pipeline
   ```

4. **Commit your changes:**
   ```bash
   git add .
   git commit -m "Add: Brief description of changes"
   ```

5. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request:**
   - Use a clear, descriptive title
   - Provide a detailed description of changes
   - Reference any related issues
   - Include screenshots or examples if applicable

### Pull Request Template

When creating a pull request, please include:

- **Description:** What changes were made and why
- **Type of Change:** Bug fix, new feature, documentation, etc.
- **Testing:** How the changes were tested
- **Checklist:** Ensure all items are completed

## Issue Reporting

When reporting issues, please include:

- **Description:** Clear description of the problem
- **Steps to Reproduce:** Detailed steps to reproduce the issue
- **Expected Behavior:** What should happen
- **Actual Behavior:** What actually happens
- **Environment:** OS, Python version, dependencies
- **Screenshots:** If applicable

## Development Workflow

### Adding New Phases

If you're adding a new phase to the validation pipeline:

1. Create the phase script in `src/phase_X.py`
2. Add corresponding models in `src/phase_X/`
3. Update `main.py` to include the new phase
4. Add tests in `tests/test_phase_X.py`
5. Update documentation

### Adding New Algorithms

To add support for new algorithms:

1. Add algorithm description to input papers
2. Update templates in `src/implementation_generator/templates.py`
3. Add validation logic if needed
4. Update documentation

### Modifying LLM Integration

When modifying LLM integration:

1. Update `src/paper_parser/llm_extractor.py`
2. Ensure fallback mechanisms work
3. Test with different LLM providers
4. Update configuration files

## Code Review Process

All pull requests require review before merging. Reviewers will check for:

- Code quality and style
- Test coverage
- Documentation completeness
- Performance implications
- Security considerations

## Release Process

Releases are managed through GitHub releases. The process includes:

1. Update version numbers
2. Update CHANGELOG.md
3. Create release notes
4. Tag the release
5. Publish to PyPI (if applicable)

## Questions?

If you have questions about contributing, please:

1. Check existing issues and discussions
2. Create a new issue with the "question" label
3. Reach out to maintainers

Thank you for contributing to the Theory Validation System!
