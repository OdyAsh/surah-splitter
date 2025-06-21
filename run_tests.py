"""
Run tests for the Surah Splitter project.

This script provides a convenient way to run the test suite with various options.
"""

import subprocess
import argparse
from pathlib import Path


def main():
    """Run the test suite with options from command line arguments."""
    parser = argparse.ArgumentParser(description="Run tests for Surah Splitter")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--cov", action="store_true", help="Generate coverage report")
    parser.add_argument("--file", type=str, help="Run specific test file")
    parser.add_argument("--verbose", "-v", action="count", default=0, help="Increase verbosity")

    args = parser.parse_args()

    # Base command
    cmd = ["uvx", "python", "-m", "pytest"]

    # Add verbosity
    if args.verbose > 0:
        cmd.extend(["-" + "v" * args.verbose])
    else:
        cmd.append("-v")

    # Add test selection
    if args.unit:
        cmd.extend(["-m", "unit"])
    elif args.integration:
        cmd.extend(["-m", "integration"])

    # Add coverage if requested
    if args.cov:
        cmd.extend(["--cov=src/surah_splitter_new", "--cov-report", "term-missing"])

    # Add specific file if provided
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: Test file {file_path} not found")
            return 1
        cmd.append(str(file_path))

    # Run the command
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    exit(main())
