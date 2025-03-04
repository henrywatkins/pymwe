import io

import pytest
from click.testing import CliRunner

from pymwe.cli import main


def test_cli_basic_functionality():
    """Test basic CLI functionality."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        # Create a test file
        with open("test_file.txt", "w") as f:
            f.write("apple banana cherry\n")
            f.write("banana cherry date\n")
            f.write("apple banana date\n")

        # Run the command
        result = runner.invoke(main, ["test_file.txt", "-n", "2"])
        assert result.exit_code == 0
        # Should return some MWEs
        assert len(result.output.strip().split("\n")) <= 2

        # Test with custom min_df and max_df
        result = runner.invoke(
            main, ["test_file.txt", "-n", "1", "--min_df", "1", "--max_df", "0.9"]
        )
        assert result.exit_code == 0
        # Should return one MWE
        assert len(result.output.strip().split("\n")) <= 1


def test_cli_with_empty_file():
    """Test CLI with empty file."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        # Create an empty test file
        with open("empty_file.txt", "w") as f:
            pass

        # Run the command
        result = runner.invoke(main, ["empty_file.txt"])
        # Should not error out
        assert result.exit_code == 0


def test_cli_with_invalid_options():
    """Test CLI with invalid options."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        # Create a test file
        with open("test_file.txt", "w") as f:
            f.write("apple banana cherry\n")

        # Test with invalid min_df (negative)
        result = runner.invoke(main, ["test_file.txt", "--min_df", "-1"])
        assert result.exit_code != 0

        # Test with invalid max_df (greater than 1)
        result = runner.invoke(main, ["test_file.txt", "--max_df", "1.5"])
        assert result.exit_code != 0
