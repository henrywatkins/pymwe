import pytest

from pymwe import hello


def test_hello():
    """Test the hello function returns the expected greeting."""
    assert hello() == "Hello from pymwe!"
    assert isinstance(hello(), str)
