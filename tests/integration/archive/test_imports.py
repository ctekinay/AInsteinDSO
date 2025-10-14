"""
Test that all package imports work correctly.
"""

import pytest


def test_core_imports():
    """Test that core modules can be imported."""
    try:
        import src
        import src.exceptions.exceptions
        assert hasattr(src.exceptions.exceptions, 'UngroundedReplyError')
        assert hasattr(src.exceptions.exceptions, 'LowConfidenceError')
        assert hasattr(src.exceptions.exceptions, 'InvalidModelChangeError')
    except ImportError as e:
        pytest.fail(f"Failed to import core modules: {e}")


def test_package_structure():
    """Test that all package directories can be imported."""
    packages = [
        'src.knowledge',
        'src.safety',
        'src.routing',
        'src.archimate',
        'src.validation',
        'src.agent',
        'src.api',
        'src.evaluation'
    ]

    for package in packages:
        try:
            __import__(package)
        except ImportError as e:
            pytest.fail(f"Failed to import package {package}: {e}")


def test_exception_classes():
    """Test that custom exception classes work correctly."""
    from src.exceptions.exceptions import (
        UngroundedReplyError,
        LowConfidenceError,
        InvalidModelChangeError,
        PerformanceError
    )

    # Test UngroundedReplyError
    error = UngroundedReplyError()
    assert "archi:id-" in str(error)
    assert "skos:" in str(error)

    # Test LowConfidenceError
    error = LowConfidenceError(0.5)
    assert "0.500" in str(error)
    assert "0.75" in str(error)

    # Test InvalidModelChangeError
    error = InvalidModelChangeError("/path/to/model.xml")
    assert "/path/to/model.xml" in str(error)

    # Test PerformanceError
    error = PerformanceError("test_operation", 1000, 500)
    assert "1000ms" in str(error)
    assert "500ms" in str(error)