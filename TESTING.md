# Testing Guide for Surah Splitter

This guide explains how to test the service-based implementation of the Surah Splitter project.

<!-- TODO later: change the commands below to use python directly, once you're able to install pytest in pyproject.toml -->

## Setting Up the Testing Environment

1. Make sure you have the required testing libraries installed:

```powershell
uvx pip install pytest pytest-cov
```

2. The test suite is organized into:
   - Unit tests: Test individual services in isolation
   - Integration tests: Test services working together
   - Fixtures: Reusable mock data and utilities

## Running Tests

### Using the run_tests.py Script

The simplest way to run tests is using the provided script:

```powershell
# Run all tests
uvx python run_tests.py

# Run only unit tests
uvx python run_tests.py --unit

# Run only integration tests
uvx python run_tests.py --integration

# Run with code coverage report
uvx python run_tests.py --cov

# Run a specific test file
uvx python run_tests.py --file tests/unit/test_quran_metadata_service.py

# Increase verbosity
uvx python run_tests.py -v
```

### Using pytest Directly

You can also run pytest commands directly:

```powershell
# Run all tests
uvx python -m pytest

# Run unit tests
uvx python -m pytest -m unit

# Run integration tests
uvx python -m pytest -m integration

# Generate coverage report
uvx python -m pytest --cov=src/surah_splitter_new --cov-report term-missing

# Run a specific file
uvx python -m pytest tests/unit/test_transcription_service.py
```

## Writing More Tests

To expand the test coverage, follow these patterns:

### Unit Tests

1. Create a new test file in the `tests/unit/` directory, named `test_<service_name>.py`
2. Use the `@pytest.mark.unit` decorator to mark unit tests
3. Use unittest.mock to mock dependencies

Example:

```python
@pytest.mark.unit
def test_my_function():
    # Arrange
    input_data = ...
    
    # Act
    result = my_function(input_data)
    
    # Assert
    assert result == expected_result
```

### Integration Tests

1. Create a new test file in the `tests/integration/` directory, named `test_<integration_scenario>.py`
2. Use the `@pytest.mark.integration` decorator
3. Test how multiple services work together

## Test Fixtures

Reusable test fixtures are defined in `tests/fixtures/conftest.py`. Add new fixtures here that can be shared across multiple tests.

## Manual Testing

For manual testing of the CLI interface:

```powershell
# Full pipeline
uvx python -m surah_splitter_new.app.main_cli pipeline process -au "./data/input_surahs_to_split/adel_ryyan/076 Al-Insaan.mp3" -su 76 -re "adel_rayyan" -si -ssu

# Just transcription
uvx python -m surah_splitter_new.app.main_cli transcribe audio -au "./data/input_surahs_to_split/adel_ryyan/076 Al-Insaan.mp3" -o "./data/outputs/transcription.json"
```

## What to Test

For a comprehensive test suite, make sure to test:

1. **Normal Cases**: Standard input and expected behavior
2. **Edge Cases**: Boundary conditions, small inputs, large inputs
3. **Error Cases**: How the code handles invalid input, missing files, etc.

## Mocking External Services

When testing code that depends on external services (like WhisperX models), use mocking to avoid actual API calls:

```python
@patch("surah_splitter_new.services.transcription_service.load_model")
def test_with_mock(mock_load_model):
    mock_load_model.return_value = MockModel()
    # Test implementation here
```
