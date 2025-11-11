# OpenSeek Tests

This directory contains unit tests, integration tests, and other test utilities for the OpenSeek project.

## Structure

```
tests/
├── unit/              # Unit tests for individual components
├── integration/       # Integration tests
├── data/              # Data processing tests
└── fixtures/          # Test fixtures and sample data
```

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run specific test suite
```bash
pytest tests/unit/
pytest tests/integration/
```

### Run with coverage
```bash
pytest tests/ --cov=openseek --cov-report=html
```

## Test Guidelines

1. **Unit Tests**: Test individual functions and classes in isolation
2. **Integration Tests**: Test component interactions and workflows
3. **Data Tests**: Test data processing pipelines and transformations
4. **Fixtures**: Use fixtures for reusable test data and setup

## Contributing Tests

When adding new features, please include corresponding tests:

1. Write tests alongside your code
2. Aim for good test coverage (>80%)
3. Use descriptive test names
4. Add docstrings to test functions explaining what they test

## Dependencies

Install test dependencies:
```bash
pip install pytest pytest-cov pytest-mock
```

