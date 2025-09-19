# stable-worldmodel | Test Suite

## Structure

The test suite is organized into two main categories:

```
stable_worldmodel/tests/
├── unit/           # Fast tests without external dependencies
├── integration/    # Tests requiring GPU, data downloads, or full training
├── conftest.py     # Shared pytest fixtures
├── utils.py        # Test utilities and mock classes
└── README.md       # This file
```

## Test Categories

### Unit Tests (`unit/`)
- Fast execution (< 1 second per test)
- No GPU requirements
- No data downloads
- Mock external dependencies
- Test individual components in isolation

### Integration Tests (`integration/`)
- May require GPU
- May download datasets
- Test full training pipelines
- Test component interactions
- Longer execution time

## Running Tests

### Run only unit tests (default, used in CI)
```bash
python -m pytest  # Default behavior
# or explicitly
python -m pytest -m unit
```

### Run integration tests
```bash
python -m pytest -m integration
```

### Run all tests
```bash
python -m pytest -m ""
```

### Run tests by specific markers
```bash
# GPU tests only
python -m pytest -m gpu

# Tests that download data
python -m pytest -m download

# Slow tests
python -m pytest -m slow

# Combine markers
python -m pytest -m "unit and not slow"
```

### Run specific test files
```bash
# Run all transform tests
python -m pytest stable_worldmodel/tests/unit/test_transforms.py
python -m pytest stable_worldmodel/tests/integration/test_transforms.py

# Run with coverage
python -m pytest --cov=stable_worldmodel -m unit
```

## Test Markers

- `@pytest.mark.unit` - Fast unit tests (no GPU, no downloads)
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.gpu` - Requires CUDA GPU
- `@pytest.mark.download` - Downloads data from internet
- `@pytest.mark.slow` - Takes more than 1 minute

## Writing New Tests

### Unit Test Example
```python
import pytest
import torch
import stable_worldmodel as swm


@pytest.mark.unit
class TestMyComponent:
    def test_initialization(self):
        # Test without GPU or data
        component = swm.MyComponent(param=10)
        assert component.param == 10

    def test_forward_mock(self):
        # Use mock data
        mock_input = torch.randn(4, 3, 32, 32)
        component = swm.MyComponent()
        output = component(mock_input)
        assert output.shape == (4, 10)
```

### Integration Test Example
```python
import pytest
import stable_worldmodel as swm


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.download
def test_full_training():
    # Test with real data and GPU
    dataset = swm.data.HFDataset(
        path="frgfm/imagenette",
        split="train",
        transform=transform
    )
    # ... full training pipeline
```

## GitHub Actions

The CI pipeline runs only unit tests by default:
```yaml
- name: Run Unit Tests
  run: python -m pytest stable_worldmodel/ -m unit --verbose --cov=stable_worldmodel
```

Integration tests can be run separately in nightly builds or on-demand.
