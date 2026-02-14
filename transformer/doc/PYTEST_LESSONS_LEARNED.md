# Lessons Learned: Pytest Test Collection Issues

## Issue Summary

During directory reorganization, pytest failed to collect tests in `test_callbacks.py` and `test_early_stopping.py` with a confusing error that appeared to be a TensorBoard/TensorFlow import issue.

## Root Cause

The actual problem was **pytest's test class naming convention**:

- Pytest automatically collects classes that start with `Test` as test classes
- Test classes in pytest **must not define a custom `__init__` method**
- When a class starting with `Test` has a custom `__init__`, pytest fails during collection

## The Specific Problem

In `test_callbacks.py`, there was a helper class:

```python
class TestCallback(TrainerCallback):
    """Test callback that records when hooks are called."""

    def __init__(self):  # <- This causes pytest to fail!
        self.calls = {
            'on_train_start': 0,
            # ...
        }
```

Pytest tried to collect `TestCallback` as a test class because it starts with "Test", but failed because it has a custom `__init__`.

## The Misleading Error

The error appeared to be about TensorFlow/TensorBoard imports:
```
AttributeError: module 'tensorflow' has no attribute 'io'
```

This was misleading because:
1. The import worked fine outside pytest
2. The import worked fine in other test files
3. The actual issue was pytest's collection mechanism, not imports

## Solution

Rename helper classes to avoid the `Test` prefix:

```python
class MockCallback(TrainerCallback):  # ✓ No longer starts with "Test"
    """Mock callback that records when hooks are called."""

    def __init__(self):  # Now pytest ignores this class
        self.calls = {
            'on_train_start': 0,
            # ...
        }
```

## Best Practices

### DO:
- ✅ Use `Mock`, `Dummy`, `Fake`, `Stub` prefixes for helper classes
- ✅ Use `Test` prefix ONLY for actual pytest test classes
- ✅ Keep actual test classes without `__init__` methods

### DON'T:
- ❌ Don't name helper classes starting with `Test`
- ❌ Don't add `__init__` to classes starting with `Test` in test files
- ❌ Don't assume import errors in pytest are always about imports

## Prevention

When creating helper classes in test files, always use naming conventions that pytest won't collect:
- `MockXxx` - for mock objects
- `DummyXxx` - for simple stub implementations
- `FakeXxx` - for fake implementations
- `StubXxx` - for minimal implementations
- `_TestXxx` - underscore prefix also works (private)

## References

- [Pytest: Conventions for Python test discovery](https://docs.pytest.org/en/stable/explanation/goodpractices.html#conventions-for-python-test-discovery)
- Pytest collects classes matching `Test*` by default
- Test classes must be instantiable without arguments
