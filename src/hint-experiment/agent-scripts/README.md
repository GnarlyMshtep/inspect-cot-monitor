# Agent Scripts

This directory contains debugging and testing scripts for the GPQA hint experiment.

## Scripts

### Testing Scripts

- **`test_gpqa.py`** - Comprehensive test suite that validates all components
  ```bash
  uv run agent-scripts/test_gpqa.py
  ```

- **`show_examples.py`** - Shows examples of complex hints with math verification
  ```bash
  uv run agent-scripts/show_examples.py
  ```

### Debugging Scripts

- **`debug_scorer.py`** - Tests basic scorer behavior with mock inputs
  ```bash
  uv run agent-scripts/debug_scorer.py
  ```

- **`debug_correctness.py`** - Specifically tests the is_correct_scorer logic
  ```bash
  uv run agent-scripts/debug_correctness.py
  ```

- **`debug_hint_scorer.py`** - Tests the hint usage scorer with different scenarios
  ```bash
  uv run agent-scripts/debug_hint_scorer.py
  ```

### Experiment Runner

- **`run_experiment.py`** - Example script for running experiments
  ```bash
  # Quick tests
  uv run agent-scripts/run_experiment.py base 2
  uv run agent-scripts/run_experiment.py simple 2
  uv run agent-scripts/run_experiment.py complex 2
  
  # Shows usage and runs a quick test
  uv run agent-scripts/run_experiment.py
  ```

## Usage Notes

- All scripts automatically add the parent directory to the Python path to import from `gpqa_hint_task.py`
- Scripts that test the hint usage scorer require an OpenAI API key (set via .env file or environment variable)
- Run from the main project directory for best results

## Troubleshooting

If you get import errors, make sure you're running from the project root:
```bash
cd /path/to/inspect-cot-monitor
uv run src/hint-experiment/agent-scripts/test_gpqa.py
```

