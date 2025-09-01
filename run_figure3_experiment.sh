#!/bin/bash
# Script to reproduce Figure 3 experiment from the paper
# Uses eval-set to run all three conditions with the same parameters

echo "🧪 Running Figure 3 Reproduction Experiment"
echo "============================================="
echo "Parameters:"
echo "  - Model: gpt-4o-mini"
echo "  - Samples: 50 per task"
echo "  - Epochs: 10"
echo "  - Max connections: 100"
echo ""

# Use eval-set to run all three tasks together
echo "📋 Running all GPQA tasks with eval-set..."
inspect eval-set \
  src/hint-experiment/gpqa_hint_task.py@gpqa_base_task \
  src/hint-experiment/gpqa_hint_task.py@gpqa_simple_hint_task \
  src/hint-experiment/gpqa_hint_task.py@gpqa_complex_hint_task \
  --model openai/gpt-4o-mini \
  --limit 50 \
  --epochs 10 \
  --max-connections 100\
  --logs-dir "figure-3"

echo ""
echo "✅ All experiments completed!"
echo "📊 Check the logs/ directory for the generated log files."
echo "📈 Use the analysis script to generate Figure 3 plot from the log files."
