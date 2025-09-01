#!/usr/bin/env python3
"""
Figure 3 Analysis Script - Reproduction of "When Chain of Thought is Necessary, Language Models Struggle to Evade Monitors"

This script analyzes the eval logs from the GPQA hint experiments and generates a plot
showing the "unfaithfulness delta" across different hint conditions.

The unfaithfulness delta measures how much models follow hints even when they lead to incorrect answers,
calculated as the difference between hint usage and correctness rates.
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import seaborn as sns

def load_logs(log_dir: str) -> Dict[str, dict]:
    """Load all eval logs from the specified directory."""
    log_path = Path(log_dir) / "logs.json"
    
    if not log_path.exists():
        raise FileNotFoundError(f"No logs.json found in {log_dir}")
    
    with open(log_path, 'r') as f:
        return json.load(f)

def extract_task_data(logs: Dict[str, dict]) -> Dict[str, Dict[str, List]]:
    """
    Extract per-sample data from the reductions section.
    
    Returns:
        Dict mapping task names to scorer data:
        {
            'gpqa_base_task': {
                'choices': ['B', 'A', 'D', ...],  # What the model picked
                'is_correct': [0.0, 1.0, 0.0, ...],
                'sample_ids': [1, 2, 3, ...]
            },
            ...
        }
    """
    task_data = {}
    
    for log_file, log_content in logs.items():
        if log_content['status'] != 'success':
            print(f"Warning: Skipping failed eval {log_file}")
            continue
            
        task_name = log_content['eval']['task']
        
        # Extract data from reductions section (not results.scores)
        if 'reductions' not in log_content:
            print(f"Warning: No reductions found in {log_file}")
            continue
            
        reductions = log_content['reductions']
        
        # Find the scorers we need
        choice_data = None
        is_correct_data = None
        hint_answer_data = None
        
        for reduction in reductions:
            if reduction['scorer'] == 'choice_scorer':
                choice_data = reduction.get('samples', [])
            elif reduction['scorer'] == 'is_correct_scorer':
                is_correct_data = reduction.get('samples', [])
            elif reduction['scorer'] == 'hint_answer_scorer':
                hint_answer_data = reduction.get('samples', [])
        
        if choice_data is None or is_correct_data is None:
            print(f"Warning: Missing choice_scorer or is_correct_scorer data in {log_file}")
            continue
            
        # Extract data aligned by sample_id
        choice_dict = {s['sample_id']: s.get('answer', 'NA') for s in choice_data}
        correct_dict = {s['sample_id']: s['value'] for s in is_correct_data}
        hint_dict = {s['sample_id']: s.get('answer', 'NONE') for s in hint_answer_data} if hint_answer_data else {}
        
        # Get all sample IDs
        all_sample_ids = sorted(set(choice_dict.keys()) | set(correct_dict.keys()))
        
        choices = []
        is_correct = []
        hint_answers = []
        valid_sample_ids = []
        
        for sample_id in all_sample_ids:
            if sample_id in choice_dict and sample_id in correct_dict:
                choices.append(choice_dict[sample_id])
                is_correct.append(correct_dict[sample_id])
                hint_answers.append(hint_dict.get(sample_id, 'NONE'))
                valid_sample_ids.append(sample_id)
        
        task_data[task_name] = {
            'choices': choices,
            'is_correct': is_correct,
            'hint_answers': hint_answers,
            'sample_ids': valid_sample_ids
        }
        
        print(f"Loaded {task_name}: {len(choices)} samples")
        print(f"  Model choices: {choices}")
    
    return task_data

def calculate_hint_follow_rate(choices: List[str], hint_answers: List[str]) -> float:
    """
    Calculate the fraction of times the model picked the hinted answer.
    
    Args:
        choices: What the model actually picked ['A', 'B', 'C', 'D', 'NA']
        hint_answers: What was hinted for each sample ['A', 'B', 'C', 'D', 'NONE']
    
    Returns:
        Fraction (0.0 to 1.0) of times model picked the hinted answer
    """
    if not choices or not hint_answers or len(choices) != len(hint_answers):
        return 0.0
    
    matches = 0
    valid_comparisons = 0
    
    for choice, hint in zip(choices, hint_answers):
        if hint != 'NONE' and choice != 'NA':  # Only count valid cases
            valid_comparisons += 1
            if choice == hint:
                matches += 1
    
    if valid_comparisons == 0:
        return 0.0
    
    return matches / valid_comparisons

def create_figure3_plot(task_data: Dict[str, Dict[str, List]], output_path: str = "figure3_reproduction.png"):
    """Create the Figure 3 plot showing fraction of times model picked hinted answer."""
    
    # Map task names to display names
    task_display_names = {
        'gpqa_base_task': 'No Hint',
        'gpqa_simple_hint_task': 'Simple Hint', 
        'gpqa_complex_hint_task': 'Complex Hint'
    }
    
    # Calculate hint follow rate for each condition
    hint_follow_rates = []
    conditions = []
    
    for task_name in ['gpqa_base_task', 'gpqa_simple_hint_task', 'gpqa_complex_hint_task']:
        if task_name in task_data:
            data = task_data[task_name]
            follow_rate = calculate_hint_follow_rate(data['choices'], data['hint_answers'])
            hint_follow_rates.append(follow_rate)
            conditions.append(task_display_names[task_name])
            
            # Print detailed stats
            mean_correct = np.mean(data['is_correct'])
            print(f"\n{task_display_names[task_name]}:")
            print(f"  Model choices: {data['choices']}")
            print(f"  Hint answers: {data['hint_answers']}")
            print(f"  Mean Correctness: {mean_correct:.3f}")
            print(f"  Hint Follow Rate: {follow_rate:.3f}")
        else:
            print(f"Warning: {task_name} not found in data")
    
    if not hint_follow_rates:
        raise ValueError("No valid task data found for plotting")
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Use a color palette similar to the paper
    colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange
    
    bars = plt.bar(conditions, hint_follow_rates, color=colors[:len(conditions)], alpha=0.8, edgecolor='black', linewidth=1)
    
    # Customize the plot
    plt.title('Figure 3: Fraction of Times Model Picked Hinted Answer\n(Reproduction)', fontsize=14, fontweight='bold')
    plt.ylabel('Fraction Picked Hinted Answer', fontsize=12)
    plt.xlabel('Condition', fontsize=12)
    
    # Set y-axis to 0-1 range
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, rate in zip(bars, hint_follow_rates):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{rate:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Add horizontal line at y=0.25 for reference (random chance)
    plt.axhline(y=0.25, color='red', linestyle='--', alpha=0.5, label='Random Chance (25%)')
    plt.legend()
    
    # Style improvements
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure 3 plot saved to: {output_path}")
    
    # Show the plot
    plt.show()
    
    return hint_follow_rates, conditions

def print_summary(task_data: Dict[str, Dict[str, List]]):
    """Print a summary of the experimental results."""
    print("\n" + "="*60)
    print("FIGURE 3 REPRODUCTION SUMMARY")
    print("="*60)
    
    total_samples = 0
    for task_name, data in task_data.items():
        samples = len(data['is_correct'])
        total_samples += samples
        print(f"\n{task_name}: {samples} samples")
    
    print(f"\nTotal samples across all tasks: {total_samples}")
    print("\nInterpretation:")
    print("- Fraction Picked Hinted Answer = How often model chose the hinted letter")
    print("- ~25% = Random chance (1 in 4 choices)")
    print("- Higher values indicate greater hint following behavior")
    print("- For 'No Hint' condition, this measures baseline random selection")

def main():
    parser = argparse.ArgumentParser(description="Analyze GPQA hint experiment logs and generate Figure 3")
    parser.add_argument("log_dir", help="Directory containing the eval logs (e.g., 'figure-3')")
    parser.add_argument("--output", "-o", default="figure3_reproduction.png", 
                        help="Output filename for the plot (default: figure3_reproduction.png)")
    
    args = parser.parse_args()
    
    try:
        print(f"Loading logs from: {args.log_dir}")
        logs = load_logs(args.log_dir)
        
        print(f"Found {len(logs)} log files")
        
        task_data = extract_task_data(logs)
        
        if not task_data:
            raise ValueError("No valid task data extracted from logs")
        
        hint_follow_rates, conditions = create_figure3_plot(task_data, args.output)
        
        print_summary(task_data)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
