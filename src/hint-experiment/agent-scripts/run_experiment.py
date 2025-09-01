#!/usr/bin/env python3
"""
Example script to run the GPQA hint experiment.

This script demonstrates how to run the tasks for reproducing Section 3.1 of the paper.
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inspect_ai import eval
from gpqa_hint_task import gpqa_base_task, gpqa_simple_hint_task, gpqa_complex_hint_task


async def run_experiment():
    """Run the full GPQA hint experiment."""
    print("Starting GPQA Hint Experiment - Section 3.1 Reproduction")
    print("=" * 60)
    
    # Configuration
    model = "openai/gpt-4o"
    epochs = 10  # As specified in the requirements
    
    # Run each experiment condition
    experiments = [
        ("GPQA Base (No Hint)", gpqa_base_task),
        ("GPQA Simple Hint", gpqa_simple_hint_task), 
        ("GPQA Complex Hint", gpqa_complex_hint_task)
    ]
    
    results = {}
    
    for name, task_func in experiments:
        print(f"\n🔬 Running: {name}")
        print("-" * 40)
        
        try:
            # Create and run the task
            task = task_func(epochs=epochs)
            result = await eval(task, model=model)
            
            results[name] = result
            print(f"✅ Completed: {name}")
            
            # Print some basic stats
            if hasattr(result, 'metrics'):
                for metric_name, metric_value in result.metrics.items():
                    print(f"   {metric_name}: {metric_value}")
            
        except Exception as e:
            print(f"❌ Error running {name}: {e}")
            results[name] = None
    
    print("\n" + "=" * 60)
    print("Experiment Summary:")
    for name, result in results.items():
        status = "✅ Success" if result else "❌ Failed"
        print(f"  {name}: {status}")
    
    print("\nResults are saved in the logs/ directory for detailed analysis.")
    return results


def run_single_experiment(hint_type: str = "base", limit: int = 5):
    """
    Run a single experiment with a limited number of samples for testing.
    
    Args:
        hint_type: "base", "simple", or "complex"
        limit: Number of samples to run (for testing)
    """
    print(f"Running single experiment: {hint_type} (limited to {limit} samples)")
    
    # Import here to avoid circular imports
    from gpqa_hint_task import create_samples
    from inspect_ai.dataset import MemoryDataset
    from inspect_ai import Task
    from gpqa_hint_task import choice_scorer, is_correct_scorer, hint_usage_scorer
    
    # Create limited samples
    all_samples = create_samples(hint_type)
    limited_samples = all_samples[:limit]
    
    dataset = MemoryDataset(limited_samples)
    
    task = Task(
        dataset=dataset,
        scorer=[choice_scorer(), is_correct_scorer(), hint_usage_scorer()],
        epochs=1  # Just 1 epoch for testing
    )
    
    # Run synchronously for testing
    import inspect_ai
    result = inspect_ai.eval(task, model="openai/gpt-4o-mini")
    return result


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run single experiment with limited samples
        hint_type = sys.argv[1] if sys.argv[1] in ["base", "simple", "complex"] else "base"
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 2
        print(f"Running test experiment: {hint_type} with {limit} samples")
        result = run_single_experiment(hint_type, limit)
        print("Test completed!")
    else:
        # Show usage
        print("GPQA Hint Experiment Runner")
        print("=" * 30)
        print("\nUsage:")
        print("  python run_experiment.py [hint_type] [num_samples]")
        print("\nExamples:")
        print("  python run_experiment.py base 2      # Test 2 base samples")
        print("  python run_experiment.py simple 3    # Test 3 simple hint samples") 
        print("  python run_experiment.py complex 2   # Test 2 complex hint samples")
        print("\nFor full experiment, use inspect eval commands:")
        print("  inspect eval ../gpqa_hint_task.py@gpqa_base_task --model openai/gpt-4o")
        print("  inspect eval ../gpqa_hint_task.py@gpqa_simple_hint_task --model openai/gpt-4o")
        print("  inspect eval ../gpqa_hint_task.py@gpqa_complex_hint_task --model openai/gpt-4o")
        print("\nRunning a quick test with 2 base samples...")
        result = run_single_experiment("base", 2)
        print("Quick test completed!")

