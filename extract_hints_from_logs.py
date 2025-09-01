#!/usr/bin/env python3
"""
Extract hint information from existing eval logs by parsing the sample inputs.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess

def load_eval_file(eval_path: str) -> dict:
    """Load an eval file using inspect view command."""
    try:
        result = subprocess.run(['uv', 'run', 'inspect', 'view', eval_path, '--format', 'json'], 
                              capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except Exception as e:
        print(f"Error loading {eval_path}: {e}")
        return {}

def extract_hint_from_input(input_text: str, task_type: str) -> str:
    """Extract what answer was hinted from the input text."""
    
    if "Carnegie Mellon professor says the answer is" in input_text:
        # Simple hint format
        match = re.search(r"says the answer is ([A-D])", input_text)
        if match:
            return match.group(1)
    
    elif "mod 4" in input_text and "mapping A = 0" in input_text:
        # Complex hint format - need to solve the equation
        # Extract X and Y from the hint
        match = re.search(r"The correct answer is \(([A-Z]) \+ ([A-Z])\) mod 4", input_text)
        if match:
            x_letter, y_letter = match.group(1), match.group(2)
            x_num = ord(x_letter) - ord('A')
            y_num = ord(y_letter) - ord('A')
            result_num = (x_num + y_num) % 4
            result_letter = chr(ord('A') + result_num)
            return result_letter
    
    return "NONE"  # No hint or base task

def analyze_eval_file(eval_path: str) -> Dict[str, List]:
    """Analyze a single eval file and return choice/hint data."""
    print(f"Analyzing {eval_path}")
    
    data = load_eval_file(eval_path)
    if not data:
        return {}
    
    task_name = data.get('eval', {}).get('task', 'unknown')
    samples = data.get('samples', [])
    
    choices = []
    hint_answers = []
    sample_ids = []
    is_correct = []
    
    for sample in samples:
        sample_id = sample.get('id')
        input_text = sample.get('input', '')
        
        # Extract model choice from scores
        model_choice = 'NA'
        correctness = 0.0
        
        # Look for choice_scorer and is_correct_scorer results
        for epoch in sample.get('epochs', []):
            scores = epoch.get('scores', {})
            
            if 'choice_scorer' in scores:
                choice_score = scores['choice_scorer']
                if isinstance(choice_score, dict) and 'answer' in choice_score:
                    model_choice = choice_score['answer']
                elif hasattr(choice_score, 'answer'):
                    model_choice = choice_score.answer
            
            if 'is_correct_scorer' in scores:
                correct_score = scores['is_correct_scorer']
                if isinstance(correct_score, dict) and 'value' in correct_score:
                    correctness = correct_score['value']
                elif hasattr(correct_score, 'value'):
                    correctness = correct_score.value
        
        # Extract hint answer
        hint_answer = extract_hint_from_input(input_text, task_name)
        
        choices.append(model_choice)
        hint_answers.append(hint_answer)
        sample_ids.append(sample_id)
        is_correct.append(correctness)
        
        print(f"  Sample {sample_id}: choice={model_choice}, hint={hint_answer}, correct={correctness}")
    
    return {
        'task_name': task_name,
        'choices': choices,
        'hint_answers': hint_answers,
        'sample_ids': sample_ids,
        'is_correct': is_correct
    }

def main():
    """Analyze all eval files in figure-3 directory."""
    figure3_dir = Path('figure-3')
    
    if not figure3_dir.exists():
        print("Error: figure-3 directory not found")
        return
    
    eval_files = list(figure3_dir.glob('*.eval'))
    
    if not eval_files:
        print("Error: No .eval files found in figure-3 directory")
        return
    
    all_data = {}
    
    for eval_file in eval_files:
        data = analyze_eval_file(str(eval_file))
        if data and data['task_name']:
            all_data[data['task_name']] = data
    
    # Print summary
    print("\n" + "="*60)
    print("HINT EXTRACTION SUMMARY")
    print("="*60)
    
    for task_name, data in all_data.items():
        choices = data['choices']
        hints = data['hint_answers']
        
        # Calculate hint follow rate
        matches = 0
        valid_comparisons = 0
        
        for choice, hint in zip(choices, hints):
            if hint != 'NONE' and choice != 'NA':
                valid_comparisons += 1
                if choice == hint:
                    matches += 1
        
        follow_rate = matches / valid_comparisons if valid_comparisons > 0 else 0.0
        mean_correct = sum(data['is_correct']) / len(data['is_correct']) if data['is_correct'] else 0.0
        
        print(f"\n{task_name}:")
        print(f"  Choices: {choices}")
        print(f"  Hints: {hints}")
        print(f"  Hint Follow Rate: {follow_rate:.3f}")
        print(f"  Mean Correctness: {mean_correct:.3f}")
    
    # Save results for plotting
    with open('extracted_hint_data.json', 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"\nData saved to extracted_hint_data.json")

if __name__ == "__main__":
    main()

