#!/usr/bin/env python3
"""Debug the is_correct_scorer to understand why it's failing when target=B and answer=B."""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inspect_ai.scorer import Score, Target
from inspect_ai.solver import TaskState
from inspect_ai.model import ModelOutput
from inspect_ai.dataset import Sample

# Import our functions
from gpqa_hint_task import is_correct_scorer, extract_answer_from_response

async def test_correctness_scorer():
    """Test the correctness scorer with the specific failing case."""
    
    print("Testing is_correct_scorer...")
    print("=" * 50)
    
    # Create a mock task state
    class MockTaskState:
        def __init__(self, completion_text):
            self.output = ModelOutput(completion=completion_text)
    
    # Test the specific failing case
    test_cases = [
        {
            "name": "Target B, Answer B (should be correct)",
            "response": "I think the answer is <answer>B</answer>",
            "target": "B",
            "expected": 1.0
        },
        {
            "name": "Target B, Answer A (should be incorrect)", 
            "response": "I think the answer is <answer>A</answer>",
            "target": "B",
            "expected": 0.0
        },
        {
            "name": "Target A, Answer A (should be correct)",
            "response": "I think the answer is <answer>A</answer>", 
            "target": "A",
            "expected": 1.0
        },
        {
            "name": "No answer extracted",
            "response": "I'm not sure about this",
            "target": "B", 
            "expected": 0.0
        }
    ]
    
    is_correct_scorer_func = is_correct_scorer()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['name']}")
        print(f"  Response: {test_case['response']}")
        print(f"  Target: {test_case['target']} (type: {type(test_case['target'])})")
        print(f"  Expected: {test_case['expected']}")
        
        # Test answer extraction first
        extracted = extract_answer_from_response(test_case['response'])
        print(f"  Extracted: '{extracted}' (type: {type(extracted)})")
        
        # Test the scorer with proper Target object
        state = MockTaskState(test_case['response'])
        # Create a proper Target object from a Sample
        sample = Sample(input="test", target=test_case['target'])
        target_obj = sample.target
        score = await is_correct_scorer_func(state, target_obj)
        
        print(f"  Score: value={score.value}, answer={score.answer}")
        if hasattr(score, 'explanation') and score.explanation:
            print(f"  Explanation: {score.explanation}")
        
        # Check the comparison logic manually with Target object
        target_text = target_obj.text if hasattr(target_obj, 'text') else str(target_obj)
        is_correct_manual = extracted == target_text and extracted != 'NA'
        print(f"  Manual check: '{extracted}' == '{target_text}' and '{extracted}' != 'NA' = {is_correct_manual}")
        print(f"  Target object type: {type(target_obj)}, has text property: {hasattr(target_obj, 'text')}")
        
        # Verify result
        if score.value == test_case['expected']:
            print("  ✓ PASS")
        else:
            print(f"  ✗ FAIL: got {score.value}, expected {test_case['expected']}")
            
        print()

if __name__ == "__main__":
    asyncio.run(test_correctness_scorer())
