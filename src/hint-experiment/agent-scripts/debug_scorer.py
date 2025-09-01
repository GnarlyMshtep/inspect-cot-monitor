#!/usr/bin/env python3
"""Debug the scorer behavior to understand why incorrect answers show green checkmarks."""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inspect_ai.scorer import Score, Target
from inspect_ai.solver import TaskState
from inspect_ai.model import ModelOutput

# Import our scorers
from gpqa_hint_task import choice_scorer, is_correct_scorer, extract_answer_from_response

async def test_scorer_behavior():
    """Test how our scorers behave with different inputs."""
    
    print("Testing scorer behavior...")
    print("=" * 50)
    
    # Create a mock task state
    class MockTaskState:
        def __init__(self, completion_text):
            self.output = ModelOutput(completion=completion_text)
    
    # Test cases: (model_response, target, expected_choice, should_be_correct)
    test_cases = [
        ("I think the answer is <answer>C</answer>", "A", "C", False),
        ("I think the answer is <answer>A</answer>", "A", "A", True),
        ("I think the answer is <answer>B</answer>", "D", "B", False),
        ("I'm not sure about this", "A", "NA", False),
        ("The answer is B", "B", "B", True),  # Test fallback extraction
    ]
    
    choice_scorer_func = choice_scorer()
    is_correct_scorer_func = is_correct_scorer()
    
    for i, (response, target, expected_choice, should_be_correct) in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"  Model Response: {response}")
        print(f"  Target: {target}")
        print(f"  Expected Choice: {expected_choice}")
        print(f"  Should Be Correct: {should_be_correct}")
        
        # Test answer extraction
        extracted = extract_answer_from_response(response)
        print(f"  Extracted Choice: {extracted}")
        
        # Test choice scorer
        state = MockTaskState(response)
        choice_score = await choice_scorer_func(state, target)
        print(f"  Choice Score: value={choice_score.value}, answer={choice_score.answer}")
        
        # Test correctness scorer  
        correct_score = await is_correct_scorer_func(state, target)
        print(f"  Correctness Score: value={correct_score.value}, answer={correct_score.answer}")
        
        # Verify expectations
        if extracted == expected_choice:
            print("  ✓ Choice extraction correct")
        else:
            print(f"  ✗ Choice extraction wrong: got {extracted}, expected {expected_choice}")
            
        if (correct_score.value == 1.0) == should_be_correct:
            print("  ✓ Correctness assessment correct")
        else:
            print(f"  ✗ Correctness assessment wrong: got {correct_score.value}, expected {1.0 if should_be_correct else 0.0}")

if __name__ == "__main__":
    asyncio.run(test_scorer_behavior())

