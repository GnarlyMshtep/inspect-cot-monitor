#!/usr/bin/env python3
"""
Test script for the GPQA hint experiment implementation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpqa_hint_task import (
    load_gpqa_dataset, 
    create_samples, 
    extract_answer_from_response,
    generate_simple_hint,
    generate_complex_hint,
    shuffle_answers
)


def test_dataset_loading():
    """Test that we can load the GPQA dataset."""
    print("Testing dataset loading...")
    try:
        data = load_gpqa_dataset()
        print(f"✓ Loaded {len(data)} samples from GPQA dataset")
        
        # Check first sample structure
        if data:
            sample = data[0]
            required_keys = ["Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]
            for key in required_keys:
                if key not in sample:
                    print(f"✗ Missing key: {key}")
                    return False
            print("✓ Dataset has expected structure")
        return True
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return False


def test_answer_shuffling():
    """Test the answer shuffling functionality."""
    print("\nTesting answer shuffling...")
    try:
        question = "What is 2+2?"
        correct = "4"
        incorrect = ["3", "5", "6"]
        
        formatted_q, correct_label, mapping = shuffle_answers(question, correct, incorrect)
        
        print(f"✓ Question formatted successfully")
        print(f"✓ Correct answer is labeled as: {correct_label}")
        print(f"✓ Answer mapping: {mapping}")
        
        # Verify correct answer is in mapping
        if mapping[correct_label] != correct:
            print(f"✗ Correct answer mapping failed")
            return False
        print("✓ Correct answer properly mapped")
        return True
    except Exception as e:
        print(f"✗ Error in answer shuffling: {e}")
        return False


def test_hint_generation():
    """Test hint generation functions."""
    print("\nTesting hint generation...")
    try:
        # Test simple hint
        simple_hint = generate_simple_hint("B")
        print(f"✓ Simple hint: {simple_hint}")
        
        # Test complex hint
        complex_hint = generate_complex_hint("B")
        print(f"✓ Complex hint: {complex_hint}")
        
        # Verify complex hint works with the new format
        import re
        pattern = r'\(([A-Z]) \+ ([A-Z])\) mod 4'
        match = re.search(pattern, complex_hint)
        if match:
            x_letter, y_letter = match.groups()
            # Convert to numbers using full alphabet mapping and verify
            x_num = ord(x_letter) - ord('A')  # A=0, B=1, C=2, ...
            y_num = ord(y_letter) - ord('A')
            result = (x_num + y_num) % 4
            
            # Expected result for 'B' (we used 'B' as input)
            letter_to_num = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
            expected = letter_to_num['B']
            
            if result == expected:
                print("✓ Complex hint math is correct")
                print(f"  Verification: ({x_letter}={x_num} + {y_letter}={y_num}) mod 4 = {result} = B")
            else:
                print(f"✗ Complex hint math failed: {result} != {expected}")
                return False
        else:
            print("✗ Could not parse complex hint")
            return False
        
        # Check that hint mentions full alphabet mapping
        if "Z = 25" in complex_hint:
            print("✓ Complex hint mentions full alphabet mapping")
        else:
            print("✗ Complex hint should mention full alphabet mapping")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Error in hint generation: {e}")
        return False


def test_answer_extraction():
    """Test answer extraction from responses."""
    print("\nTesting answer extraction...")
    try:
        test_cases = [
            ("I think the answer is <answer>B</answer>", "B"),
            ("After careful consideration <answer>D</answer>", "D"),
            ("The correct choice is A", "A"),
            ("I believe it's C based on...", "C"),
            ("This is unclear", "NA"),
            ("<answer>X</answer>", "NA"),  # Invalid choice
        ]
        
        for response, expected in test_cases:
            result = extract_answer_from_response(response)
            if result == expected:
                print(f"✓ '{response[:30]}...' -> {result}")
            else:
                print(f"✗ '{response[:30]}...' -> {result} (expected {expected})")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Error in answer extraction: {e}")
        return False


def test_sample_creation():
    """Test sample creation for different hint types."""
    print("\nTesting sample creation...")
    try:
        # Test with small subset to avoid rate limits
        for hint_type in ["base", "simple", "complex"]:
            print(f"  Testing {hint_type} samples...")
            samples = create_samples(hint_type)
            
            if not samples:
                print(f"✗ No samples created for {hint_type}")
                return False
                
            print(f"✓ Created {len(samples)} samples for {hint_type}")
            
            # Check first sample
            sample = samples[0]
            if not hasattr(sample, 'input') or not hasattr(sample, 'target'):
                print(f"✗ Sample missing required attributes")
                return False
            
            # Check hint presence
            if hint_type == "base":
                if "Hint:" in sample.input:
                    print(f"✗ Base sample should not have hint")
                    return False
                if "You are not given a hint" not in sample.input:
                    print(f"✗ Base sample should mention no hint")
                    return False
            else:
                if "Hint:" not in sample.input:
                    print(f"✗ {hint_type} sample should have hint")
                    return False
                    
            # Check complex hint format
            if hint_type == "complex":
                if "Z = 25" not in sample.input:
                    print(f"✗ Complex hint should mention full alphabet mapping")
                    return False
            
            print(f"✓ {hint_type} samples have correct hint structure")
        
        return True
    except Exception as e:
        print(f"✗ Error in sample creation: {e}")
        return False


def main():
    """Run all tests."""
    print("=== GPQA Hint Experiment Test Suite ===\n")
    
    tests = [
        test_dataset_loading,
        test_answer_shuffling, 
        test_hint_generation,
        test_answer_extraction,
        test_sample_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"=== Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("🎉 All tests passed! The implementation looks good.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

