#!/usr/bin/env python3
"""Show examples of complex hints with random alphabet letters."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpqa_hint_task import generate_complex_hint, generate_simple_hint

print('Examples of complex hints with random alphabet letters:')
print('=' * 60)

for letter in ['A', 'B', 'C', 'D']:
    print(f'\nCorrect answer: {letter}')
    for hint_choice in ['A', 'B', 'C', 'D']:
        simple_hint = generate_simple_hint(letter, hint_choice)
        complex_hint = generate_complex_hint(letter, hint_choice)
        print(f'  Hint choice {hint_choice} ({"✓ correct" if hint_choice == letter else "✗ incorrect"}):')
        print(f'    Simple:  {simple_hint}')
        print(f'    Complex: {complex_hint}')
        
        # Verify the complex hint math
        import re
        pattern = r'\(([A-Z]) \+ ([A-Z])\) mod 4'
        match = re.search(pattern, complex_hint)
        if match:
            x_letter, y_letter = match.groups()
            x_num = ord(x_letter) - ord('A')
            y_num = ord(y_letter) - ord('A')
            result = (x_num + y_num) % 4
            letter_to_num = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
            expected = letter_to_num[hint_choice]  # Should match hint_choice, not correct answer
            check = "✓" if result == expected else "✗"
            print(f'    {check} Verification: ({x_letter}={x_num} + {y_letter}={y_num}) mod 4 = {result} (expected {expected})')
        print()
