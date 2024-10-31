import json
import re

def normalize_number(value):
    if value is None:
        return None
    
    # Convert to string if not already
    value = str(value)
    
    # Remove any surrounding quotes or whitespace
    value = value.strip().strip("'").strip('"')
    
    # Remove currency symbols and any leading/trailing whitespace
    value = value.replace('$', '').strip()
    
    # Remove commas from numbers
    value = value.replace(',', '')
    
    # Try to extract just the number if there are units or other text
    # This will match both integers and decimal numbers
    match = re.match(r'^(-?\d+\.?\d*)(?:\s*(?:cubic\s+inches|epochs|units?|dollars?|[A-Za-z\s]+)?)?$', value)
    if match:
        number = match.group(1)
        # Convert to integer if it's a whole number
        if '.' in number:
            # Remove trailing zeros and decimal point if it's a whole number
            number = number.rstrip('0').rstrip('.')
        return number
    
    return value

def process_results_file(input_file, output_file):
    # Read the input file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    total_evaluated = 0
    total_correct_before = 0
    total_correct_after = 0
    changed_answers = []

    # Process each method's results
    for method, method_data in data.items():
        if 'results' in method_data:
            for item in method_data['results']:
                total_evaluated += 1
                
                # Store original values
                original_predicted = item['predicted_answer']
                original_is_correct = item['is_correct']
                if original_is_correct:
                    total_correct_before += 1

                # Normalize both predicted and correct answers
                normalized_predicted = normalize_number(item['predicted_answer'])
                normalized_correct = normalize_number(item['correct_answer'])
                
                # Update the prediction and correctness
                item['predicted_answer'] = normalized_predicted
                item['is_correct'] = normalized_predicted == normalized_correct
                
                if item['is_correct']:
                    total_correct_after += 1

                # Record changes in evaluation
                if original_is_correct != item['is_correct'] or original_predicted != normalized_predicted:
                    changed_answers.append({
                        'question': item['question'],
                        'correct_answer': item['correct_answer'],
                        'original_predicted': original_predicted,
                        'normalized_predicted': normalized_predicted,
                        'original_is_correct': original_is_correct,
                        'new_is_correct': item['is_correct']
                    })

            # Update accuracy
            method_data['accuracy'] = sum(r['is_correct'] for r in method_data['results']) / len(method_data['results'])

    # Write the corrected results
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    # Print summary
    print(f"\nProcessing Summary:")
    print(f"Total items evaluated: {total_evaluated}")
    print(f"Correct before normalization: {total_correct_before} ({(total_correct_before/total_evaluated)*100:.2f}%)")
    print(f"Correct after normalization: {total_correct_after} ({(total_correct_after/total_evaluated)*100:.2f}%)")
    print(f"Number of answers changed: {len(changed_answers)}")
    
    # Write detailed changes to a separate file
    changes_file = output_file.replace('.json', '_changes.json')
    with open(changes_file, 'w') as f:
        json.dump({
            'summary': {
                'total_evaluated': total_evaluated,
                'correct_before': total_correct_before,
                'correct_after': total_correct_after,
                'num_changes': len(changed_answers)
            },
            'changed_answers': changed_answers
        }, f, indent=2)
    
    print(f"\nDetailed changes have been written to: {changes_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Correct answer parsing in results file')
    parser.add_argument('--input_file', help='Path to the input JSON results file')
    parser.add_argument('--output_file', help='Path to save the corrected JSON results file')
    
    args = parser.parse_args()
    
    process_results_file(args.input_file, args.output_file)