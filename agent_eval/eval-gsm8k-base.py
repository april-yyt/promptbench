import os
import json
import argparse
import time
import getpass
from tqdm import tqdm
import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets
from promptbench.prompt_engineering.chain_of_thought import ZSCoT, CoT
from langchain_mistralai import ChatMistralAI
import promptbench as pb
from promptbench.prompts.method_oriented import get_prompt

# Constants
# MODEL_NAME = "mistralai/Mixtral-8x7B-v0.1"
MODEL_NAME = "open-mixtral-8x7b"
DATASET_NAME = "gsm8k"
DATASET_SPLIT = "main"
CHECKPOINT_DIR = "checkpoints"
RESULTS_FILE = "test_results_gsm8k.json"
MAX_NEW_TOKENS = 1024

class MistralModel:
    def __init__(self, use_api=True):
        self.use_api = use_api
        if not use_api:
            print("Loading model using PromptBench...")
            self.model = LLMModel(
                model=MODEL_NAME,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.1,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            print("Using Mistral API...")
            self.llm = ChatMistralAI(
                model=MODEL_NAME,
                temperature=0,
                max_tokens=MAX_NEW_TOKENS,
                mistral_api_key=os.getenv('MISTRAL_API_KEY')
            )

    def __call__(self, prompt):
        if not self.use_api:
            return self.model(prompt)
        else:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.invoke(messages)
            print(f"RAW RESPONSE: {response}")
            return response.content

    def convert_text_to_prompt(self, text):
        return f"{text}\n\nPlease provide the final NUMERICAL answer after '#### Answer:'."

    def concat_prompts(self, prompts):
        return "\n".join(prompts)


def load_dataset():
    print("Loading dataset...")
    return pb.DatasetLoader.load_dataset(DATASET_NAME)

def extract_final_answer(text):
    try:
        # Define the special marker
        marker = "#### Answer:"
        # Find the position of the marker
        marker_index = text.find(marker)
        if marker_index != -1:
            # Extract everything after the marker
            answer = text[marker_index + len(marker):].strip()
            # Optionally, take only the first line in case the model provides extra content
            answer = answer.splitlines()[0]
            return answer.strip()
        else:
            print(f"Special marker not found in response: {text}")
            return None
    except Exception as e:
        print(f"Error extracting answer: {str(e)}")
        return None

def evaluate_model(model, dataset, method, num_samples):
    results = []
    correct = 0
    total = 0
    max_retries = 5  # Maximum number of retries
    base_wait_time = 2  # Base wait time in seconds
    
    for item in tqdm(dataset[:num_samples], desc=f"Evaluating {method}"):
        try:
            # Check if item is a dictionary
            if not isinstance(item, dict):
                print(f"Warning: Skipping invalid item: {item}")
                continue
            
            question = item.get('content')
            answer = item.get('label')
            
            if question is None or answer is None:
                print(f"Warning: Skipping item with missing question or answer: {item}")
                continue
            
            # Retry logic for rate limit errors
            for attempt in range(max_retries):
                try:
                    # Generate prompt based on the method
                    if method == "Base":
                        prompt = model.convert_text_to_prompt(question)
                        # Generate response
                        response = model(prompt)
                        break
                    elif method == "ZSCoT":
                        cot_method = ZSCoT(dataset_name=DATASET_NAME, output_range="a number", verbose=False)
                        response = cot_method.query(question, model)
                        print(f"ZSCoT response: {response}\n\n")
                        break
                    elif method == "CoT":
                        cot_method = CoT(dataset_name=DATASET_NAME, output_range="a number", verbose=False)
                        response = cot_method.query(question, model)
                        print(f"COT response: {response}\n\n")
                        break
                    else:
                        raise ValueError(f"Unknown method: {method}")
                except Exception as e:
                    if "Requests rate limit exceeded" in str(e):
                        wait_time = base_wait_time * (2 ** attempt)
                        print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        raise  # Re-raise if it's not a rate limit error


            # Extract final answer
            predicted_answer = extract_final_answer(response)

            # Compare predicted answer with the correct answer
            is_correct = predicted_answer == answer
            if is_correct:
                correct += 1

            # Store result
            results.append({
                "question": question,
                "correct_answer": answer,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "full_response": response
            })

            total += 1

            # Save checkpoint every 100 items
            if total % 100 == 0:
                save_checkpoint(results, correct, total, method)

        except Exception as e:
            print(f"Error processing item: {item}")
            print(f"Error details: {str(e)}")
            continue
    
    return results, correct, total

def save_checkpoint(results, correct, total, method):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"checkpoint_{method}_{total}.json")
    with open(checkpoint_file, 'w') as f:
        json.dump({
            "results": results,
            "correct": correct,
            "total": total,
            "accuracy": correct / total
        }, f, indent=2)
    print(f"Checkpoint saved: {checkpoint_file}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate on GSM8K dataset")
    parser.add_argument("--use_api", action="store_true", help="Use OpenAI API instead of local Mistral model")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate (default: all)")
    args = parser.parse_args()

    model = MistralModel(use_api=args.use_api)
    dataset = load_dataset()
    
    if args.use_api:
        if "MISTRAL_API_KEY" not in os.environ:
            os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter your Mistral API key: ")
            
    num_samples = args.num_samples if args.num_samples is not None else len(dataset)
    print(f"Evaluating on {num_samples} samples")
        
    methods = ["ZSCoT", "CoT"]
    final_results = {}

    for method in methods:
        print(f"\nEvaluating {method} method...")
        results, correct, total = evaluate_model(model, dataset, method, num_samples)
        
        # Add a check before calculating accuracy
        if total == 0:
            print("Warning: No valid items were processed. Unable to calculate accuracy.")
            accuracy = 0  
        else:
            accuracy = correct / total
        
        print(f"{method} Accuracy: {accuracy:.2%}")
        
        final_results[method] = {
            "results": results,
            "accuracy": accuracy
        }

    with open(RESULTS_FILE, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()