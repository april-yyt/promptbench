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
from langchain_ollama.llms import OllamaLLM

# Constants
# MODEL_NAME = "mistralai/Mixtral-8x7B-v0.1"
MODEL_NAME = "open-mixtral-8x7b"
DATASET_NAME = "gsm8k"
DATASET_SPLIT = "main"
CHECKPOINT_DIR = "checkpoints"
RESULTS_FILE = "1111_results_gsm8k.json"
MAX_NEW_TOKENS = 1024

class MistralModel:
    def __init__(self, use_api=True, use_ollama=False):
        self.use_api = use_api
        self.use_ollama = use_ollama
        
        if use_ollama:
            print("Using Ollama...")
            self.model = OllamaLLM(model="mixtral")
            # test the model
            print(self.model.invoke("What is the capital of France?"))
        elif not use_api:
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
        if self.use_ollama:
            response = self.model.invoke(prompt)
            print(f"RAW RESPONSE: {response}")
            return response
        elif not self.use_api:
            return self.model(prompt)
        else:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.invoke(messages)
            print(f"RAW RESPONSE: {response}")
            return response.content

    def convert_text_to_prompt(self, text, role): # adding role here but would not be used unless for gpt models
        return str(text) + '\n' 
        # text_content = str(text)
        # return f"{text_content}\n\nPlease provide the final NUMERICAL answer after '#### Answer:'."

    def concat_prompts(self, prompts):
        # Initialize an empty string to hold all prompts
        all_prompts = ""

        # Iterate over each keyword argument
        for arg in prompts:
            # Check if the argument is a string, and if so, add it to the list
            if isinstance(arg, str):
                all_prompts = all_prompts + '\n' + arg
            else:
                raise ValueError("All arguments must be strings.")

        return all_prompts
        # return pb_models.concat_prompts(prompts)
        # return "\n".join(prompts)


def load_dataset():
    print("Loading dataset...")
    return pb.DatasetLoader.load_dataset(DATASET_NAME)

def extract_final_answer(text):
    try:
        # First try to find answer with ## marker
        marker = "##"
        marker_index = text.find(marker)
        if marker_index != -1:
            after_marker = text[marker_index + len(marker):].strip()
            first_token = after_marker.split()[0]
            number = ''.join(filter(str.isdigit, first_token))
            if number:
                return number

        # If ## marker not found, try "The answer is X" format
        answer_phrase = "The answer is"
        answer_index = text.lower().find(answer_phrase.lower())
        if answer_index != -1:
            after_phrase = text[answer_index + len(answer_phrase):].strip()
            # Extract first number found after "The answer is"
            number = ''.join(filter(str.isdigit, after_phrase.split()[0]))
            if number:
                return number

        print(f"No answer format found in response: {text}")
        return None
    except Exception as e:
        print(f"Error extracting answer: {str(e)}")
        return None

def evaluate_model(model, dataset, method, num_samples):
    results = []
    correct = 0
    total = 0
    max_retries = 20  # Maximum number of retries
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
            predicted_answer = None
            for attempt in range(max_retries):
                try:
                    # Get response from model
                    if method == "Base":
                        prompt = model.convert_text_to_prompt(question)
                        response = model(prompt)
                    elif method == "ZSCoT":
                        cot_method = ZSCoT(dataset_name=DATASET_NAME, output_range="arabic numerals", verbose=True)
                        response = cot_method.query(question, model)
                        print(f"ZSCoT response: {response}\n\n")
                    elif method == "CoT":
                        cot_method = CoT(dataset_name=DATASET_NAME, output_range="arabic numerals", verbose=True)
                        response = cot_method.query(question, model)
                        print(f"COT response: {response}\n\n")
                    else:
                        raise ValueError(f"Unknown method: {method}")
                    
                    # Try to extract answer from this response
                    predicted_answer = extract_final_answer(response)
                    if predicted_answer is not None:
                        print(f"Successfully extracted answer on attempt {attempt + 1}")
                        break
                    else:
                        print(f"Failed to extract answer on attempt {attempt + 1}/{max_retries}")
                        if attempt < max_retries - 1:
                            print("Retrying with new model inference...")
                            continue
                        
                except Exception as e:
                    if "Requests rate limit exceeded" in str(e):
                        wait_time = base_wait_time * (2 ** attempt)
                        print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        print(f"Error on attempt {attempt + 1}: {str(e)}")
                    if attempt < max_retries - 1:
                        print("Retrying...")
                    continue
                        
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
    parser.add_argument("--use_ollama", action="store_true", help="Use Ollama local deployment")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate (default: all)")
    args = parser.parse_args()
    
    if args.use_api and args.use_ollama:
        raise ValueError("Cannot use both API and Ollama at the same time")
    
    if args.use_api:
        if "MISTRAL_API_KEY" not in os.environ:
            os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter your Mistral API key: ")

    model = MistralModel(use_api=args.use_api, use_ollama=args.use_ollama)
    dataset = load_dataset()
            
    num_samples = args.num_samples if args.num_samples is not None else len(dataset)
    print(f"Evaluating on {num_samples} samples")
        
    methods = ["CoT"]
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