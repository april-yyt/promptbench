import os
import sys
import logging
import json
from datetime import datetime
import torch
import promptbench as pb
from promptbench.models import LLMModel
from promptbench.prompt_attack import Attack

torch.cuda.empty_cache()

# Set up logging
logging.basicConfig(
    level=print,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('eval.log')
    ]
)

# Check GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Define models to evaluate 
MODELS = [
    "llama2-7b-chat"
    # "mistralai/Mixtral-8x7B-v0.1"
]

# Define attacks from paper
ATTACKS = [
    'textbugger',    # Character-level
    'deepwordbug',   # Character-level  
    'textfooler',    # Word-level
    'bertattack',    # Word-level
    'checklist',     # Sentence-level
    'stresstest',    # Sentence-level
    # 'semantic'       # Semantic-level
]

# Projection function for sentiment analysis
def proj_func(pred):
    mapping = {
        "positive": 1,
        "negative": 0,
        "ositive": 1,
        "egative": 0
    }
    return mapping.get(pred.lower().strip(), -1)

def eval_func(prompt, dataset, model):
    preds = []
    labels = []
    for d in dataset:
        # Use InputProcess.basic_format to properly handle content replacement
        input_text = pb.InputProcess.basic_format(prompt, d)
        
        # Debug print first few examples
        if len(preds) < 2:
            print(f"Sample input text: {input_text}")
            
        raw_output = model(input_text)
        print(f"raw output is : {raw_output}")
        output = pb.OutputProcess.cls(raw_output, proj_func)
        preds.append(output)
        labels.append(d["label"])
    
    acc = pb.Eval.compute_cls_accuracy(preds, labels)
    return acc



def evaluate_model(model_name, dataset):
    print(f"Evaluating model: {model_name}")
    
    model = pb.LLMModel(
        model=model_name,
        max_new_tokens=10,
        temperature=0.0001,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Define prompt format
    prompt = "As a sentiment classifier, determine whether the following text is 'positive' or 'negative', please output strictly only 'positive' or 'negative' as text. Please classify: \nQuestion: {content}\nAnswer:"

    # Define unmodifiable words - match exactly what should be preserved
    unmodifiable_words = ["positive", "negative", "content"]

    results = {}

    # Test a single sample first to verify formatting
    test_sample = dataset[0] 
    test_input = pb.InputProcess.basic_format(prompt, test_sample)
    print(f"Test input format: {test_input}")

    # Try all attacks
    for attack_name in ATTACKS:
        attack = Attack(
            model,
            attack_name,
            dataset,
            prompt,
            eval_func,
            unmodifiable_words,
            verbose=True
        )
        
        try:
            result = attack.attack()
            print(f"Attack {attack_name} result: {result}")
            results[attack_name] = result
        except Exception as e:
            logging.error(f"Attack {attack_name} failed: {e}", exc_info=True)
            results[attack_name] = None

    return results


if __name__ == "__main__":
    # Set debug logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('eval_debug.log')
        ]
    )
    
    # Define results directory
    results_dir = "evaluation_results"
    os.makedirs(results_dir, exist_ok=True)

    # Load and validate dataset
    dataset = pb.DatasetLoader.load_dataset("sst2")
    dataset = dataset[:100]

    # Validate dataset structure
    print("Dataset sample review:")
    for i, sample in enumerate(dataset[:3]):
        print(f"Sample {i}:")
        print(f"Content: {sample.get('content', 'MISSING')}")
        print(f"Label: {sample.get('label', 'MISSING')}")

    # Define test prompt
    test_prompt = "As a sentiment classifier, determine whether the following text is 'positive' or 'negative'. Please classify: \nQuestion: {content}\nAnswer:"
    
    # Test formatting
    test_sample = dataset[0]
    print(f"test sample: \n {test_sample}")
    test_input = test_prompt.replace("{content}", test_sample["content"])
    print(f"Test prompt formatting:")
    print(f"Original: {test_prompt}")
    print(f"Formatted: {test_input}")
    
    all_results = {}
    
    # Evaluate each model
    for model_name in MODELS:
        try:
            model_results = evaluate_model(model_name, dataset)
            if model_results:
                all_results[model_name] = model_results
                
                print(f"Results for {model_name}:")
                for attack, result in model_results.items():
                    print(f"{attack}: {result}")
            else:
                logging.error(f"No results obtained for {model_name}")
                
        except Exception as e:
            logging.error(f"Error evaluating {model_name}: {str(e)}", exc_info=True)

    # Save complete results after all tasks are done
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_results_file = os.path.join(results_dir, f"complete_results_{timestamp}.json")
    with open(final_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"All results saved to {final_results_file}")