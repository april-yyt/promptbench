import os
import sys
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import promptbench as pb
from promptbench.models import LLMModel
from promptbench.prompt_attack import Attack

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

# Define models to evaluate
models = [
    "meta-llama/Llama-2-7b-hf",
    "mistralai/Mixtral-8x7B-v0.1"
]

# Define attacks
attacks = ['textbugger', 'deepwordbug', 'textfooler', 'bertattack', 'checklist', 'stresstest', 'semantic']

# Create dataset
dataset = pb.DatasetLoader.load_dataset("sst2")
dataset = dataset[:100]  # Limit to 100 samples for faster evaluation

# Create prompt
prompt = "As a sentiment classifier, determine whether the following text is 'positive' or 'negative'. Please classify: \nQuestion: {content}\nAnswer:"

# Define projection function
def proj_func(pred):
    mapping = {
        "positive": 1,
        "negative": 0
    }
    return mapping.get(pred.lower().strip(), -1)

# Define evaluation function
def eval_func(prompt, dataset, model):
    preds = []
    labels = []
    for d in dataset:
        input_text = pb.InputProcess.basic_format(prompt, d)
        raw_output = model(input_text)
        output = pb.OutputProcess.cls(raw_output, proj_func)
        preds.append(output)
        labels.append(d["label"])
    return pb.Eval.compute_cls_accuracy(preds, labels)

# Define unmodifiable words
unmodifiable_words = ["positive", "negative", "content"]

# Main evaluation function
def evaluate_model(model_name):
    logging.info(f"Evaluating model: {model_name}")
    
    # Load model
    model = LLMModel(model=model_name)
    
    results = {}
    
    # Evaluate each attack
    for attack_name in attacks:
        logging.info(f"Running attack: {attack_name}")
        attack = Attack(model, attack_name, dataset, prompt, eval_func, unmodifiable_words, verbose=True)
        result = attack.attack()
        results[attack_name] = result
        logging.info(f"Attack {attack_name} result: {result}")
    
    return results

# Main execution
if __name__ == "__main__":
    for model_name in models:
        try:
            results = evaluate_model(model_name)
            logging.info(f"Results for {model_name}:")
            for attack, result in results.items():
                logging.info(f"{attack}: {result}")
        except Exception as e:
            logging.error(f"Error evaluating {model_name}: {str(e)}")