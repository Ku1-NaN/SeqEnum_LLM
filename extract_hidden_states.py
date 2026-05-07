"""
extract_hidden_states.py

This script processes a set of prompts through a Large Language Model (e.g., Llama-3) 
using the Hugging Face `transformers` library. It validates the output and extracts 
both the input and output hidden states (internal embeddings), saving them locally 
as PyTorch tensors for further analysis.

Author: Kuinan
Usage:
    Ensure you have set your Hugging Face token as an environment variable 
    or replace the token placeholder before running.
"""

import pickle
import torch
import os
import json
from tqdm.auto import tqdm
from huggingface_hub import login
from utils import extract_valid_answer, validate_answer # Ensure utils.py is in your working directory
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import io

# Function to save tensor locally
def save_tensor_locally(tensor, directory, filename):
    # Ensure directory exists
    os.makedirs(directory, exist_ok=True)
    
    # Save tensor to file
    filepath = os.path.join(directory, filename)
    torch.save(tensor, filepath)
    print(f"Saved tensor to {filepath}")

# Login to Hugging Face
# Recommend using environment variable for security: os.environ.get("HF_TOKEN")
login(token="YOUR_HUGGING_FACE_TOKEN_HERE") 

print(os.environ.get("CUDA_VISIBLE_DEVICES"))

model_id = "meta-llama/Llama-3.3-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16,
    output_hidden_states=True, 
    return_dict_in_generate=True,
    device_map="auto"
)
print('----device map----')
print(model.hf_device_map)

# Paths (Update these to match your local directory structure)
prompts_path = "./data/production_prompts.json"
input_hidden_states_dir = "./hidden_states/input"
output_hidden_states_dir = "./hidden_states/output"
results_path = './results/llama70b_results.pkl'
log_dir = './logs'
log_file = os.path.join(log_dir, f'non_valid_outputs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

# Create local directories for logs and results
os.makedirs(input_hidden_states_dir, exist_ok=True)
os.makedirs(output_hidden_states_dir, exist_ok=True)
os.makedirs(os.path.dirname(results_path), exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

with open(log_file, 'a') as f:
    f.write(f'currently working on:\n{prompts_path}\n')
    f.write('='*80 + '\n')
    
# Load prompts
with open(prompts_path, 'r') as f:
    prompts = json.load(f)

reprocess_id = []
# Load existing results if available
if os.path.exists(results_path):
    print('loading existing results')
    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    processed_id = [p['id'] for p in results if p['id'] not in reprocess_id]
    print(f'Num of processed_id: {len(processed_id)}')
else:
    results = []
    processed_id = []

# Maximum number of retries for a single prompt
MAX_RETRIES = 3

print(f'reprocess_id:{reprocess_id}')
print(f"valid_results: {len(results)}")

# Process prompts
for idx, prompt in enumerate(tqdm(prompts)):
    if prompt['id'] in processed_id and prompt['id'] not in reprocess_id:
        continue  # Skip already processed prompts
    
    if prompt['condition'] != 'explicit_counting':
        continue

    if prompt['task_type'] == 'production' and prompt['variation'] == 'non_uniform' and prompt['item_type'] == 'word':
        initial_max_new_tokens = 2048
    elif prompt['task_type'] == 'production' and prompt['item_type'] == 'letter':
        initial_max_new_tokens = 512
    else:
        initial_max_new_tokens = 1000
    
    max_new_tokens = initial_max_new_tokens
    token_limit_reached_count = 0
    first_failure = True
    idx = prompt['id']
    
    # Retry mechanism
    for attempt in range(MAX_RETRIES):
        messages = prompt['messages']
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize input
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            if attempt == 0:
                # Get input embeddings (hidden states)
                with torch.no_grad():
                    input_outputs = model(
                        input_ids=model_inputs.input_ids,
                        attention_mask=model_inputs.attention_mask,
                        output_hidden_states=True,
                        return_dict=True
                    )
                    input_hidden_states = input_outputs.hidden_states
                    
                    # Save input hidden states locally
                    input_blob_name = f"{idx}_{attempt}.pt"
                    save_tensor_locally(input_hidden_states, input_hidden_states_dir, input_blob_name)
            
            # Generate output
            generate_results = model.generate(
                **model_inputs, 
                max_new_tokens=max_new_tokens, 
                output_hidden_states=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Get output tokens and their decoded representations
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generate_results['sequences'])
            ]
            
            # Full output as a string
            full_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            
            # Token-by-token output (list of strings)
            token_by_token_output = [tokenizer.decode(token_id) for token_id in generated_ids[0]]
            
            # Get output hidden states
            output_hidden_states = generate_results['hidden_states']
            
            # Check if the output is valid
            if validate_answer(full_output):
                # Extract the validated answer
                validated_output = extract_valid_answer(full_output)
                
                # Update prompt with outputs
                prompt['outputs'] = full_output
                prompt['validated_outputs'] = validated_output
                prompt['token_by_token_outputs'] = token_by_token_output
                prompt['num_tries'] = attempt+1
                prompt['max_new_tokens_used'] = max_new_tokens
                results.append(prompt)
                
                # Save output hidden states locally
                output_blob_name = f"{idx}_{attempt}.pt"
                save_tensor_locally(output_hidden_states, output_hidden_states_dir, output_blob_name)
                
                # Save updated results incrementally
                with open(results_path, 'wb') as file:
                    pickle.dump(results, file)
                
                # Break the retry loop if successful
                break
            
            # Check if we reached the token limit (all tokens were used)
            reached_token_limit = len(generated_ids[0]) >= max_new_tokens
            
            # Log non-valid outputs to file
            with open(log_file, 'a') as f:
                f.write(f"Prompt ID: {prompt['id']}, Attempt: {attempt+1}\n")
                
                # Only write the full input prompt for the first failure
                if first_failure:
                    f.write(f"Input prompt:\n{text}\n\n")
                    first_failure = False
                
                f.write(f"Output:\n{full_output}\n")
                f.write(f"Max new tokens: {max_new_tokens}\n")
                f.write(f"Token limit reached: {reached_token_limit}\n")
                f.write("="*80 + "\n\n")
            
            # Print information about token limit
            if reached_token_limit:
                token_limit_reached_count += 1
                print(f"Token limit of {max_new_tokens} reached on attempt {attempt+1}.")
                
                # Only reduce max_new_tokens for the first two times we hit the limit
                if token_limit_reached_count <= 2:
                    max_new_tokens = max(max_new_tokens // 2, 300)  # Ensure we don't go too low
                    print(f"Reducing max_new_tokens to {max_new_tokens}")
                
                # If this is the third time hitting the token limit, make it the last attempt
                if token_limit_reached_count == 2:
                    print("Token limit reached ten times, skipping remaining attempts.")
                    attempt = MAX_RETRIES - 1  # This will make the next iteration the last one
            
            # If it's the last attempt and still not valid
            if attempt == MAX_RETRIES - 1:
                print(f"Failed to get a valid output for prompt {idx} after {MAX_RETRIES} attempts.")
                prompt['outputs'] = full_output
                prompt['validated_outputs'] = None
                prompt['token_by_token_outputs'] = token_by_token_output
                prompt['num_tries'] = attempt+1
                prompt['max_new_tokens_used'] = max_new_tokens
                results.append(prompt)
                
                # Save updated results incrementally
                with open(results_path, 'wb') as file:
                    pickle.dump(results, file)
                
                # Save failed output hidden states locally
                output_blob_name = f"{idx}_{attempt}_failed.pt"
                save_tensor_locally(output_hidden_states, output_hidden_states_dir, output_blob_name)
                break
            else:
                print(f'Retrying No.{attempt+2}...') 
                
        except Exception as e:
            print(f"Error on attempt {attempt+1}: {str(e)}")
            
            # Log the error to file
            with open(log_file, 'a') as f:
                f.write(f"Prompt ID: {prompt['id']}, Attempt: {attempt+1}\n")
                f.write(f"Error: {str(e)}\n")
                
                if first_failure:
                    f.write(f"Input prompt:\n{text if 'text' in locals() else 'Text not available'}\n\n")
                    first_failure = False
                
                if 'full_output' in locals():
                    f.write(f"Output:\n{full_output}\n")
                f.write("="*80 + "\n\n")
            
            continue