# LLM Hidden State Extractor

This script utilizes the Hugging Face `transformers` library to process a batch of prompts through an LLM and collect the model's internal hidden states (embeddings) during inference.

## Purpose

The primary goal of this tool is to extract and save the layer-by-layer internal states of a model (such as `meta-llama/Llama-3.3-70B-Instruct`) when processing specific prompts. By passing `output_hidden_states=True` to the generation arguments, the script safely intercepts both the input sequence embeddings and the output generation embeddings, saving them locally as PyTorch tensors (`.pt`) for downstream interpretability or activation analysis.

## Prerequisites

- Python 3.8+
- PyTorch
- Hugging Face `transformers` and `huggingface_hub`
- A valid Hugging Face Access Token with read permissions for the desired model.

Install the required packages:
```bash
pip install torch transformers huggingface_hub tqdm
```

## Usage

1. **Set up Authentication:** 
   Update the script to use your Hugging Face API token. It is recommended to use an environment variable (e.g., `export HF_TOKEN="your_token_here"`) or replace the placeholder in the code.
2. **Configure File Paths:**
   Ensure the `prompts_path` points to your dataset (JSON format) and the output directories for the hidden states (`input_hidden_states_dir` and `output_hidden_states_dir`) are set to your preferred local storage locations.
3. **Run the script:**
   ```bash
   python extract_hidden_states.py
   ```

*Note: Extracting hidden states, especially for 70B models, requires significant memory and storage. Ensure your environment has sufficient disk space for the resulting `.pt` files and GPU VRAM (140GB at least to run in full percesion.) to load the model.*