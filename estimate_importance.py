from answer_parsing import extract_answer_letter
import argparse
import json
import logging
import random
import re
import asyncio
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset, get_dataset_config_names
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from scipy.special import binom

# Import provider-specific libraries
from openai import AsyncOpenAI, RateLimitError
import replicate
import google.generativeai as genai

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Helper Functions ---

def load_adjectives(file_path: Path) -> list[str]:
    """Reads a list of adjectives from a text file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Adjective file not found at: {file_path}")
    with open(file_path, 'r') as f:
        adjectives = [line.strip() for line in f if line.strip()]
    logging.info(f"Loaded {len(adjectives)} adjectives.")
    return adjectives

def sample_mmlu_tasks(num_per_category: int, seed: int) -> list[dict]:
    """Loads MMLU from Hugging Face and samples N questions per category."""
    from datasets import get_dataset_split_names

    mmlu_subjects = get_dataset_config_names("cais/mmlu")
    logging.info(f"Found {len(mmlu_subjects)} MMLU subjects. Sampling {num_per_category} from each.")

    all_samples = []
    for subject in tqdm(mmlu_subjects, desc="Sampling MMLU Subjects"):
        available_splits = get_dataset_split_names("cais/mmlu", subject)
        if 'test' not in available_splits:
            logging.warning(f"Subject '{subject}' does not have a 'test' split. Available: {available_splits}. Skipping.")
            continue

        dataset = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=True)
        sample_size = min(num_per_category, len(dataset))
        if sample_size < num_per_category:
            logging.warning(f"Subject '{subject}' has only {len(dataset)} questions. Using all of them.")

        random.seed(seed)
        sampled_indices = random.sample(range(len(dataset)), sample_size)
        for idx in sampled_indices:
            sample = dataset[idx]
            sample['subject'] = subject
            all_samples.append(sample)
            
    logging.info(f"Total questions sampled: {len(all_samples)}")
    return all_samples

def format_prompt_content(question_data: dict, adjectives: list[str]) -> str:
    """Formats the MMLU question and choices into the text for the API prompt."""
    question = question_data['question']
    choices = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(question_data['choices'])])
    
    adj_string = ", ".join(adjectives) if adjectives else "clear and direct"

    prompt_content = f"""
The following is a multiple-choice question. Your answer should be {adj_string}. Provide only the letter corresponding to the correct option.

Question: {question}

Choices:
{choices}
"""
    return prompt_content

# --- API Prediction Functions ---

async def get_openai_prediction(
    prompt_content: str, 
    client: AsyncOpenAI,
    model_name: str,
    semaphore: asyncio.Semaphore
) -> str:
    """Gets a prediction from OpenAI and extracts the answer letter."""
    async with semaphore:
        try:
            messages = [{"role": "user", "content": prompt_content}]
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages 
            )
            content = response.choices[0].message.content
            logging.info(f"API Request (OpenAI):\n---\n{prompt_content}\n---\nAPI Response: {content}")
            return extract_answer_letter(content)

        except RateLimitError:
            logging.warning("OpenAI rate limit reached. Sleeping for 20 seconds.")
            await asyncio.sleep(20)
            return "Z"
        except Exception as e:
            logging.error(f"OpenAI API call failed: {e}")
            return "Z"

async def get_replicate_prediction(
    prompt_content: str,
    client: None, # Client object not used by this function
    model_name: str,
    semaphore: asyncio.Semaphore
) -> str:
    """
    Gets a prediction from Replicate, inferring the correct prompt template
    and handling both streaming and non-streaming model outputs.
    """
    async with semaphore:
        template = None
        # Infer the correct prompt template based on the model name
        if 'llama-3' in model_name:
            template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            logging.info("Detected Llama 3 model family. Using Llama 3 prompt template.")
        elif 'phi-3' in model_name:
            template = "<|user|>\n{prompt}<|end|>\n<|assistant|>"
            logging.info("Detected Phi-3 model family. Using Phi-3 prompt template.")
        elif 'deepseek-r1' in model_name:
            template = "User: {prompt}\n\nAssistant:"
            logging.info("Detected DeepSeek R1 model family. Using DeepSeek prompt template.")
        else:
            template = "{prompt}"
            logging.warning(f"Could not infer prompt template for model '{model_name}'. Using a generic template.")

        try:
            api_input = {
                "prompt": prompt_content,
                "prompt_template": template,
                "temperature": 0.1
            }
            # The output can be either a streaming iterator or a complete list
            output = await replicate.async_run(model_name, input=api_input)
            
            content = ""
            # FIX: Handle both streaming and non-streaming responses
            if hasattr(output, '__aiter__'):
                # Handle streaming response
                async for item in output:
                    content += item
            elif isinstance(output, list):
                # Handle non-streaming response (list of strings)
                content = "".join(output)

            logging.info(f"API Request (Replicate):\n---\n{prompt_content}\n---\nAPI Response: {content}")
            return extract_answer_letter(content)

        except replicate.exceptions.ReplicateError as e:
            logging.warning(f"Replicate API error: {e}. Sleeping for 20 seconds.")
            await asyncio.sleep(20)
            return "Z"
        except Exception as e:
            logging.error(f"Replicate API call failed: {e}")
            return "Z"
        
async def get_gemini_prediction(
    prompt_content: str, 
    model: genai.GenerativeModel, # The client object is the model itself
    model_name: str, # Not used here, but kept for consistent signature
    semaphore: asyncio.Semaphore
) -> str:
    """Gets a prediction from Gemini and extracts the answer letter."""
    async with semaphore:
        try:
            # Per-request delay to manage rate limits.
            await asyncio.sleep(1) 
            response = await model.generate_content_async(prompt_content)
            content = response.text
            logging.info(f"API Request (Gemini):\n---\n{prompt_content}\n---\nAPI Response: {content}")
            return extract_answer_letter(content)

        except Exception as e:
            logging.error(f"Gemini API call failed: {e}")
            return "Z"

# --- Core SHAP Logic ---

async def calculate_shap_for_question(
    question_data: dict,
    adjectives: list[str],
    client, # Can be OpenAI/Gemini client or None for Replicate
    model_name: str,
    num_samples: int,
    semaphore: asyncio.Semaphore,
    prediction_function: callable
) -> np.ndarray:
    """Calculates Shapley values for all adjectives for a single question."""
    M = len(adjectives)
    
    coalitions = np.zeros((num_samples, M), dtype=int)
    for i in range(num_samples):
        num_features = random.randint(1, M)
        features_on = random.sample(range(M), num_features)
        coalitions[i, features_on] = 1
        
    coalitions[0, :] = 0
    coalitions[1, :] = 1
    
    correct_answer_idx = question_data['answer']
    correct_answer_letter = chr(65 + correct_answer_idx)

    tasks = []
    for coalition_vec in coalitions:
        active_adjectives = [adj for adj, active in zip(adjectives, coalition_vec) if active]
        prompt_content = format_prompt_content(question_data, active_adjectives)
        tasks.append(prediction_function(prompt_content, client, model_name, semaphore))

    predictions = await asyncio.gather(*tasks)
    
    scores = np.array([1 if pred == correct_answer_letter else 0 for pred in predictions])
    
    weights = np.zeros(num_samples)
    for i, coalition in enumerate(coalitions):
        z = np.sum(coalition)
        if z == 0 or z == M:
            weights[i] = 0
        else:
            weights[i] = (M - 1) / (binom(M, z) * z * (M - z) + 1e-6)
            
    model_lr = LinearRegression()
    model_lr.fit(X=coalitions, y=scores, sample_weight=weights)
    
    return model_lr.coef_

def save_results(output_file: Path, data: dict):
    """Saves the results dictionary to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

# --- Main Execution ---

async def main(args):
    """Main execution function."""
    logging.info(f"Starting SHAP analysis script with API provider: {args.api_provider}.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    client = None
    prediction_function = None

    if args.api_provider == 'openai':
        if not args.openai_api_key:
            raise ValueError("OpenAI API key is required via --openai_api_key.")
        client = AsyncOpenAI(api_key=args.openai_api_key)
        prediction_function = get_openai_prediction
        logging.info(f"Using OpenAI model: {args.model}")
    
    elif args.api_provider == 'replicate':
        if not args.replicate_api_token:
            raise ValueError("Replicate API token is required via --replicate_api_token.")
        os.environ['REPLICATE_API_TOKEN'] = args.replicate_api_token
        prediction_function = get_replicate_prediction
        logging.info(f"Using Replicate model: {args.model}")
    
    elif args.api_provider == 'gemini':
        if not args.google_api_key:
            raise ValueError("Google API key is required via --google_api_key.")
        genai.configure(api_key=args.google_api_key)
        client = genai.GenerativeModel(args.model) 
        prediction_function = get_gemini_prediction
        logging.info(f"Using Gemini model: {args.model}")

    adjectives = load_adjectives(args.adjectives_file)
    adjective_map = {adj: i for i, adj in enumerate(adjectives)}
    
    sampled_questions = sample_mmlu_tasks(args.num_per_category, args.seed)
    
    output_data = {}
    if args.output_file.exists():
        logging.info(f"Found existing results file at {args.output_file}. Loading...")
        try:
            with open(args.output_file, 'r') as f:
                output_data = json.load(f)
            logging.info(f"Loaded {len(output_data.get('per_question_results', {}))} previously computed question results.")
        except json.JSONDecodeError:
            logging.warning("Could not decode existing results file. Starting from scratch.")
            output_data = {}

    all_results = output_data.get('per_question_results', {})
    semaphore = asyncio.Semaphore(args.max_concurrent_requests)

    question_iterator = tqdm(sampled_questions, desc="Processing Questions")

    for i, question in enumerate(question_iterator):
        subject = question.get('subject', 'unknown')
        question_id = f"q_{i}_{subject}"

        if question_id in all_results:
            logging.info(f"Skipping already processed question {question_id}.")
            continue
        
        question_iterator.set_description(f"Processing Question {question_id}")
        
        shapley_values = await calculate_shap_for_question(
            question, adjectives, client, args.model, args.num_samples, semaphore, prediction_function
        )
        all_results[question_id] = shapley_values.tolist()

        print(f"\n\n--- Shapley Values for {question_id} ---")
        shap_series = pd.Series(shapley_values, index=adjectives).sort_values(ascending=False)
        print(shap_series.to_string())
        print("-------------------------------------------\n")

        current_output = {
            'parameters': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if 'key' not in k and 'token' not in k},
            'adjective_mapping': adjective_map,
            'per_question_results': all_results
        }
        save_results(args.output_file, current_output)

    logging.info("All questions processed. Aggregating final results...")
    if not all_results:
        logging.warning("No results were generated. Skipping final aggregation.")
        return

    df = pd.DataFrame(all_results).T
    df.columns = adjectives
    
    agg_results = pd.DataFrame({
        'mean_shapley': df.mean(),
        'std_shapley': df.std(),
        'mean_abs_shapley': df.abs().mean()
    }).sort_values(by='mean_abs_shapley', ascending=False)
    
    print("\n--- Aggregated Shapley Value Results ---")
    print(agg_results.to_string())
    
    final_output_data = {
        'parameters': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if 'key' not in k and 'token' not in k},
        'adjective_mapping': adjective_map,
        'aggregated_results': agg_results.to_dict('index'),
        'per_question_results': all_results
    }
    save_results(args.output_file, final_output_data)
    logging.info(f"All results saved to {args.output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run SHAP analysis on MMLU with adjectives using different API providers.")
    
    parser.add_argument("--adjectives_file", type=Path, required=True, help="Path to a .txt file with one adjective per line.")
    
    parser.add_argument("--api_provider", type=str, default='openai', choices=['openai', 'replicate', 'gemini'], help="The API provider to use.")
    
    parser.add_argument("--openai_api_key", type=str, help="Your OpenAI API key. Required if --api_provider is 'openai'.")
    parser.add_argument("--replicate_api_token", type=str, help="Your Replicate API token. Required if --api_provider is 'replicate'.")
    parser.add_argument("--google_api_key", type=str, help="Your Google Gemini API key. Required if --api_provider is 'gemini'.")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model identifier. For Replicate, templates are inferred for 'llama-3', 'phi-3', and 'deepseek-r1' models.")

    parser.add_argument("--output_file", type=Path, default="shap_results.json", help="Path to save the final JSON output. Used for resuming.")
    parser.add_argument("--num_per_category", type=int, default=5, help="Number of questions to sample from each MMLU category.")
    parser.add_argument("--num_samples", type=int, default=200, help="Number of coalitions to sample per question for SHAP.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--max_concurrent_requests", type=int, default=10, help="Maximum number of parallel requests to the API.")
    
    args = parser.parse_args()
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        logging.info("\nScript interrupted by user. Progress has been saved.")
    except ValueError as e:
        logging.error(f"Configuration Error: {e}")