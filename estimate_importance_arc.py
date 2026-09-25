from answer_parsing import extract_answer_letter
"""
ARC-Challenge pilot experiment for cross-benchmark validation.

Adapted from estimate_importance.py. Key differences:
- Loads ARC-Challenge instead of MMLU
- Supports --base_url for local llama-server (OpenAI-compatible endpoint)
- Samples N questions total (ARC has no subject categories)
- Handles ARC's variable answer counts and label formats
"""

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
from datasets import load_dataset
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from scipy.special import binom

# Import provider-specific libraries
from openai import AsyncOpenAI, RateLimitError

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

def sample_arc_tasks(num_questions: int, seed: int) -> list[dict]:
    """Loads ARC-Challenge from Hugging Face and samples N questions.

    Normalizes the ARC schema to match the MMLU format used by the rest
    of the pipeline:
      - 'question': str
      - 'choices': list[str]
      - 'answer': int (0-based index of the correct answer)
      - 'subject': str (always 'arc_challenge' for this dataset)
    """
    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    logging.info(f"Loaded ARC-Challenge test split with {len(dataset)} questions.")

    sample_size = min(num_questions, len(dataset))
    random.seed(seed)
    sampled_indices = random.sample(range(len(dataset)), sample_size)

    all_samples = []
    for idx in sampled_indices:
        raw = dataset[idx]
        # Normalize choices: ARC stores them as {"text": [...], "label": [...]}
        choice_texts = raw['choices']['text']
        choice_labels = raw['choices']['label']
        answer_key = raw['answerKey']

        # Find the correct answer index. ARC labels can be letters (A,B,C,D,E)
        # or numbers (1,2,3,4,5).
        if answer_key in choice_labels:
            answer_idx = choice_labels.index(answer_key)
        else:
            # Fallback: try matching letter to position
            letter_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4,
                          '1': 0, '2': 1, '3': 2, '4': 3, '5': 4}
            answer_idx = letter_map.get(answer_key, 0)

        sample = {
            'question': raw['question'],
            'choices': choice_texts,
            'answer': answer_idx,
            'subject': 'arc_challenge',
            'num_choices': len(choice_texts),
        }
        all_samples.append(sample)

    logging.info(f"Sampled {len(all_samples)} ARC-Challenge questions.")
    return all_samples

def format_prompt_content(question_data: dict, adjectives: list[str]) -> str:
    """Formats the ARC question and choices into the text for the API prompt."""
    question = question_data['question']
    num_choices = question_data.get('num_choices', len(question_data['choices']))
    choices = "\n".join([f"{chr(65+i)}. {choice}"
                         for i, choice in enumerate(question_data['choices'])])

    adj_string = ", ".join(adjectives) if adjectives else "clear and direct"

    prompt_content = f"""The following is a multiple-choice question. Your answer should be {adj_string}. Provide only the letter corresponding to the correct option.

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
    semaphore: asyncio.Semaphore,
    num_choices: int = 4,
) -> str:
    """Gets a prediction from an OpenAI-compatible endpoint."""
    async with semaphore:
        try:
            messages = [{"role": "user", "content": prompt_content}]
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=256,
                temperature=0.1
            )
            content = response.choices[0].message.content
            logging.debug(f"API Response: {content[:200]}")
            return extract_answer_letter(content, num_choices=num_choices)

        except RateLimitError:
            logging.warning("Rate limit reached. Sleeping for 20 seconds.")
            await asyncio.sleep(20)
            return "Z"
        except Exception as e:
            logging.error(f"API call failed: {e}")
            return "Z"

# --- Core SHAP Logic ---

async def calculate_shap_for_question(
    question_data: dict,
    adjectives: list[str],
    client,
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
        tasks.append(prediction_function(
            prompt_content, client, model_name, semaphore,
            num_choices=question_data.get('num_choices', len(question_data['choices'])),
        ))

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
    logging.info(f"Starting ARC-Challenge SHAP analysis.")
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Initialize OpenAI-compatible client (works with local llama-server)
    client_kwargs = {"api_key": args.openai_api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = AsyncOpenAI(**client_kwargs)
    prediction_function = get_openai_prediction
    logging.info(f"Using model: {args.model} at {args.base_url or 'OpenAI API'}")

    adjectives = load_adjectives(args.adjectives_file)
    adjective_map = {adj: i for i, adj in enumerate(adjectives)}

    sampled_questions = sample_arc_tasks(args.num_questions, args.seed)

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
        question_id = f"q_{i}_arc_challenge"

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
            'parameters': {
                'benchmark': 'arc_challenge',
                'model': args.model,
                'base_url': args.base_url,
                'num_questions': args.num_questions,
                'num_samples': args.num_samples,
                'seed': args.seed,
                'num_per_category': args.num_questions,  # compatibility key
                'api_provider': 'openai',
            },
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
        'parameters': {
            'benchmark': 'arc_challenge',
            'model': args.model,
            'base_url': args.base_url,
            'num_questions': args.num_questions,
            'num_samples': args.num_samples,
            'seed': args.seed,
            'num_per_category': args.num_questions,
            'api_provider': 'openai',
        },
        'adjective_mapping': adjective_map,
        'aggregated_results': agg_results.to_dict('index'),
        'per_question_results': all_results
    }
    save_results(args.output_file, final_output_data)
    logging.info(f"All results saved to {args.output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run SHAP analysis on ARC-Challenge with adjectives using an OpenAI-compatible endpoint.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("--adjectives_file", type=Path, required=True,
                        help="Path to a .txt file with one adjective per line.")
    parser.add_argument("--model", type=str, default="local-model",
                        help="Model identifier. For llama-server, any string works.")
    parser.add_argument("--openai_api_key", type=str, default="not-needed",
                        help="OpenAI API key. Use 'not-needed' for local llama-server.")
    parser.add_argument("--base_url", type=str, default=None,
                        help="Base URL for OpenAI-compatible endpoint (e.g., http://localhost:8080/v1).")
    parser.add_argument("--output_file", type=Path, default="arc_shap_results.json",
                        help="Path to save the final JSON output. Used for resuming.")
    parser.add_argument("--num_questions", type=int, default=100,
                        help="Number of ARC-Challenge questions to sample.")
    parser.add_argument("--num_samples", type=int, default=200,
                        help="Number of coalitions to sample per question for SHAP.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--max_concurrent_requests", type=int, default=1,
                        help="Max parallel requests. Use 1 for local llama-server.")

    parser.epilog = (
        "Example usage with local llama-server:\n"
        "  1. Start llama-server:\n"
        "     llama-server -m model.gguf --port 8080 -ngl 99\n\n"
        "  2. Run this script:\n"
        "     python estimate_importance_arc.py \\\n"
        "       --adjectives_file adjectives100.txt \\\n"
        "       --base_url http://localhost:8080/v1 \\\n"
        "       --output_file arc_phi3_results.json \\\n"
        "       --num_questions 100 \\\n"
        "       --num_samples 200\n"
    )

    args = parser.parse_args()
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        logging.info("\nScript interrupted by user. Progress has been saved.")
    except ValueError as e:
        logging.error(f"Configuration Error: {e}")
