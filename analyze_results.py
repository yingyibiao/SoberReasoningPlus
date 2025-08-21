import os
import json
import glob
import pandas as pd
from transformers import AutoTokenizer
import logging
from tqdm import tqdm
import argparse

# --- Configuration ---
# Output file name for the aggregated results
OUTPUT_FILENAME = "analysis_results.json"

# Column name in the Parquet file for the response
RESPONSE_COLUMN = "completion"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_tokenizer(tokenizer_name):
    """
    Load the tokenizer for token length calculation.
    
    Returns:
        The loaded tokenizer or None if loading fails.
    """
    logging.info(f"Loading tokenizer from '{tokenizer_name}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        logging.info("Tokenizer loaded successfully.")
        return tokenizer
    except Exception as e:
        logging.error(f"Failed to load tokenizer: {e}")
        return None

def calculate_total_token_length(details_path, tokenizer):
    """
    Calculate the total token length of responses in parquet files.
    
    Args:
        details_path (str): Path to the details directory containing parquet files.
        tokenizer: The HuggingFace tokenizer to use.
        
    Returns:
        dict: Dictionary with total token length and number of samples, or None if processing fails.
    """
    # Check if the directory exists
    if not os.path.exists(details_path) or not os.path.isdir(details_path):
        logging.warning(f"Details directory does not exist: {details_path}")
        return None
        
    # Find all parquet files in the details directory
    parquet_files = glob.glob(os.path.join(details_path, "**", "*.parquet"), recursive=True)
    
    if not parquet_files:
        logging.warning(f"No parquet files found in {details_path}")
        return None
    
    # Calculate the total token length across all responses
    total_token_length = 0
    num_samples = 0
    
    for file_path in parquet_files:
        try:
            df = pd.read_parquet(file_path)
            
            if RESPONSE_COLUMN not in df.columns:
                # Try to find the right column for responses
                possible_columns = ["completion", "predictions", "generated_text", "response"]
                for col in possible_columns:
                    if col in df.columns:
                        logging.info(f"Found response column: {col} instead of {RESPONSE_COLUMN}")
                        response_column = col
                        break
                else:
                    logging.warning(f"Could not find response column in {file_path}. Available columns: {df.columns.tolist()}")
                    continue
            else:
                response_column = RESPONSE_COLUMN
            
            # Calculate token length for each response and add to total
            for _, row in df.iterrows():
                response = str(row.get(response_column, ""))
                tokens = tokenizer.encode(response, add_special_tokens=False)
                total_token_length += len(tokens)
                num_samples += 1
            
            logging.info(f"Processed {file_path}: added {num_samples} responses")
            
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")
            continue
    
    if num_samples > 0:
        return {
            "total_token_length": int(total_token_length),
            "num_samples": num_samples
        }
    else:
        logging.warning("No valid responses found to calculate token length")
        return None

def extract_accuracy_from_results(results_path):
    """
    Extract the accuracy (extractive_match) from the results JSON file.
    
    Args:
        results_path (str): Path to the results directory.
        
    Returns:
        dict: Dictionary containing accuracy and stderr, or None if extraction fails.
    """
    # Check if the directory exists
    if not os.path.exists(results_path) or not os.path.isdir(results_path):
        logging.warning(f"Results directory does not exist: {results_path}")
        return None
        
    # Find the results JSON file
    results_files = glob.glob(os.path.join(results_path, "**", "results_*.json"), recursive=True)
    
    if not results_files:
        logging.warning(f"No results files found in {results_path}")
        return None
    
    # Use the most recent results file if there are multiple
    results_file = sorted(results_files)[-1]
    
    try:
        with open(results_file, 'r') as f:
            results_data = json.load(f)
        
        # Extract accuracy information
        if "results" in results_data and "all" in results_data["results"]:
            acc = results_data["results"]["all"].get("extractive_match")
            stderr = results_data["results"]["all"].get("extractive_match_stderr")
            return {"accuracy": acc, "stderr": stderr}
        else:
            # Check if it's in a different structure
            for key, value in results_data.get("results", {}).items():
                if "extractive_match" in value:
                    return {
                        "accuracy": value["extractive_match"],
                        "stderr": value.get("extractive_match_stderr")
                    }
            
            logging.warning(f"Could not find extractive_match in results file {results_file}")
            return None
            
    except Exception as e:
        logging.error(f"Error extracting accuracy from {results_file}: {e}")
        return None

def process_experiment(exp_path, tokenizer):
    """
    Process a single experiment directory to extract accuracy and token length.
    
    Args:
        exp_path (str): Path to the experiment directory.
        tokenizer: The tokenizer to use for token length calculation.
        
    Returns:
        dict: Dictionary containing experiment results, or None if processing fails.
    """
    # Check if the necessary directories exist (results and details)
    results_path = os.path.join(exp_path, "results")
    details_path = os.path.join(exp_path, "details")
    
    if not os.path.exists(results_path) or not os.path.exists(details_path):
        logging.info(f"Experiment {os.path.basename(exp_path)} is incomplete (missing results or details directory), skipping...")
        return None
    
    # Check if results file already exists
    output_file = os.path.join(exp_path, OUTPUT_FILENAME)
    if os.path.exists(output_file):
        logging.info(f"Results file already exists for {exp_path}, skipping...")
        try:
            with open(output_file, 'r') as f:
                return json.load(f)
        except:
            logging.warning(f"Existing results file is invalid, will recalculate")
            # Continue with processing
    
    # Extract experiment configuration from directory name
    exp_name = os.path.basename(exp_path)
    config_parts = exp_name.split('-')
    
    # Parse configuration parameters (seed, temperature, top_p, dtype, etc.)
    try:
        seed = config_parts[0]
        temperature = config_parts[1]
        top_p = config_parts[2]
        dtype = config_parts[3]
        max_num_seqs = config_parts[4] 
        max_num_batched_tokens = config_parts[5]
        dataset = config_parts[6]
        max_model_length = config_parts[7]
    except IndexError:
        seed, temperature, top_p, dtype, max_num_seqs, max_num_batched_tokens, dataset, max_model_length = ("unknown",) * 8
        logging.warning(f"Could not parse configuration from directory name: {exp_name}")
    
    # Extract accuracy from results
    results_path = os.path.join(exp_path, "results")
    accuracy_data = extract_accuracy_from_results(results_path)
    
    # Calculate total token length
    details_path = os.path.join(exp_path, "details")
    token_length_data = calculate_total_token_length(details_path, tokenizer)
    
    # Compile results
    results = {
        "experiment": exp_name,
        "configuration": {
            "seed": seed,
            "temperature": temperature,
            "top_p": top_p,
            "dtype": dtype,
            "max_num_seqs": max_num_seqs,
            "max_num_batched_tokens": max_num_batched_tokens,
            "dataset": dataset,
            "max_model_length": max_model_length
        },
        "results": {}
    }
    
    if accuracy_data:
        results["results"]["accuracy"] = accuracy_data["accuracy"]
    
    if token_length_data is not None:
        results["results"]["total_token_length"] = token_length_data["total_token_length"]
        results["results"]["num_samples"] = token_length_data["num_samples"]
    
    # Save results to file
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Saved results to {output_file}")
    except Exception as e:
        logging.error(f"Error saving results to {output_file}: {e}")
    
    return results

def find_experiment_directories(base_dir, model_name_pattern):
    """
    Find all experiment directories for the specified model pattern.
    
    Returns:
        list: List of paths to experiment directories.
    """
    model_path = os.path.join(base_dir, model_name_pattern)
    if not os.path.exists(model_path):
        logging.error(f"Model directory not found: {model_path}")
        return []
    
    # Get all subdirectories (experiment configurations)
    exp_dirs = [os.path.join(model_path, d) for d in os.listdir(model_path) 
                if os.path.isdir(os.path.join(model_path, d))]
    
    logging.info(f"Found {len(exp_dirs)} experiment directories")
    return exp_dirs

def process_all_experiments(base_dir, model_name_pattern, tokenizer_name):
    """
    Process all experiment directories and compile the results.
    
    Returns:
        dict: Dictionary containing all experiment results.
    """
    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_name)
    if not tokenizer:
        return None
    
    # Find experiment directories
    exp_dirs = find_experiment_directories(base_dir, model_name_pattern)
    
    # Process each experiment
    all_results = {}
    for exp_dir in tqdm(exp_dirs, desc="Processing experiments"):
        exp_name = os.path.basename(exp_dir)
        results = process_experiment(exp_dir, tokenizer)
        if results:
            all_results[exp_name] = results
    
    # Save aggregated results
    output_path = os.path.join(base_dir, model_name_pattern, "all_experiments_results.json")
    try:
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        logging.info(f"Saved aggregated results to {output_path}")
    except Exception as e:
        logging.error(f"Error saving aggregated results: {e}")
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description='Analyze experiment results.')
    parser.add_argument('--base_dir', type=str, required=True, help='Base directory containing experiment results.')
    parser.add_argument('--model_name_pattern', type=str, required=True, help='Model name pattern to process.')
    parser.add_argument('--tokenizer_name', type=str, required=True, help='Tokenizer for calculating token lengths.')
    args = parser.parse_args()

    logging.info("Starting experiment analysis...")
    
    # Process all experiments
    all_results = process_all_experiments(args.base_dir, args.model_name_pattern, args.tokenizer_name)
    
    if all_results:
        logging.info(f"Successfully processed {len(all_results)} experiments")
    else:
        logging.error("Failed to process experiments")
    
    logging.info("Analysis complete.")

if __name__ == "__main__":
    main()
