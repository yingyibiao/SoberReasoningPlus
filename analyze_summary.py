import os
import json
import numpy as np
import pandas as pd
import argparse
import logging
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# List of seed values to analyze0, 1, 2, 3, 4, 42, 100, 123, 666, 2023
SEEDS = [0, 1, 2, 3, 4, 42, 100, 110, 123, 666, 888, 911, 999, 1000, 2025, 2026]

def load_results(file_path):
    """
    Load the results from the JSON file.
    
    Args:
        file_path (str): Path to the JSON file containing results.
        
    Returns:
        dict: The loaded JSON data or None if loading fails.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logging.info(f"Successfully loaded results from {file_path}")
        return data
    except Exception as e:
        logging.error(f"Error loading results from {file_path}: {e}")
        return None

def compute_mean_and_std(values):
    """
    Compute mean and standard deviation of values.
    
    Args:
        values (list): List of values to analyze.
        
    Returns:
        tuple: (mean, std_dev)
    """
    import numpy as np
    mean = np.mean(values)
    std_dev = np.std(values, ddof=1) if len(values) > 1 else 0.0
    return mean, std_dev

def analyze_by_seed(data):
    """
    Analyze results by seed, averaging across all configurations for each seed.
    
    Args:
        data (dict): The loaded results data.
        
    Returns:
        dict: A dictionary containing average results for each seed.
    """
    seed_results = {}
    
    # Group experiments by seed
    for seed in SEEDS:
        seed_experiments = []
        
        # Find all experiments with this seed
        for exp_name, exp_data in data.items():
            try:
                exp_seed = exp_data["configuration"]["seed"]
                if exp_seed == str(seed):
                    seed_experiments.append(exp_data)
            except KeyError:
                continue
                
        if not seed_experiments:
            logging.warning(f"No experiments found for seed {seed}")
            continue
            
        # Calculate average results for this seed
        accuracy_values = []
        token_length_values = []
        sample_counts = []
        
        for exp in seed_experiments:
            try:
                accuracy_values.append(exp["results"]["accuracy"])
                token_length_values.append(exp["results"]["total_token_length"])
                sample_counts.append(exp["results"]["num_samples"])
            except KeyError as e:
                logging.warning(f"Missing key in experiment: {e}")
                continue
                
        # Calculate means and standard deviations
        avg_accuracy = np.mean(accuracy_values)
        std_accuracy = np.std(accuracy_values, ddof=1) if len(accuracy_values) > 1 else 0.0
        avg_token_length = np.mean(token_length_values)
        std_token_length = np.std(token_length_values, ddof=1) if len(token_length_values) > 1 else 0.0
        
        # Calculate new metrics
        total_tokens_for_seed = np.sum(token_length_values)
        total_samples_for_seed = np.sum(sample_counts)
        avg_token_length_per_sample = (
            total_tokens_for_seed / total_samples_for_seed
            if total_samples_for_seed > 0
            else 0
        )

        # Store results
        seed_results[str(seed)] = {
            "seed": seed,
            "results": {
                "accuracy": avg_accuracy,
                "accuracy_std_dev": std_accuracy,
                "total_token_length": avg_token_length,
                "token_length_std_dev": std_token_length,
                "avg_token_length_per_sample": avg_token_length_per_sample,
                "num_experiments": len(seed_experiments),
                "avg_samples": np.mean(sample_counts) if sample_counts else 0
            }
        }
        
        logging.info(f"Seed {seed}: Processed {len(seed_experiments)} experiments, " 
                    f"avg accuracy: {avg_accuracy:.4f}, "
                    f"avg total token length: {avg_token_length:.2f}")
    
    return seed_results

def analyze_by_config(data):
    """
    Analyze results by configuration, averaging across seeds for each config.
    
    Args:
        data (dict): The loaded results data.
        
    Returns:
        dict: A dictionary containing average results for each configuration.
    """
    # Initialize storage for different configurations
    config_results = {}
    
    # Group experiments by configuration
    config_experiments = defaultdict(list)
    
    for exp_name, exp_data in data.items():
        try:
            # Extract configuration parameters
            temp = exp_data["configuration"]["temperature"]
            top_p = exp_data["configuration"]["top_p"]
            dtype = exp_data["configuration"]["dtype"]
            max_num_seqs = exp_data["configuration"]["max_num_seqs"]
            max_num_batched_tokens = exp_data["configuration"]["max_num_batched_tokens"]
            dataset = exp_data["configuration"]["dataset"]
            max_model_length = exp_data["configuration"]["max_model_length"]
            
            # Create a configuration key
            config_key = (
                f"temp_{temp}_topp_{top_p}_dtype_{dtype}_seqs_{max_num_seqs}_"
                f"tokens_{max_num_batched_tokens}_dataset_{dataset}_len_{max_model_length}"
            )
            
            config_experiments[config_key].append(exp_data)
        except KeyError:
            continue
    
    # Calculate average results for each configuration
    for config_key, experiments in config_experiments.items():
        if not experiments:
            continue
            
        # Get configuration details from the first experiment
        first_exp = experiments[0]
        temp = first_exp["configuration"]["temperature"]
        top_p = first_exp["configuration"]["top_p"]
        dtype = first_exp["configuration"]["dtype"]
        max_num_seqs = first_exp["configuration"]["max_num_seqs"]
        max_num_batched_tokens = first_exp["configuration"]["max_num_batched_tokens"]
        dataset = first_exp["configuration"]["dataset"]
        max_model_length = first_exp["configuration"]["max_model_length"]
        
        # Extract metrics from all experiments with this configuration
        accuracy_values = []
        total_token_length_values = []
        avg_token_length_per_sample_values = []
        sample_counts = []
        
        for exp in experiments:
            try:
                accuracy_values.append(exp["results"]["accuracy"])
                total_token_length_values.append(exp["results"]["total_token_length"])
                
                # Calculate avg token length per sample for each experiment (seed)
                total_tokens = exp["results"]["total_token_length"]
                num_samples = exp["results"]["num_samples"]
                if num_samples > 0:
                    avg_len_per_sample = total_tokens / num_samples
                    avg_token_length_per_sample_values.append(avg_len_per_sample)
                
                sample_counts.append(num_samples)
            except KeyError as e:
                logging.warning(f"Missing key in experiment: {e}")
                continue
        
        # Calculate means and standard deviations
        avg_accuracy = np.mean(accuracy_values)
        std_accuracy = np.std(accuracy_values, ddof=1) if len(accuracy_values) > 1 else 0.0
        avg_total_token_length = np.mean(total_token_length_values)
        std_total_token_length = np.std(total_token_length_values, ddof=1) if len(total_token_length_values) > 1 else 0.0
        
        # Calculate mean and std for average token length per sample (across seeds)
        avg_token_length_per_sample = np.mean(avg_token_length_per_sample_values)
        std_token_length_per_sample = np.std(avg_token_length_per_sample_values, ddof=1) if len(avg_token_length_per_sample_values) > 1 else 0.0
        
        # Store results
        config_results[config_key] = {
            "configuration": {
                "temperature": temp,
                "top_p": top_p,
                "dtype": dtype,
                "max_num_seqs": max_num_seqs,
                "max_num_batched_tokens": max_num_batched_tokens,
                "dataset": dataset,
                "max_model_length": max_model_length
            },
            "results": {
                "accuracy": avg_accuracy,
                "accuracy_std_dev": std_accuracy,
                "total_token_length": avg_total_token_length,
                "total_token_length_std": std_total_token_length,
                "avg_token_length_per_sample": avg_token_length_per_sample,
                "avg_token_length_per_sample_std": std_token_length_per_sample,
                "num_experiments": len(experiments),
                "avg_samples": np.mean(sample_counts) if sample_counts else 0
            }
        }
        
        logging.info(f"Config temp={temp}, top_p={top_p}, dtype={dtype}: "
                    f"Processed {len(experiments)} experiments, "
                    f"avg accuracy: {avg_accuracy:.4f}, "
                    f"avg total token length: {avg_total_token_length:.2f}")
    
    return config_results

def main():
    parser = argparse.ArgumentParser(description='Analyze LLM experiment results')
    
    parser.add_argument('--results-path', type=str, required=True,
                        help='Path to the results JSON file')
    
    parser.add_argument('--output-path', type=str, required=True,
                        help='Path to save the analysis results')
    
    args = parser.parse_args()
    
    # Load results
    data = load_results(args.results_path)
    if data is None:
        return
    
    # Analyze by seed
    seed_results = analyze_by_seed(data)
    
    # Analyze by configuration
    config_results = analyze_by_config(data)
    
    # Compile final results
    final_results = {
        "seed_analysis": seed_results,
        "configuration_analysis": config_results,
    }
    
    # Save results
    try:
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        
        with open(args.output_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        logging.info(f"Successfully saved analysis results to {args.output_path}")
        
        # Also save a more readable CSV for configuration analysis
        config_df = pd.DataFrame([
            {
                "temperature": data["configuration"]["temperature"],
                "top_p": data["configuration"]["top_p"],
                "dtype": data["configuration"]["dtype"],
                "max_num_seqs": data["configuration"]["max_num_seqs"],
                "max_num_batched_tokens": data["configuration"]["max_num_batched_tokens"],
                "dataset": data["configuration"]["dataset"],
                "max_model_length": data["configuration"]["max_model_length"],
                "accuracy": data["results"]["accuracy"],
                "accuracy_std_dev": data["results"]["accuracy_std_dev"],
                "total_token_length": data["results"]["total_token_length"],
                "total_token_length_std": data["results"]["total_token_length_std"],
                "avg_token_length_per_sample": data["results"]["avg_token_length_per_sample"],
                "avg_token_length_per_sample_std": data["results"]["avg_token_length_per_sample_std"],
                "num_experiments": data["results"]["num_experiments"],
                "avg_samples": data["results"]["avg_samples"]
            }
            for config, data in config_results.items()
        ])
        
        csv_path = os.path.join(os.path.dirname(args.output_path), "config_analysis.csv")
        config_df.to_csv(csv_path, index=False)
        logging.info(f"Saved configuration analysis CSV to {csv_path}")
        
    except Exception as e:
        logging.error(f"Error saving analysis results: {e}")

if __name__ == "__main__":
    main()
