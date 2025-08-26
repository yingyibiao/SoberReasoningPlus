import os

import lighteval
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.model_loader import load_model
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.models.model_input import GenerationParameters
from lighteval.pipeline import Pipeline, PipelineParameters, ParallelismManager
from datetime import datetime
import argparse
import logging
from fsspec import url_to_fs
import asyncio

__version__ = f"2.0_lighteval@{lighteval.__version__}"

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    
    # I/O and General Parameters
    parser.add_argument("--output_dir", default="output", type=str, help="Directory to save the output files")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--task_seeds", type=str, nargs='+', required=True, help="List of task and seed mappings, e.g., 'task_name:seed1,seed2,seed3'")
    parser.add_argument("--custom_tasks_directory", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")

    # Sampling Parameters
    parser.add_argument("--temperature", type=float, nargs='+', required=True)
    parser.add_argument("--top_p", type=float, nargs='+', required=True)
    parser.add_argument("--max_new_tokens", type=int, default=32768)

    # Model Configuration Parameters
    parser.add_argument("--max_model_length", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--cot_prompt", type=str, default=None)
    parser.add_argument("--system_prompt", type=str, default=None)
    parser.add_argument("--launcher_type", type=str, default="VLLM")
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--data_parallel_size", type=int, default=1)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_num_seqs", type=int, default=128)
    parser.add_argument("--max_num_batched_tokens", type=int, default=32768)
    parser.add_argument("--use_chat_template", action="store_true")

    return parser.parse_args()


def main():
    start = datetime.now()
    args = parse_args()
    fs, output_dir = url_to_fs(args.output_dir)

    max_model_length = args.max_model_length
    if args.max_model_length is None:
        print("max_model_length not set. Setting it to max_new_tokens.")
        max_model_length = args.max_new_tokens
    elif args.max_model_length == -1:
        print("max_model_length is -1. Setting it to None.")
        max_model_length = None

    # Parse the task_seeds argument
    tasks_with_seeds = {}
    for ts in args.task_seeds:
        try:
            task_name, seeds_str = ts.split(':', 1)
            seeds = [int(s) for s in seeds_str.split(',')]
            tasks_with_seeds[task_name] = seeds
        except ValueError:
            raise ValueError(f"Invalid format for --task_seeds argument: '{ts}'. Expected 'task_name:seed1,seed2,...'")

    # Load the model once
    model_config = VLLMModelConfig(
        model_name=args.model,
        dtype=args.dtype,
        seed=42,
        max_model_length=max_model_length,
        gpu_memory_utilization=args.gpu_memory_utilization,
        pipeline_parallel_size=args.pipeline_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        use_chat_template=args.use_chat_template,
        generation_parameters=GenerationParameters(
            max_new_tokens=args.max_new_tokens,
        ),
    )
    model = load_model(config=model_config)

    print(args.temperature, args.top_p)
    print(args.temperature.__class__, args.top_p.__class__)
    for temp in args.temperature:
        for top_p in args.top_p:
            for task, seeds in tasks_with_seeds.items():
                print(f"Running evaluation for task: {task}, temperature: {temp}, top_p: {top_p}")
                for seed in seeds:
                    print(f"\n---> Starting run: seed={seed}")

                    # Create a meaningful run name based on parameters
                    model_folder_name = f"{args.model.replace('/', '-')}-{args.dtype}-{max_model_length}"
                    run_name = (
                        f"{seed}-{temp}-{top_p}-{task.split('|')[1]}-"
                        f"{args.tensor_parallel_size}-{args.max_num_seqs}-"
                        f"{args.max_num_batched_tokens}-{args.max_new_tokens}"
                    )
                    if not args.use_chat_template:
                        run_name += "-nochat"

                    # Define the dedicated output directory for this specific run
                    run_output_dir = os.path.join(output_dir, model_folder_name, run_name)

                    # Check for existing results to allow for resuming runs
                    if not args.overwrite and fs.exists(run_output_dir):
                        json_files = fs.glob(os.path.join(run_output_dir, "**", "*.json"))
                        parquet_files = fs.glob(os.path.join(run_output_dir, "**", "*.parquet"))
                        if json_files and parquet_files:
                            print(f"Skipping run, results found in {run_output_dir}")
                            continue
                        else:
                            print(f"Incomplete run found in {run_output_dir}. Deleting and re-running.")
                            fs.rm(run_output_dir, recursive=True)
                    
                    fs.makedirs(run_output_dir, exist_ok=True)

                    evaluation_tracker = EvaluationTracker(
                        output_dir=run_output_dir,  # Now using the run-specific output directory
                        save_details=True,
                        push_to_hub=False,
                        push_to_tensorboard=False,
                        public=False,
                        hub_results_org=None,
                    )

                    pipeline_params = PipelineParameters(
                        launcher_type=ParallelismManager.VLLM,
                        job_id=0,
                        dataset_loading_processes=1,
                        custom_tasks_directory=args.custom_tasks_directory,
                        num_fewshot_seeds=1,
                        max_samples=None,
                        use_chat_template=args.use_chat_template,
                        system_prompt=args.system_prompt,
                        cot_prompt=args.cot_prompt,
                        load_responses_from_details_date_id=None,
                    )
                    pipeline = Pipeline(
                        tasks=task,
                        pipeline_parameters=pipeline_params,
                        evaluation_tracker=evaluation_tracker,
                        model=model,
                        metric_options={},
                    )
                    # Update generation parameters for the current run
                    model_config.generation_parameters.temperature = temp
                    model_config.generation_parameters.top_p = top_p
                    model_config.generation_parameters.seed = seed

                    pipeline.evaluate()
                    pipeline.show_results()
                    pipeline.save_and_push_results()
                    print(f"Finished run for all seeds")
    
    print(f"Total execution time: {datetime.now() - start}")


if __name__ == "__main__":
    main()
