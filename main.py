import os
import uuid

import lighteval
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.models.model_input import GenerationParameters
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from datetime import datetime
import argparse
from fsspec import url_to_fs

__version__ = f"2.0_lighteval@{lighteval.__version__}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="output",
        type=str,
        help="Directory to save the output files",
    )
    parser.add_argument(
        "--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--task", type=str, default="lighteval|aime24|0|0")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed; if not set, a unique value is generated",
    )
    parser.add_argument("--max_new_tokens", type=int, default=32768)
    parser.add_argument("--max_model_length", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--cot_prompt", type=str, default=None)
    parser.add_argument("--system_prompt", type=str, default=None)
    parser.add_argument("--custom_tasks_directory", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
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
    seed = args.seed if args.seed is not None else uuid.uuid4().int & 0xFFFFFFFF
    print(f"Using seed: {seed}")
    fs, output_dir = url_to_fs(args.output_dir)

    max_model_length = args.max_model_length
    if args.max_model_length is None:
        print("max_model_length not set. Setting it to max_new_tokens.")
        max_model_length = args.max_new_tokens
    elif args.max_model_length == -1:
        print("max_model_length is -1. Setting it to None.")
        max_model_length = None

    # Create a meaningful run name based on parameters
    model_folder_name = args.model.replace("/", "_")
    run_name = f"{seed}-{args.temperature}-{args.top_p}-{args.dtype}-{args.max_num_seqs}-{args.max_num_batched_tokens}-{args.task.split('|')[1]}-{args.max_new_tokens}"
    if max_model_length != args.max_new_tokens:
        run_name += f"-{max_model_length}"
    if not args.use_chat_template:
        run_name += "-nochat"

    # Define the dedicated output directory for this specific run
    run_output_dir = os.path.join(output_dir, model_folder_name, run_name)
    fs.makedirs(run_output_dir, exist_ok=True)

    # Check if results already exist in this dedicated directory
    fpath = os.path.join(run_output_dir, "summary.json")
    if fs.exists(fpath) and not args.overwrite:
        print(f"File {fpath} already exists. Skipping.")
        return

    evaluation_tracker = EvaluationTracker(
        output_dir=run_output_dir,  # Now using the run-specific output directory
        save_details=True,
        push_to_hub=False,
        push_to_tensorboard=False,
        public=False,
        hub_results_org=None,
    )
    # assert args.launcher_type == "VLLM", "Only VLLM is supported for now"
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.VLLM,
        job_id=0,
        dataset_loading_processes=1,
        custom_tasks_directory=args.custom_tasks_directory,
        # override_batch_size=-1,  # Cannot override batch size when using VLLM
        num_fewshot_seeds=1,
        max_samples=None,
        use_chat_template=args.use_chat_template,
        system_prompt=args.system_prompt,
        cot_prompt=args.cot_prompt,
        load_responses_from_details_date_id=None,
    )

    model_config = VLLMModelConfig(
        model_name=args.model,
        dtype=args.dtype,
        seed=seed,
        max_model_length=max_model_length,
        gpu_memory_utilization=args.gpu_memory_utilization,
        pipeline_parallel_size=args.pipeline_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        use_chat_template=args.use_chat_template,
        generation_parameters=GenerationParameters(
            max_new_tokens=args.max_new_tokens,
            seed=seed,
            temperature=args.temperature,
            top_p=args.top_p,
        ),
    )

    pipeline = Pipeline(
        tasks=args.task,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
        metric_options={},
    )

    pipeline.evaluate()
    pipeline.show_results()
    pipeline.save_and_push_results()


if __name__ == "__main__":
    main()
