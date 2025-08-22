"""Entry point for running evaluation sweeps with a single vLLM instance.

This script initialises the vLLM model once and then iterates over a list of
sampling parameter combinations (temperature, top-p …) to run multiple
evaluations without repeatedly spawning Python processes.  The heavy
initialisation cost of `LLM` is therefore amortised across all experiments.

Only arguments that require re‑initialising the model (model path, tensor
parallel size, dtype …) are exposed via the command line.  Parameters that can
be changed between requests (temperature, top_p, number of runs …) are supplied
from the shell script and applied during the sweep.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from lm_eval import evaluator
from lm_eval.models.vllm_causallms import VLLM
from vllm import LLM

# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation sweep")
    parser.add_argument("--model", "-m", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--tensor_parallel_size", "-tp", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument(
        "--tasks",
        type=str,
        default="aime24,aime25,amc23,minerva,olympiadbench,math_500",
        help="Comma separated list of evaluation tasks",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=100, help="Limit samples")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Where to store result JSON files",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate per example",
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[0.0, 0.6, 0.8, 0.9, 1.0],
        help="List of temperatures to sweep over",
    )
    parser.add_argument(
        "--top_ps",
        type=float,
        nargs="+",
        default=[0.6, 0.8, 0.9, 0.95, 1.0],
        help="List of top-p values to sweep over",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="Number of runs to repeat for each parameter combination",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# main sweep logic
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Initialise vLLM once
    # ------------------------------------------------------------------
    print("Initialising vLLM…")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        dtype=args.dtype,
    )
    print("vLLM initialised.")

    # ------------------------------------------------------------------
    # Build the list of sampling parameter combinations to evaluate
    # ------------------------------------------------------------------
    temperatures: List[float] = args.temperatures
    top_ps: List[float] = args.top_ps
    param_combinations: List[Dict[str, float]] = [
        {"temperature": t, "top_p": p} for t in temperatures for p in top_ps
    ]

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Run evaluation for each parameter combination
    # ------------------------------------------------------------------
    for combo in param_combinations:
        temp = combo["temperature"]
        top_p = combo["top_p"]
        for run_idx in range(args.num_runs):
            print(
                f"\n--- Evaluating temperature={temp} top_p={top_p} run={run_idx} ---"
            )

            vllm_model = VLLM(
                llm=llm,
                batch_size=args.batch_size,
                temperature=temp,
                top_p=top_p,
                max_gen_toks=args.max_tokens,
            )

            results = evaluator.simple_evaluate(
                model=vllm_model,
                tasks=tasks,
                batch_size=args.batch_size,
                limit=args.limit,
                log_samples=False,
                seed=run_idx,
            )

            dumped = json.dumps(results, indent=2)
            filename = (
                output_dir / f"results_temp{temp}_topp{top_p}_run{run_idx}.json"
            )
            with filename.open("w") as f:
                f.write(dumped)
            print(f"Results saved to {filename}")


if __name__ == "__main__":
    main()

