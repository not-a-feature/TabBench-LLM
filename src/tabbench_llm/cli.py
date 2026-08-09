"""Command-line interface for TabBench-LLM.

Usage
-----
::

    # Run the full pipeline (predictions → metrics) for one grid cell's config
    tabbench-llm run --config results/feature_sweep/cap_full/config.json

    # Run individual steps
    tabbench-llm run --config <cfg> --step predictions
    tabbench-llm run --config <cfg> --step metrics

    # Show a leaderboard built from a results directory
    tabbench-llm leaderboard --results-dir results/feature_sweep/cap_full

    # Build the leaderboard payload from a results directory
    tabbench-llm site --results-dir results/feature_sweep/cap_full --out site_data

    # Package and dataset info
    tabbench-llm info

See Also
--------
- Source: https://github.com/not-a-feature/TabBench-LLM
"""

import argparse
import logging
import sys
import warnings

warnings.filterwarnings("ignore", message="'force_all_finite' was renamed")

from tabbench_llm.logging_utils import LOG_FORMAT  # noqa: E402


def cmd_run(args):
    """Run the benchmark pipeline (predictions → metrics)."""
    from tabbench_llm.config import load_config

    config = load_config(args.config)

    if args.output:
        config["output_dir"] = args.output

    if args.model:
        config["models"] = [args.model]

    step = args.step

    if step == "prepare":
        from tabbench_llm.predictions import prepare_splits

        prepare_splits(config, seed_index=args.seed_index)
        return

    if step in ("all", "predictions"):
        from tabbench_llm.predictions import compute_predictions

        compute_predictions(
            config,
            seed_index=args.seed_index,
            overwrite=args.overwrite,
            reverse=args.reverse,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
        )

    if step in ("all", "metrics"):
        from tabbench_llm.evaluation import compute_metrics_from_predictions

        compute_metrics_from_predictions(config)


def cmd_leaderboard(args):
    """Print a leaderboard built from a results directory."""
    from tabbench_llm.leaderboard import Leaderboard

    lb = Leaderboard.from_results_dir(args.results_dir)
    print(lb.summary())

    if args.plot:
        fig = lb.plot(task=args.task)
        fig.savefig("leaderboard.png", dpi=150, bbox_inches="tight")
        print("Saved leaderboard.png")


def cmd_site(args):
    """Build the leaderboard payload the public page consumes."""
    from tabbench_llm.site import build_site

    path = build_site(
        results_dir=args.results_dir,
        out_dir=args.out,
        config_path=args.config,
        title=args.title,
    )
    print(f"Wrote leaderboard payload to {path}")
    print(f"Publish: commit {args.out}/ and push to main; the upload workflow ships it.")


def cmd_info(_args):
    """Print package and ecosystem info."""
    import tabbench_llm

    print(f"tabbench-llm {tabbench_llm.__version__}")
    print()
    print("Controlled evaluation of LLMs as few-shot tabular classifiers.")
    print()
    print("  Source: https://github.com/not-a-feature/TabBench-LLM")


def main():
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    parser = argparse.ArgumentParser(
        prog="tabbench-llm",
        description="TabBench-LLM — benchmark for LLM few-shot tabular classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ---- run ----
    run_p = sub.add_parser("run", help="Run the benchmark pipeline")
    run_p.add_argument("--config", required=True, help="Path to config JSON")
    run_p.add_argument(
        "--step",
        choices=["all", "prepare", "predictions", "metrics"],
        default="all",
        help="Pipeline step (default: all). 'prepare' only warms the split cache.",
    )
    run_p.add_argument("--output", default=None, help="Override output directory")
    run_p.add_argument("--seed-index", type=int, default=None, help="Run only this seed index")
    run_p.add_argument("--model", default=None, help="Run only this model")
    run_p.add_argument("--overwrite", action="store_true", help="Overwrite existing predictions")
    run_p.add_argument("--reverse", action="store_true", help="Iterate datasets in reverse order")
    run_p.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the model x dataset grid across this many workers (multi-GPU).",
    )
    run_p.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Which shard this worker runs (0 <= shard-index < num-shards).",
    )
    run_p.set_defaults(func=cmd_run)

    # ---- leaderboard ----
    lb_p = sub.add_parser("leaderboard", help="Show a leaderboard from a results directory")
    lb_p.add_argument("--results-dir", required=True, help="Pipeline results directory")
    lb_p.add_argument(
        "--task",
        choices=["overall", "classification", "regression"],
        default="overall",
    )
    lb_p.add_argument("--plot", action="store_true", help="Save leaderboard.png")
    lb_p.set_defaults(func=cmd_leaderboard)

    # ---- site ----
    site_p = sub.add_parser("site", help="Build the leaderboard payload for the public page")
    site_p.add_argument("--results-dir", required=True, help="Pipeline results directory")
    site_p.add_argument("--out", default="site_data", help="Output directory (default: site_data)")
    site_p.add_argument(
        "--config",
        default=None,
        help="Optional benchmark config (used to look up dataset feature counts)",
    )
    site_p.add_argument(
        "--title",
        default="TabBench-LLM Leaderboard",
        help="Page title",
    )
    site_p.set_defaults(func=cmd_site)

    # ---- info ----
    info_p = sub.add_parser("info", help="Show package and ecosystem info")
    info_p.set_defaults(func=cmd_info)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
