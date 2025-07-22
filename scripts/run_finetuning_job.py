#!/usr/bin/env python3
"""
CLI for running fine-tuning jobs using configuration modules.

Usage:
    python scripts/run_finetuning_job.py cfgs/my_finetuning_config.py cfg_var_name
"""

import argparse
import asyncio
import sys
from pathlib import Path
from loguru import logger
from sl.finetuning.services import Cfg, run_finetuning_job
from sl.utils import module_utils


async def main():
    parser = argparse.ArgumentParser(
        description="Run fine-tuning job using a configuration module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_finetuning_job.py cfgs/my_finetuning_config.py my_cfg
        """,
    )

    parser.add_argument(
        "config_module",
        help="Path to Python module containing fine-tuning configuration",
    )

    parser.add_argument(
        "cfg_var_name",
        nargs="?",
        default="cfg",
        help="Name of the configuration variable in the module (default: 'cfg')",
    )

    args = parser.parse_args()

    # Validate config file exists
    config_path = Path(args.config_module)
    if not config_path.exists():
        logger.error(f"Config module {args.config_module} does not exist")
        sys.exit(1)

    try:
        # Load configuration from module
        logger.info(
            f"Loading configuration from {args.config_module} (variable: {args.cfg_var_name})..."
        )
        cfg = module_utils.get_obj(args.config_module, args.cfg_var_name)
        assert isinstance(cfg, Cfg)

        # Run fine-tuning job
        logger.info("Starting fine-tuning job...")
        await run_finetuning_job(cfg)
        logger.success("Fine-tuning job completed successfully!")

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
