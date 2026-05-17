#!/usr/bin/env python
import os
import sys
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='RD-VLA Training and Evaluation')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], required=True)
    args, remaining = parser.parse_known_args()
    return args, remaining


def train(config_path: str, cli_overrides: list):
    from configs import TrainConfig, load_config, get_legacy_config, save_config

    cfg = load_config(TrainConfig, config_path, cli_overrides)

    run_dir = Path(cfg.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / 'config.yaml')

    legacy_cfg = get_legacy_config(cfg, 'train')

    sys.path.insert(0, 'vla-scripts')
    from finetune import finetune
    finetune(legacy_cfg)


def evaluate(config_path: str, cli_overrides: list):
    from configs import EvalConfig, load_config, get_legacy_config

    cfg = load_config(EvalConfig, config_path, cli_overrides)
    legacy_cfg = get_legacy_config(cfg, 'eval')

    from experiments.robot.libero.run_libero_eval import eval_libero
    eval_libero(legacy_cfg)


def main():
    args, remaining = parse_args()
    if args.mode == 'train':
        train(args.config, remaining)
    else:
        evaluate(args.config, remaining)


if __name__ == '__main__':
    main()
