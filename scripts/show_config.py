import argparse
import json

from coadaptive_vqvae.config.runtime import config_to_dict, update_coadaptive_config, update_vqvae_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show experiment configuration.")
    parser.add_argument("experiment", choices=["vqvae", "coadaptive"])
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--pretrained-vqvae-path", type=str, default=None)
    parser.add_argument("--encoder-model-dir", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.experiment == "vqvae":
        config = update_vqvae_config(
            dataset_root=args.dataset_root,
            log_dir=args.log_dir,
            model_dir=args.model_dir,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            num_workers=args.num_workers,
        )
    else:
        config = update_coadaptive_config(
            dataset_root=args.dataset_root,
            pretrained_vqvae_path=args.pretrained_vqvae_path,
            encoder_model_dir=args.encoder_model_dir,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            num_workers=args.num_workers,
        )
    print(json.dumps(config_to_dict(config), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
