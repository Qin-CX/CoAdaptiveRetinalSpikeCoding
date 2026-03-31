import argparse

from coadaptive_vqvae.config.runtime import update_coadaptive_config
from coadaptive_vqvae.training.coadaptive import main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the co-adaptive retinal encoder.")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--pretrained-vqvae-path", type=str, default=None)
    parser.add_argument("--encoder-model-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def run() -> None:
    args = parse_args()
    config = update_coadaptive_config(
        dataset_root=args.dataset_root,
        pretrained_vqvae_path=args.pretrained_vqvae_path,
        encoder_model_dir=args.encoder_model_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
    )
    main(config=config, device=args.device)


if __name__ == "__main__":
    run()
