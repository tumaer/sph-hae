import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="tgv", choices=["tgv", "rpf"])
    parser.add_argument(
        "--model_dir", type=str, default=None, help="Path to the model checkpoint."
    )

    # run arguments
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "infer"],
        help="Train or evaluate.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--step_max", type=int, default=1e7, help="Batch size.")
    parser.add_argument(
        "--lr_start", type=float, default=5e-4, help="Starting learning rate."
    )
    parser.add_argument(
        "--lr_final", type=float, default=1e-6, help="Learning rate after decay."
    )
    parser.add_argument(
        "--lr_decay_rate", type=float, default=0.1, help="Learning rate decay."
    )
    parser.add_argument(
        "--lr_decay_steps", type=int, default=5e6, help="Learning rate decay steps."
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=6.7e-5,
        help="Additive noise standard deviation.",
    )
    parser.add_argument(
        "--test",
        default=False,
        help="Run test mode instead of validation.",
    )
    parser.add_argument(
        "--data_dir", type=str, help="Absolute/relative path to the dataset."
    )

    # model arguments
    parser.add_argument(
        "--model", type=str, default="segnn", choices=["gns", "segnn", "haesegnn"]
    )
    parser.add_argument(
        "--input_seq_length",
        type=int,
        default=6,
        help="Input position sequence length.",
    )
    parser.add_argument(
        "--num_mp_steps",
        type=int,
        default=10,
        help="Number of message passing layers.",
    )
    parser.add_argument(
        "--latent_dim", type=int, default=64, help="Hidden layer dimension."
    )
    parser.add_argument(
        "--magnitudes",
        action=argparse.BooleanOptionalAction,
        help="Whether to include velocity magnitudes in node features.",
    )
    parser.add_argument(
        "--isotropic_norm",
        action=argparse.BooleanOptionalAction,
        help="Use isotropic normalization.",
    )

    # segnn-specific arguments
    parser.add_argument(
        "--lmax_attributes",
        type=int,
        default=1,
        help="Maximum degree of attributes.",
    )
    parser.add_argument(
        "--lmax_hidden",
        type=int,
        default=1,
        help="Maximum degree of hidden layers.",
    )
    parser.add_argument(
        "--segnn_norm",
        type=str,
        default="none",
        choices=["instance", "batch", "none"],
        help="Normalisation type.",
    )
    # HAE-specific arguments
    parser.add_argument(
        "--hae_mode",
        type=str,
        required=False,
        choices=["lin", "tp"],
        help="Historical Attribute Embedding type (HAE-SEGNN only).",
    )
    parser.add_argument(
        "--right_attribute",
        default=False,
        help="Whether to use last velocity to steer the attribute embedding.",
    )

    # misc arguments
    parser.add_argument("--seed", type=int, default=6, help="Random seed.")
    parser.add_argument(
        "--gpu", type=int, required=False, help="CUDA device ID to use."
    )
    parser.add_argument(
        "--f64",
        default=False,
        help="Whether to use double precision.",
    )

    parser.add_argument(
        "--eval_n_trajs",
        default=1,
        type=int,
        help="Number of trajectories to evaluate.",
    )

    args = parser.parse_args()

    # PF config
    args.pushforward = {
        "steps": [-1, 100000, 125000, 150000, 175000, 200000],
        "unrolls": [0, 1, 2, 3, 4, 5],
        "probs": [16, 2, 1, 1, 1, 1],
    }

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from sph_hae.run import run

    run(args)
