import os
from datetime import datetime

import jax.numpy as jnp
import jmp
import haiku as hk
import numpy as np

from lagrangebench import case_builder, Trainer
from lagrangebench.utils import PushforwardConfig
from lagrangebench.evaluate import infer, averaged_metrics

from .utils import setup_data, setup_model


def run(args):
    data_train, data_eval = setup_data(args)

    # neighbors search
    bounds = np.array(data_train.metadata["bounds"])
    args.box = bounds[:, 1] - bounds[:, 0]

    # setup core functions
    case = case_builder(
        box=args.box,
        metadata=data_train.metadata,
        input_seq_length=args.input_seq_length,
        isotropic_norm=args.isotropic_norm,
        noise_std=args.noise_std,
        magnitude_features=args.magnitudes,
        external_force_fn=data_train.external_force_fn,
        dtype=(jnp.float64 if args.f64 else jnp.float32),
    )

    args.metadata = data_train.metadata
    args.normalization_stats = case.normalization_stats
    args.has_external_force = data_train.external_force_fn is not None

    # setup model from configs
    model, MODEL = setup_model(args)
    model = hk.without_apply_rng(hk.transform_with_state(model))

    # mixed precision training based on this reference:
    # https://github.com/deepmind/dm-haiku/blob/main/examples/imagenet/train.py
    policy = jmp.get_policy("params=float32,compute=float32,output=float32")
    hk.mixed_precision.set_policy(MODEL, policy)

    if args.mode == "train":
        # save config file
        run_prefix = f"{args.model}_{data_train.name}"
        data_and_time = datetime.today().strftime("%Y%m%d-%H%M%S")
        run_name = f"{run_prefix}_{data_and_time}"

        new_checkpoint = os.path.join("ckp", run_name)

        pf_config = PushforwardConfig(
            steps=args.pushforward["steps"],
            unrolls=args.pushforward["unrolls"],
            probs=args.pushforward["probs"],
        )

        trainer = Trainer(
            model,
            case,
            data_train,
            data_eval,
            pushforward=pf_config,
            metrics=["mse"],
            seed=args.seed,
            batch_size=args.batch_size,
            input_seq_length=args.input_seq_length,
            noise_std=args.noise_std,
            lr_start=args.lr_start,
            lr_final=args.lr_final,
            lr_decay_steps=args.lr_decay_steps,
            lr_decay_rate=args.lr_decay_rate,
            n_rollout_steps=10,  # 1000 real simulator steps
            eval_n_trajs=args.eval_n_trajs,
        )
        trainer(
            step_max=args.step_max,
            load_checkpoint=args.model_dir,
            store_checkpoint=new_checkpoint,
        )
    elif args.mode == "infer":
        assert args.model_dir, "model_dir must be specified for inference."
        metrics = infer(
            model,
            case,
            data_eval,
            load_checkpoint=args.model_dir,
            metrics=["mse", "sinkhorn", "e_kin"],
            eval_n_trajs=args.eval_n_trajs,
            n_rollout_steps=10,  # 1000 real simulator steps
            seed=args.seed,
        )

        print(averaged_metrics(metrics))
