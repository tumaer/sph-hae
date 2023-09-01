from typing import Callable, Tuple, Type

from e3nn_jax import Irreps

from lagrangebench import models, H5Dataset
from lagrangebench.models.utils import node_irreps
from lagrangebench.utils import NodeType

from .haesegnn import HAESEGNN


def setup_data(args):
    # dataloader
    train_seq_l = args.input_seq_length + args.pushforward["unrolls"][-1]

    if args.dataset.lower() == "tgv":
        dataset_path = "datasets/3D_TGV_8000_10kevery100"
        split_valid_traj_into_n = 1
    elif args.dataset.lower() == "rpf":
        dataset_path = "datasets/3D_RPF_8000_10kevery100"
        split_valid_traj_into_n = 20
    else:
        raise ValueError(f"Unknown dataset {args.dataset}.")

    data_train = H5Dataset(
        "train",
        dataset_path=dataset_path,
        name=args.dataset.lower(),
        input_seq_length=train_seq_l,
        split_valid_traj_into_n=split_valid_traj_into_n,
        is_rollout=False,
    )
    data_eval = H5Dataset(
        "test" if args.test else "valid",
        dataset_path=dataset_path,
        name=args.dataset.lower(),
        input_seq_length=args.input_seq_length,
        split_valid_traj_into_n=split_valid_traj_into_n,
        is_rollout=True,
    )

    return data_train, data_eval


def setup_model(args) -> Tuple[Callable, Type]:
    """Setup model based on args."""
    model_name = args.model.lower()
    metadata = args.metadata

    if model_name == "gns":

        def model_fn(x):
            return models.GNS(
                particle_dimension=metadata["dim"],
                latent_size=args.latent_dim,
                num_mlp_layers=1,
                num_mp_steps=args.num_mp_steps,
                num_particle_types=NodeType.SIZE,
                particle_type_embedding_size=16,
            )(x)

        MODEL = models.GNS
    elif model_name == "segnn":
        # segnn with average attribute aggregation

        # Hx1o vel, Hx0e vel, 2x1o boundary, 9x0e type
        node_feature_irreps = node_irreps(
            metadata,
            args.input_seq_length,
            args.has_external_force,
            args.magnitudes,
            True,
        )
        # 1o displacement, 0e distance
        edge_feature_irreps = Irreps("1x1o + 1x0e")

        def model_fn(x):
            return models.SEGNN(
                node_features_irreps=node_feature_irreps,
                edge_features_irreps=edge_feature_irreps,
                scalar_units=args.latent_dim,
                lmax_hidden=args.lmax_hidden,
                lmax_attributes=args.lmax_attributes,
                output_irreps=Irreps("1x1o"),
                num_layers=args.num_mp_steps,
                n_vels=args.input_seq_length - 1,
                homogeneous_particles=True,
                velocity_aggregate="avg",
                blocks_per_layer=2,
                norm=args.segnn_norm,
            )(x)

        MODEL = models.SEGNN
    elif model_name == "haesegnn":
        # segnn with linear/tensor product attribute aggregation

        assert args.hae_mode is not None, "HAE mode must be specified for HAE-SEGNN."
        # Hx1o vel, Hx0e vel, 2x1o boundary, 9x0e type
        node_feature_irreps = node_irreps(
            metadata,
            args.input_seq_length,
            args.dataset == "rpf",
            args.magnitudes,
            True,
        )
        # 1o displacement, 0e distance
        edge_feature_irreps = Irreps("1x1o + 1x0e")

        def model_fn(x):
            return HAESEGNN(
                node_features_irreps=node_feature_irreps,
                edge_features_irreps=edge_feature_irreps,
                scalar_units=args.latent_dim,
                lmax_hidden=args.lmax_hidden,
                lmax_attributes=args.lmax_attributes,
                output_irreps=Irreps("1x1o"),
                num_layers=args.num_mp_steps,
                n_vels=args.input_seq_length - 1,
                homogeneous_particles=True,
                blocks_per_layer=2,
                norm=args.segnn_norm,
                right_attribute=args.right_attribute,
                attribute_embedding_blocks=(1 if args.hae_mode == "lin" else 2),
            )(x)

        MODEL = models.SEGNN

    return model_fn, MODEL
