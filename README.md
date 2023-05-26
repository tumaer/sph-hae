# Learning Lagrangian Fluid Mechanics with E(3)-Equivariant Graph Neural Networks

## :warning: Work in progress! :warning:

__Jax__ implementation of:

__Learning Lagrangian Fluid Mechanics with E(3)-Equivariant GNNs__<br>
Artur P. Toshev, Gianluca Galletti, Johannes Brandstetter, Stefan Adami and Nikolaus A. Adams.<br>
https://arxiv.org/abs/2305.15603

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="/assets/gsi_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="/assets/gsi.png">
  <img alt="Left: time snapshots of velocity magnitude of Taylor-Green vortex flow (top), reverse Poiseuille flow (bottom). Right: attribute embedding model (top), effect of different embedding strategies on velocity (bottom)." src="/assets/gsi.png">
</picture>


>__Abstract:__ We contribute to the vastly growing field of machine learning for engineering systems by demonstrating that equivariant graph neural networks have the potential to learn more accurate dynamic-interaction models than their non-equivariant counterparts. We benchmark two well-studied fluid-flow systems, namely 3D decaying Taylor-Green vortex and 3D reverse Poiseuille flow, and evaluate the models based on different performance measures, such as kinetic energy or Sinkhorn distance. In addition, we investigate different embedding methods of physical-information histories for equivariant models. We find that while currently being rather slow to train and evaluate, equivariant models with our proposed history embeddings learn more accurate physical interactions.


## Citing
This codebase was created by Artur Toshev and Gianluca Galletti. If you use our work in your research, please cite it:
```bibtex
@misc{toshev2023learning,
      title={Learning Lagrangian Fluid Mechanics with E($3$)-Equivariant Graph Neural Networks}, 
      author={Artur P. Toshev and Gianluca Galletti and Johannes Brandstetter and Stefan Adami and Nikolaus A. Adams},
      year={2023},
      eprint={2305.15603},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
