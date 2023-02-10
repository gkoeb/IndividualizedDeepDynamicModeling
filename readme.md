# Individualized Deep Dynamic Modeling Repository

This is the [Julia](https://julialang.org) implementation of the individualized deep dynamic model as introduced in this [paper](http://dx.doi.org/10.1038/s41598-022-11650-6) and extended to intra-individual sub-periods in a subsequent [publication](https://arxiv.org/abs/2202.07403). The [Jupyter Notebook](notebooks/subperiods.ipynb) demonstrates the approach with simulated data on a high level. Many details about the implementation are documented in the `src/` folder.

In the second publication, we extend the approach to learn coupled intra-individual dynamics in pre-defined sub-periods in real-world data situations (i.e., a small number of variable-length time series, missing observations, and a low signal-to-noise ratio). The strength of the coupling can be controlled via a hyperparameter; see the figure below.

![sp_comp](/figures/sp_changes.png) 