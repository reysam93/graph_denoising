# Robust Graph Filter Identification

This repository contains the code used on the paper "Robust Graph Filter Identification and Graph Denoising from Signal Observations" (pending acceptance). A preliminary version of this paper was presented at ICASSP 2021, in a work titled "Robust Graph-Filter Identification With Graph Denoising Regularization".

You can find the journal paper in arXiv: [https://arxiv.org/abs/2210.08488](https://arxiv.org/abs/2210.08488)

The abstract of the paper reads as follows:

> When facing graph signal processing tasks, the workhorse assumption is that the graph describing the support of the signals is known. However, in many relevant applications _the available graph suffers from observation errors and perturbations_. As a result, any method relying on the graph topology may yield suboptimal results if those imperfections are ignored. Motivated by this, we propose a novel approach for handling perturbations on the links of the graph and apply it to the problem of robust graph-filter identification from input-output observations. Different from existing works, we formulate a non-convex optimization problem that operates in the vertex domain and jointly performs graph-filter identification and graph denoising. As a result, on top of estimating the desired graph filter (GF) at hand, a modified (true) graph is obtained as a byproduct. To handle the resulting bi-convex problem, we design an algorithm that blends techniques from alternating optimization and majorization minimization, showing its convergence to a stationary point. The second part of the paper i) generalizes the design to a robust setup where several GFs (all defined over the same graph) are jointly estimated, and ii) introduces an alternative algorithmic implementation that reduces the computational complexity when dealing with large graphs. Finally, the detrimental influence of the perturbations and the benefits resulting from the robust approach are numerically analyzed over synthetic and real-world datasets, comparing them with other state-of-the-art alternatives.

## Structure of the Repository

The repository is organized as follows:

* `ICASSP21`: contains the code for the experiments of the preliminary work presented in the conference ICASSP 2021. The code is written in MATLAB.
* `synth_data`: contains the scripts for the sythentic data experiments, presented in section VII.A of the paper.
* `temp_data`: contains the scripts for the first real-world dataset in the paper, about predicting temperatures with past data. In this experiment, we postulate the relationship through the multiplication with a matrix, which, in our algorithm, is a graph filter. The folder is organized as follows:
  * The data folder contains both the scripts to process the raw data downloaded from the GSOD dataset, along with the processed data files used in the scripts. Raw data is not included, but can be downloaded from [the GSOD webpage](https://www.ncei.noaa.gov/data/global-summary-of-the-day/archive/). 
  * The notebook `test_temp_dataset` contains the code for a simple test with pre-specified parameters.
  * The files `temp_dataset.py` and `temp_dataset_steps.py` contain the code for the experiments presented in the paper, either modifying the percentage of samples used for training or the time horizon to make the prediction. 
* `airQual_data`: a folder with a similar structure than the previous one, in this case for the second real-world experiment in the paper about predicting air quality data using past measurements. In this case, the database used is from the United States Environmental Protection Agency, and can be accessed through [their webpage](https://www.epa.gov/outdoor-air-quality-data).
* The optimization scripts contain the code for the different algorithms proposed in the paper, specifically:
  * `opt.py`: contains the code for the base robust filter identification algorithm in the paper, both the basic implementation and the one that substitutes the L-1 norm with a logarithmic penalty.
  * `sev_filters_opt.py`: same algorithms for the problem where we are trying to identify several filters over the same graph. 
  * `robust_ARopt.py`: same algorithms for the AR model. 
  * `opt_lls.py`: implements one of the algorithms used for the comparison with our techniques, specifically SCP.
  * `opt_efficient.py`: contains the efficient implementation of our algorithm, with reduced computational complexity and better suited for large graphs.

## Citing

If you use this code or find the paper relevant, please cite the paper as follows:
```
@misc{rey22robustgfi,
  url = {https://arxiv.org/abs/2210.08488},
  author = {Rey, Samuel and Tenorio, Victor M. and Marques, Antonio G.},
  title = {Robust Graph Filter Identification and Graph Denoising from Signal Observations},
  year = {2022}
}
```
