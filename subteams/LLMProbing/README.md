# Probing Sub-Team of the Mechanistic Interpretability via Learning Differential Equations (MIVLDE) Project
## Table of Contents:
- [Contributors](#contributors)
- [Description](#description)
- [Workflow](#workflow)
	- [Generating Sample Data](#generating-sample-data)
	- [Extracting Activations](#extracting-activations)
	- [Training Probes](#training-probes)
	- [Evaluating Probes](#evaluating-probes)
- [Experiments](#experiments)
## Contributors
Probing Sub-Team Lead: [*Dylan Ponsford*](https://github.com/dylanxp)

Probing Sub-Team Members: [*Axel Ahlqvist*](https://github.com/PotentialKillScreen), [*Helen Saville*](https://github.com/helensaville), *Melwina Albuquerque*
## Description
This section of the repo contains the probing sub-team's tools, and some notebooks of our experiments. 

Our work aims to using probing as a method to try to understand different components of the ODEFormer. Probing has been demonstrated to be a method that works well for studying LLMs, and so it seems reasonable that it should also be a useful method for studying the ODEFormer.

The main tools we use to work towards this goal are classifier probes (implemented via logistic regression), and regression probes (implemented via linear regression). Previous works have demonstrated the effectiveness of such tools: for example, [Marks & Tegmark, 2023](https://arxiv.org/abs/2310.06824) for classifier probes, and [Gurnee & Tegmark, 2024](https://arxiv.org/abs/2310.02207) for regression probes.

We design various experiments to test different parts of the ODEFormer. Each experiment requires a well-defined feature associated with each sample. For classifier probes, this is a binary label, and for regression probes, this is a scalar value.

The challenge in using probing for this purpose is to design meaningful features which will allow us to start determining how to ODEFormer comes to its predicted equation(s). Some of our experiments can be found in the `/notebooks/` directory, and will be described later in the [Experiments](#experiments) section.

Experiments can be created using the tools in this sub-repo by cloning the repo, and then creating notebooks in the `/notebooks/` directory and importing functionality from `/src/` or `/experiments/`.
## Workflow
The package contains various tools for setting up and running experiments. The pipeline for running experiments is detailed in the following diagram.

![experiment_pipeline_updated](https://github.com/user-attachments/assets/3d788c3d-d903-4908-97fa-2c1c26159539)
### Generating Sample Data
For sample generation, we can either generate samples manually, which requires defining and integrating a specified ODE (via functionality from scikit-learn), or generate samples randomly, which utilises the pre-training data creation process used in the original ODEFormer paper ([contained in Section 3](https://arxiv.org/abs/2310.05573)). These are implemented in the `src.sample_generation.sample_generators.ManualSamplesGenerator` and `src.sample_generation.sample_generators.RandomSamplesGenerator` classes respectively.

The choice of which type of generation to use depends on what kind of experiment is to be run. The class `ManualSamplesGenerator` cannot generate data for arbitrary equations, so new functionality would have to be added in order to generate custom samples. Currently, this class can generate exponential, hyperbolic, and sigmoid samples, which are used in some of the experiments below. The class `RandomSamplesGenerator` takes various parameters which can control the type of samples which are produced.

This section of the pipeline is where most of the features to be studied with the probe must be created. For example, this could be whether the sample is of the exponential or hyperbolic type, or for a sigmoid sample, what the maximum derivative is or where the inflection point lies.

Once created, the path to the raw samples can be passed to `src.datasets.samples_dataset.SamplesDataset` to be wrapped in a PyTorch Dataset object.

Note: the `parsers.py` file contained within `src/sample_generation` is copied from [the original ODEFormer repo](https://github.com/sdascoli/odeformer/blob/main/parsers.py).
### Extracting Activations
Once samples have been generated, we need to extract activations from the ODEFormer to create data for the probes to use. This functionality is implemented in `src.activation_extraction.ActivationsExtractor`. This takes an ODEFormer object and the path to the samples generated in the previous step, and then extracts and saves activations from all encoder and decoder layers after their FeedForward layers by default.

The `ActivationsExtractor` class also computes the [R^2 score](https://en.wikipedia.org/wiki/Coefficient_of_determination) of the trajectory under the predicted equation vs the input trajectory, and adds this as another feature to the sample. This allows for later filtering of samples to only allow those for which the ODEFormer has good (numerical) performance.

Similarly to before, once created, the path to the raw activations can be passed to `src.datasets.activations_dataset.ActivationsDataset` to be wrapped in a PyTorch Dataset object. The `ActivationsDataset` class also takes an optional parameter `r2_threshold`, which allows for the filtering of samples described above. This can exclude outliers which affect the probes performance later.
### Training Probes
Once the datasets above have been created, we train probes to predict the desired feature. If the feature is a binary label, then classifier probes can be trained, and if the feature is a scalar value, then regression probes can be trained.

Training can either be done via gradient-based training using PyTorch functionality, or using a direct numerical solver which utilises functionality from scikit-learn. Through our testing, we found that the solver appeared to be more reliable, but for large datasets, it may make sense to use gradient-based training.

Classifier probes can be trained via `src.probes.utils.train_classifier_probe` for gradient-based training, and `src.probes.utils.train_classifier_probe_w_solver` for the numerical solver. Similarly, regression probes can be trained via  `src.probes.utils.train_regression_probe` for gradient-based training, and `src.probes.utils.train_regression_probe_w_solver` for the numerical solver.

These functions work on the `src.probes.lr_probe.LRProbe` class, which has a single hidden layer of a specified size. This class can be used either for classifier or regression probes.
### Evaluating Probes
To evaluate regression probes, you can simply use `src.probes.utils.eval_regression_probe`. 

However, for classifier probes, you must know how the probes was trained, as this currently requires additional functionality. For gradient-based probes, you can use `src.probes.utils.eval_classifier_probe`, and for solver-based probes, you can use `src.probes.utils.eval_solver_classifier_probe`.

To get the full list of probe outputs for a regression probe, `src.probes.utils.verbose_eval_regression_probe` can be used. This is useful for plots of ground truth against probe prediction.
## Experiments
Notebooks of our experiments can be found in the `/notebooks/` directory. In order to wrap up experiments into an easy to use and run function, the `/experiments/` directory contains various experiment runners, which allow you to specify the desired feature, and to train and save probes of the desired type over specified layers. This requires the samples and activations to already have been generated, using the functionality described above.

To run a classifier probes experiment, either the `experiments.run_experiment.separability_testing` or `experiments.run_experiment.separability_testing_w_solver` functions can be used.

To run a regression probes experiment, either `experiments.run_experiment.scalar_prediction_experiment` or `experiments.run_experiment.scalar_prediction_experiment_w_solver` functions can be used.

Our experiments in `/notebooks/` include:
- Prediction of ODE dimension, in `dimensions.ipynb`
- Prediction of R^2 score, across `r2_experiment.ipynb`, `more_layers_r2.ipynb`, and `r2_experiment_w_solver.ipynb`
- Prediction of the derivatives at time point 0 and time point 3, in `derivatives_exp.ipynb` and `derivatives_exp_w_solver.ipynb`
- Prediction of the sigmoid inflection point and maximum derivative, in `derivatives_sigmoid.ipynb`
- A preliminary notebook containing prediction of the eigenvalues and coefficients of a 2D linear system, in `2D_Linear.ipynb`

As a starting point for future experiments, `demo_classification_with_solver.ipynb` shows how to train and run classifier probe experiments, and `r2_experiment_w_solver.ipynb` shows how to train and run regression probe experiments with the most up to date functionality.
