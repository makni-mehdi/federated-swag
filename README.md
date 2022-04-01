# Uncertainty Quantification in Federated Learning

This repository contains a PyTorch code for the experiments conducted in my Bachelor Thesis.

[Deriving FL-SWAG for the sake of calibration and privacy protection](https://github.com/makni-mehdi/federated-swag/blob/main/Bachelor%20Thesis%20Report.pdf)

Bachelor Thesis Report at [École Polytechnique](https://www.polytechnique.edu/en)  and the [Lagrange Mathematics and Computing Research Center](https://www.huawei.com/fr/news/fr/2020/centre-lagrange).

Advisors: [Mérouane Debbah](https://en.wikipedia.org/wiki/M%C3%A9rouane_Debbah) (Director of research at the Lagrange Mathematics and Computing Research Center) and [Éric Moulines](https://en.wikipedia.org/wiki/%C3%89ric_Moulines) (French Academy of Science 2017)

Much of the internship was also supervised by [Vincent Plassier](https://www.linkedin.com/in/vincent-plassier-179161172/?originalSubdomain=fr) (PhD student at École Polytechnique and Lagrange Mathematics and Computing Research Center) who also contributed to the code and guided the research project.

## Abstract

In this Bachelor Thesis report, we study one-shot methods whose objective is to obtain a well-calibrated model that results from a unique consensus step of the parameters of models trained in a federated fashion. To this end, we introduce uncertainty quantification and the calibration scores known in the field as well as the constraints of the Federated Learning setting. 
We motivate the problem of finding well-calibrated models that respect privacy issues encountered in Federated Learning and explore the performance of SWAG, a rising star in uncertainty quantification, by comparison to the results of a ground-truth Hamiltonian Monte Carlo sampling method which is prohibitively expensive in the context of Deep Learning. We finally derive a highly-efficient, SWAG-inspired, last-layer model that trains in a distributed way to allow clients to collaborate and solve a machine learning task without data sharing, while ensuring the calibration of their outputs. 
The results of the experiments are performed on MNIST, FashionMNIST and CIFAR 10 datasets and benchmarked against leading models in uncertainty quantification like SGLD and pSGLD.


## SWAG - HMC Toyish Comparison

We start by building a toy example and try to visualize and quantify to what extent can the SWAG algorithm be close to Hamiltonian Monte Carlo method which samples asymptotically from the true posterior distribution. The high-dimensional space where weights lie is projected into a plance obtained thanks to PCA (Principal Compenent Analysis).


![Alt text](codes/Visualizing%20posterior%20distribution%20in%20PCA%20subspace/hmc_sample_2d.pdf?raw=true "HMC samples")

![Alt text](/blob/main/codes/Visualizing%20posterior%20distribution%20in%20PCA%20subspace/swag_samples_2d.pdf?raw=true "SWAG samples")

![Alt text](/blob/main/codes/Visualizing%20posterior%20distribution%20in%20PCA%20subspace/posterior_distributions_subspace.pdf?raw=true "Posterior Distributions")


## FL-SWAG Comparison to SGLD and pSGLD on CIFAR 10

![Alt text](/blob/main/codes/Visualizing%20posterior%20distribution%20in%20PCA%20subspace/posterior_distributions_subspace.pdf?raw=true "Posterior Distributions")

![Alt text](/blob/main/codes/Visualizing%20posterior%20distribution%20in%20PCA%20subspace/posterior_distributions_subspace.pdf?raw=true "Posterior Distributions")
