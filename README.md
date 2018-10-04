# Landmine Detection Using Autoencoders on Multi-polarization GPR Volumetric Data

This repository contains a running example and part of the dataset related to the paper:

Paolo Bestagini, Federico Lombardi, Maurizio Lualdi, Francesco Picetti, Stefano Tubaro, *Landmine Detection Using Autoencoder on Multi-polarization GPR Volumetric Data*, Oct. 2018, https://arxiv.org/abs/1810.01316

### Abstract

Buried landmines and unexploded remnants of war are a constant threat for the population of many countries that have been hit by wars in the past years.
The huge amount of human lives lost due to this phenomenon has been a strong motivation for the research community toward the development of safe and robust techniques designed for landmine clearance.
Nonetheless, being able to detect and localize buried landmines with high precision in an automatic fashion is still considered a challenging task due to the many different boundary conditions that characterize this problem (e.g., several kinds of objects to detect, different soils and meteorological conditions, etc.).
In this paper, we propose a novel technique for buried object detection tailored to unexploded landmine discovery.
The proposed solution exploits a specific kind of convolutional neural network (CNN) known as autoencoder to analyze volumetric data acquired with ground penetrating radar (GPR) using different polarizations.
This method works in an anomaly detection framework, indeed we only train the autoencoder on GPR data acquired on landmine-free areas.
The system then recognizes landmines as objects that are dissimilar to the soil used during the training step.
Experiments conducted on real data show that the proposed technique requires little training and no ad-hoc data pre-processing to achieve accuracy higher than 93% on challenging datasets.
