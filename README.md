# Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting implemented in Pytorch
Authors: Bryan Lim, Sercan Arik, Nicolas Loeff and Tomas Pfister

Paper Link: [https://arxiv.org/pdf/1912.09363.pdf](https://arxiv.org/pdf/1912.09363.pdf)

# Implementation
This repository contains the source code for the Temporal Fusion Transformer reproduced in Pytorch using [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) which is used to scale models and write less boilerplate . In the moment, the model is trained with the Electricity dataset from the paper. However, im currently working on the code to allow the use of the other 3 datasets described in the paper and reproduce the results.

- **data_formatters**: Stores the main dataset-specific column definitions, along with functions for data transformation and normalization. For compatibility with the TFT, new experiments should implement a unique GenericDataFormatter (see base.py), with examples for the default experiments shown in the other python files.

- **data**: Stores the main dataset-specific download procedure, along with the pytorch dataset class ready to use as input to the dataloader and then the model.

# Training
To run the training procedure, open up **training_tft.ipynb** and execute all cells to start training.
