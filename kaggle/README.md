# Kaggle Setup Guide

This directory contains files and resources specific to running the project on Kaggle. Below are the key components:

1. **Kernel Metadata**: The `kernel-metadata.json` file contains metadata about the Kaggle kernel, including the title, description, and tags.

2. **Dataset Metadata**: The `dataset-metadata.json` file provides information about the datasets used in this project, including their sources and descriptions.

3. **Requirements**: The `requirements_kaggle.txt` file lists all the Python dependencies required to run the project on Kaggle. Ensure that these packages are installed in your Kaggle environment.

4. **Kaggle Workflows**: The `kaggle_workflows` directory contains Jupyter notebooks that outline the workflow for data preparation, training, and evaluation. These notebooks are designed to be run sequentially to facilitate the entire project lifecycle on Kaggle.

### Usage Instructions

- To get started, open the `00_setup_and_installation.ipynb` notebook in the `notebooks` directory to set up your environment.
- Follow the notebooks in the `kaggle_workflows` directory for a step-by-step guide on preparing data, training models, and evaluating results.
- Make sure to check the `requirements_kaggle.txt` file to install any necessary dependencies before running the notebooks.

For any issues or contributions, please refer to the main project documentation in the root directory.