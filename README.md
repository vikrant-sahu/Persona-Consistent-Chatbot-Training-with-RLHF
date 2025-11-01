# Persona Consistent Chatbot

This project aims to develop a persona-consistent chatbot using advanced machine learning techniques, including supervised fine-tuning (SFT) and reinforcement learning from human feedback (RLHF). The chatbot is designed to maintain a consistent persona throughout conversations, enhancing user engagement and interaction quality.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Data](#data)
- [Training](#training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd persona-consistent-chatbot
pip install -r requirements.txt
```

## Usage

After installation, you can run the chatbot using the provided scripts or Jupyter notebooks. For an interactive demo, use the Gradio app:

```bash
python scripts/inference/demo_app.py
```

## Configuration

Configuration files are located in the `configs` directory. You can modify these files to adjust settings for data paths, model architecture, training parameters, and evaluation benchmarks.

## Data

The project utilizes various datasets, including PersonaChat and Blended Skill Talk. Raw datasets are stored in the `data/raw` directory, while processed data can be found in `data/processed`.

## Training

Training scripts are available in the `scripts/training` directory. You can run the following commands to train different components of the model:

- Supervised Fine-Tuning: `python scripts/training/01_train_sft.py`
- Reward Model Training: `python scripts/training/02_train_reward_model.py`
- PPO/RLHF Training: `python scripts/training/03_train_ppo.py`

## Evaluation

Evaluation scripts are located in the `scripts/evaluation` directory. Use these scripts to assess the performance of the chatbot on various metrics, including persona consistency and engagement.

## Contributing

Contributions are welcome! Please refer to the `CONTRIBUTING.md` file for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Citation

If you use this project in your research, please cite it using the information in `CITATION.bib`.