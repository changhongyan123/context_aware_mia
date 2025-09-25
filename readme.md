# Context-aware Membership Inference Attacks against Pre-trained Models
[![Webpage](https://img.shields.io/badge/🌐-Webpage-blue)](https://hongyanchang.com/camia/) [![Paper](https://img.shields.io/badge/📄-Paper-red)](URL_TO_BE_PROVIDED)

This repository contains the official code for the paper "Context-aware Membership Inference Attacks against Pre-trained Models", accepted at the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025).


## Installation

```bash
conda create -n mia python=3.10
conda activate mia
pip install -r requirements.txt
```

## Usage

### Running Individual Scripts

- **`run_baselines.py`**: Run baseline attacks and save results in `results_new` folder
- **`run_ref_baselines.py`**: Run attacks on reference models for preparing reference attack
- **`run_ours_construct_mia_data.py`**: Generate train and test data for our attacks
- **`run_ours_train_lr.py`**: Get all our attack results
- **`run_ours_different_agg.py`**: Get our attack results with p-value combination for different aggregations
- **`run_ours_get_roc.py`**: Get the complete ROC curves for our attacks

### Running All Attacks

- **`run.sh`**: Execute all attacks using the provided bash script

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{chang2024context,
  title={Context-aware membership inference attacks against pre-trained large language models},
  author={Chang, Hongyan and Shamsabadi, Ali Shahin and Katevas, Kleomenis and Haddadi, Hamed and Shokri, Reza},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2025}
}
```

## Acknowledgements

This code is based on the [MIMIR codebase](https://github.com/iamgroot42/mimir) under an MIT license. 