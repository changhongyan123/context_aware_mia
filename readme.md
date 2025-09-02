This folder includes the code for paper "Context-aware membership inference attacks against pre-trained models". 



## Installation

```bash
conda create -n mia python=3.10
conda activate mia
pip install -r requirements.txt
```

## Run the code
- `run_baselines.py`: run baseline attacks to save the results in `results_new` folder.
- `run_ref_baselines.py`: run the attacks on the reference models for preparing for the reference attack.
- `run_ours_construct_mia_data.py`: run the code to generate the train and test data for preparing for our attacks.
- `run_ours_ours_train_lr.py`: run the code to get all our attack results.
- `run_ours_ours_different_agg.py`: run the code to get our attack results with p-value combination for different aggregations.
- `run_ours_ours_get_roc.py`: run the code to get the whole roc for our attacks.


## Bash scripts
- `run.sh`: run the code for all the attacks.

## Acknowledgement
This code is based on the MIMIR codebase (https://github.com/grafana/mimir) under an MIT license. 