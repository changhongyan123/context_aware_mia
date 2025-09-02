# Objective: compute the probability over each token as the features for each pint in the dataset and save the results in a csv file

import torch
from tqdm import tqdm
import math
from collections import defaultdict
import time
from utils import *

from mimir.config import (
    ExperimentConfig,
)
from omegaconf import DictConfig
import hydra
import os
import pickle
import numpy as np
from typing import List

import mimir.data_utils as data_utils
from mimir.utils import fix_seed
from mimir.models_without_debugging import LanguageModel



def generate_data(
    config: ExperimentConfig,
    dataset: str,
    train: bool = True,
    presampled: str = None,
    specific_source: str = None,
    mask_model_tokenizer=None,
):
    data_obj = data_utils.Data(dataset, config=config, presampled=presampled)
    data = data_obj.load(
        train=train,
        mask_tokenizer=mask_model_tokenizer,
        specific_source=specific_source,
    )
    print(data[0])
    return data
    # return generate_samples(data[:n_samples], batch_size=batch_size)


def get_probability_history(
    data,
    target_model: LanguageModel,
    config: ExperimentConfig,
    n_samples: int = 100,
    batch_size: int = 100,
):
    num_repeatitions = 5 # number of repeatitions
    fix_seed(config.random_seed)
    n_samples = len(data["records"]) if n_samples is None else n_samples
    results = []
    for batch in tqdm(
        range(math.ceil(n_samples / batch_size)), desc=f"Computing criterion"
    ):
        texts = data["records"][batch * batch_size : (batch + 1) * batch_size]

        # For each entry in batch
        for idx in range(len(texts)):
            sample_information = defaultdict(list)
            sample = (
                texts[idx][: config.max_substrs] if config.full_doc else [texts[idx]]
            )

            for idx, substr in enumerate(sample):
                start_time = time.time()
                s_tk_probs, s_all_probs, labels = (
                    target_model.get_probabilities_with_tokens(
                        substr, return_all_probs=True
                    )
                )
                sample_information["tk_probs"].append(s_tk_probs)
                sample_information["labels"].append(labels)

                # consider the repeating the input text
                all_str = substr
                for r_idx in range(num_repeatitions):
                    all_str = all_str + " " + substr
                s_tk_probs, s_all_probs, labels = (
                    target_model.get_probabilities_with_tokens(
                        all_str, return_all_probs=True
                    )
                )
                sample_information[f"tk_probs_repeated_{num_repeatitions}"].append(
                    s_tk_probs
                )
                sample_information[f"labels_repeated_{num_repeatitions}"].append(labels)
            results.append(sample_information)

    return results


def generate_data_processed(
    config, raw_data_member, batch_size: int, raw_data_non_member: List[str] = None
):
    torch.manual_seed(42)
    np.random.seed(42)
    data = {
        "nonmember": [],
        "member": [],
    }

    seq_lens = []
    num_batches = (len(raw_data_member) // batch_size) + 1
    iterator = tqdm(range(num_batches), desc="Generating samples")
    for batch in iterator:
        member_text = raw_data_member[batch * batch_size : (batch + 1) * batch_size]
        non_member_text = raw_data_non_member[
            batch * batch_size : (batch + 1) * batch_size
        ]

        # TODO make same len
        for o, s in zip(non_member_text, member_text):
            if not config.full_doc:
                seq_lens.append((len(s.split(" ")), len(o.split())))

            if config.tok_by_tok:
                for tok_cnt in range(len(o.split(" "))):
                    data["nonmember"].append(" ".join(o.split(" ")[: tok_cnt + 1]))
                    data["member"].append(" ".join(s.split(" ")[: tok_cnt + 1]))
            else:
                data["nonmember"].append(o)
                data["member"].append(s)

    n_samples = len(data["nonmember"])
    return data, seq_lens, n_samples


@hydra.main(config_path="conf", config_name="config")
def main(config: DictConfig):
    freeze_test_set = True
    env_config: EnvironmentConfig = config.env_config
    device = torch.device("cuda")
    fix_seed(config.random_seed)

    exp_name = "construct_dataset_with_repeat_10"

    sf = os.path.join(exp_name, config.base_model.replace("/", "_"))
    new_folder = os.path.join(env_config.results, sf)
    SAVE_FOLDER = new_folder

    if not (os.path.exists(SAVE_FOLDER) or config.dump_cache):
        os.makedirs(SAVE_FOLDER)
    print(f"Saving results to absolute path: {os.path.abspath(SAVE_FOLDER)}")
    cache_dir = env_config.cache_dir
    print(f"LOG: cache_dir is {cache_dir}")
    os.environ["XDG_CACHE_HOME"] = cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    print(f"Using cache dir {cache_dir}")

    # generic generative model
    base_model = LanguageModel(config)
    print("MOVING BASE MODEL TO GPU...", end="", flush=True)
    # base_model.to(device)

    print(f"Loading dataset {config.dataset_nonmember}...")
    data_nonmember = generate_data(
        config,
        config.dataset_nonmember,
        train=False,
        presampled=config.presampled_dataset_nonmember,
        mask_model_tokenizer=None,
    )
    print(f"Loading dataset {config.dataset_member}...")
    data_member = generate_data(
        config,
        config.dataset_member,
        presampled=config.presampled_dataset_member,
        mask_model_tokenizer=None,
    )

    # this is to prepare the text for the members and non-members.
    data, seq_lens, n_samples = generate_data_processed(
        config,
        data_member,
        batch_size=config.batch_size,
        raw_data_non_member=data_nonmember,
    )

    data_members = {
        "records": data["member"],
    }
    data_nonmembers = {
        "records": data["nonmember"],
    }

    member_features = get_probability_history(
        data_members,
        target_model=base_model,
        config=config,
        n_samples=n_samples,
        batch_size=config.batch_size,
    )
    # Collect scores for non-members
    nonmember_features = get_probability_history(
        data_nonmembers,
        target_model=base_model,
        config=config,
        n_samples=n_samples,
        batch_size=config.batch_size,
    )
    with open(
        os.path.join(new_folder, f"all_features_{config.specific_source}.pkl"), "wb"
    ) as f:
        pickle.dump(
            {"member_preds": member_features, "nonmember_preds": nonmember_features}, f
        )

    print(f"Saved results to {new_folder}")


if __name__ == "__main__":
    main()
