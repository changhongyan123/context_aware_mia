import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
from tqdm import tqdm
import datetime
import os
import json
import math
from collections import defaultdict
from typing import List, Dict

import argparse
from utils import *

from mimir.config import (
    ExperimentConfig,
    EnvironmentConfig,
)
import torch.nn.functional as F

import zlib
import mimir.data_utils as data_utils
import mimir.plot_utils as plot_utils
from mimir.utils import fix_seed
from mimir.models_without_debugging import LanguageModel, ReferenceModel, OpenAI_APIModel
from mimir.attacks.all_attacks import AllAttacks, Attack
from mimir.attacks.utils import get_attacker
from mimir.attacks.attack_utils import (
    get_roc_metrics,
    get_precision_recall_metrics,
    get_auc_from_thresholds,
)
import math
from nltk.corpus import stopwords
import string




def get_zlib_compression_size(input_string):
    # Encode the string to bytes
    input_bytes = input_string.encode('utf-8')
    # Compress the bytes
    compressed_data = zlib.compress(input_bytes)
    # Calculate the length of the compressed data
    compressed_size = len(compressed_data)
    return compressed_size



def generate_data(
    config: ExperimentConfig,
    dataset: str,
    train: bool = True,
    presampled: str = None,
    specific_source: str = None,
    mask_model_tokenizer = None,
):
    data_obj = data_utils.Data(dataset, config=config, presampled=presampled)
    data = data_obj.load(
        train=train,
        mask_tokenizer=mask_model_tokenizer,
        specific_source=specific_source,
    )
    return data
    # return generate_samples(data[:n_samples], batch_size=batch_size)



def get_mia_scores(
    data,
    target_model: LanguageModel,
    config: ExperimentConfig,
    n_samples: int = 100,
    batch_size: int = 100,
    incontext_config=None,
    ref_config=None,
    nn_config=None,
):


    fix_seed(config.random_seed)
    n_samples = len(data["records"]) if n_samples is None else n_samples   
    results = []
    if ref_config is not None:
        ref_model = ReferenceModel(config, ref_config['model'])
        
    for batch in tqdm(range(math.ceil(n_samples / batch_size)), desc=f"Computing criterion"):
        texts = data["records"][batch * batch_size : (batch + 1) * batch_size]

        # For each entry in batch
        for idx in range(len(texts)):
            sample_information = defaultdict(list)
            sample = (
                texts[idx][: config.max_substrs]
                if config.full_doc
                else [texts[idx]]
            )

            # This will be a list of integers if pretokenized
            sample_information["sample"] = sample
            # For each substring
            for i, substr in enumerate(sample):                    
                s_tk_probs, s_all_probs, labels  = target_model.get_probabilities_with_tokens(substr, return_all_probs=True)
                loss =  target_model.get_ll(substr, probs=s_tk_probs)
                median_loss = target_model.get_ll_agg(substr, probs=s_tk_probs, agg="median")
                
                zlib_size = get_zlib_compression_size(substr)
                sample_information["Zlib"].append(loss/zlib_size)
                sample_information["Loss"].append(loss)
                

                window = config.min_k_window
                strid = config.min_k_strid
                k = config.min_k_k
                ngram_probs = []
                for i in range(0, len(s_tk_probs) - window + 1, strid):
                    ngram_prob = s_tk_probs[i : i + window]
                    ngram_probs.append(np.mean(ngram_prob))
                min_k_probs = sorted(ngram_probs)[: int(len(ngram_probs) * k)]
                sample_information["MIN-K"].append(-np.mean(min_k_probs))


                input_ids = torch.tensor(target_model.tokenizer.encode(substr)).unsqueeze(0)
                input_ids = input_ids[0][1:].unsqueeze(-1)
                probs = F.softmax(s_all_probs, dim=-1)
                log_probs = F.log_softmax(s_all_probs, dim=-1) #F.log_softmax(s_all_probs[0, :-1], dim=-1)
                mu = (probs * log_probs).sum(-1)
                sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)
                token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)
                

                for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                    k_length = int(len(token_log_probs) * ratio)
                    topk =  sorted(token_log_probs)[: k_length]
                    sample_information[f'mink_{ratio}'].append(-np.mean(topk).item())

                # import pdb; pdb.set_trace()
                mink_plus = (token_log_probs - mu) / sigma.sqrt()
                for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                    k_length = int(len(mink_plus) * ratio)
                    topk = sorted(mink_plus)[: k_length]
                    sample_information[f'mink++_{ratio}'].append(-np.mean(topk).item())

            results.append(sample_information)

    samples = []
    predictions = defaultdict(lambda: [])
    for r in results:
        samples.append(r["sample"])
        for attack, scores in r.items():
            if attack != "sample" and attack != "detokenized":
                # TODO: Is there a reason for the np.min here?
                predictions[attack].append(np.min(scores))

    return predictions, samples


def compute_metrics_from_scores(
        config,
        preds_member: dict,
        preds_nonmember: dict,
        samples_member: List,
        samples_nonmember: List,
        n_samples: int):

    attack_keys = list(preds_member.keys())
    if attack_keys != list(preds_nonmember.keys()):
        raise ValueError("Mismatched attack keys for member/nonmember predictions")

    # Collect outputs for each attack
    blackbox_attack_outputs = {}
    for attack in attack_keys:
        preds_member_ = preds_member[attack]
        preds_nonmember_ = preds_nonmember[attack]

        fpr, tpr, roc_auc, roc_auc_res, thresholds = get_roc_metrics(
            preds_member=preds_member_,
            preds_nonmember=preds_nonmember_,
            perform_bootstrap=True,
            return_thresholds=True,
        )
        tpr_at_low_fpr = {
            upper_bound: tpr[np.where(np.array(fpr) < upper_bound)[0][-1]]
            for upper_bound in config.fpr_list
        }
        p, r, pr_auc = get_precision_recall_metrics(
            preds_member=preds_member_,
            preds_nonmember=preds_nonmember_
        )

        print(
            f"{attack}_threshold ROC AUC: {roc_auc}, PR AUC: {pr_auc}, tpr_at_low_fpr: {tpr_at_low_fpr}"
        )
        blackbox_attack_outputs[attack] = {
            "name": f"{attack}_threshold",
            "predictions": {
                "member": preds_member_,
                "nonmember": preds_nonmember_,
            },
            "info": {
                "n_samples": n_samples,
            },
            # "raw_results": (
            #     {"member": samples_member, "nonmember": samples_nonmember}
            #     if not config.pretokenized
            #     else []
            # ),
            "metrics": {
                "roc_auc": roc_auc,
                "fpr": fpr,
                "tpr": tpr,
                "bootstrap_roc_auc_mean": np.mean(roc_auc_res.bootstrap_distribution),
                "bootstrap_roc_auc_std": roc_auc_res.standard_error,
                "tpr_at_low_fpr": tpr_at_low_fpr,
                "thresholds": thresholds,
            },
            "pr_metrics": {
                "pr_auc": pr_auc,
                "precision": p,
                "recall": r,
            },
            "loss": 1 - pr_auc,
        }

    return blackbox_attack_outputs



def generate_data_processed(
    config,
    raw_data_member,
    batch_size: int,
    raw_data_non_member: List[str] = None
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
        non_member_text = raw_data_non_member[batch * batch_size : (batch + 1) * batch_size]

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
    incontext_config =  OmegaConf.to_container(config.incontext_config, resolve=True)
    ref_config =  None #OmegaConf.to_container(config.ref_config, resolve=True)
    nn_config = OmegaConf.to_container(config.neighborhood_config, resolve=True)
    fix_seed(config.random_seed)

    
    START_DATE = datetime.datetime.now().strftime("%Y-%m-%d")
    START_TIME = datetime.datetime.now().strftime("%H-%M-%S-%f")

    base_model_name = config.base_model.replace("/", "_")

    ### The following is about creating the output folder ###
    # define SAVE_FOLDER as the timestamp - base model name - mask filling model name
    # create it if it doesn't exist
    output_subfolder = f"{config.output_name}/"
    exp_name = config.experiment_name+"_"+str(incontext_config["num_shots"])

    # Add pile source to suffix, if provided
    # TODO: Shift dataset-specific processing to their corresponding classes
    # Results go under target model
    sf = os.path.join(exp_name, config.base_model.replace("/", "_"))
    SAVE_FOLDER = os.path.join(env_config.tmp_results, sf)
    new_folder = os.path.join(env_config.results, sf)
    print(f"{new_folder}")
    if os.path.isdir((new_folder)):
        print(f"HERE folder exists, not running this exp {new_folder}")
        exit(0)
    
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

    with open(os.path.join(SAVE_FOLDER, "raw_data.json"), "w") as f:
            print(f"Writing raw data to {os.path.join(SAVE_FOLDER, 'raw_data.json')}")
            json.dump(data, f)

    with open(os.path.join(SAVE_FOLDER, "raw_data_lens.json"), "w") as f:
        print(
            f"Writing raw data to {os.path.join(SAVE_FOLDER, 'raw_data_lens.json')}"
        )
        json.dump(seq_lens, f)


    data_members = {
        "records": data["member"],
    }
    data_nonmembers = {
        "records": data["nonmember"],
    }

    outputs = []
    if config.blackbox_attacks is None:
        raise ValueError("No blackbox attacks specified in config!")

    member_preds, member_samples = get_mia_scores(
        data_members,
        target_model=base_model,
        config=config,
        n_samples=n_samples,
        incontext_config=incontext_config,
        batch_size=config.batch_size,
        ref_config=ref_config,
        nn_config=nn_config,

    )
    # Collect scores for non-members
    nonmember_preds, nonmember_samples = get_mia_scores(
        data_nonmembers,
        target_model=base_model,
        config=config,
        n_samples=n_samples,
        incontext_config=incontext_config,
        batch_size=config.batch_size,
        ref_config=ref_config,
        nn_config=nn_config,
    )
    blackbox_outputs = compute_metrics_from_scores(
        config,
        member_preds,
        nonmember_preds,
        member_samples,
        nonmember_samples,
        n_samples=n_samples,
    )


    # for attack, output in blackbox_outputs.items():
    #     outputs.append(output)

    for attack, output in blackbox_outputs.items():
        outputs.append(output)


    with open(os.path.join(SAVE_FOLDER, "all_results.pkl"), "wb") as f:
        pickle.dump(blackbox_outputs, f)

            
    plot_utils.save_roc_curves(
        outputs,
        save_folder=SAVE_FOLDER,
        model_name=base_model_name,
        neighbor_model_name=None,
    )
    plot_utils.save_ll_histograms(outputs, save_folder=SAVE_FOLDER)
    plot_utils.save_llr_histograms(outputs, save_folder=SAVE_FOLDER)

    # move results folder from env_config.tmp_results to results/, making sure necessary directories exist
    if not os.path.exists(os.path.dirname(new_folder)):
        os.makedirs(os.path.dirname(new_folder))
    os.rename(SAVE_FOLDER, new_folder)

    
    # Continue with the rest of your logic using the configuration

if __name__ == "__main__":
    main()
