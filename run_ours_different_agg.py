import pickle
import numpy as np
from util_features import AverageMeter, collect_all_features, load_data_from_model_history
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.decomposition import PCA
import shap

from util_features import find_sublist_indices
import os
from sklearn.preprocessing import MinMaxScaler
from util_features import GroupPCA

import time
import argparse
from util_features import approximate_entropy, get_slope
import scipy
from scipy.stats import percentileofscore
from scipy.stats import chi2
from scipy.stats import norm, logistic
import matplotlib.pyplot as plt
from scipy.stats import combine_pvalues
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RANSACRegressor
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.ensemble import RandomForestClassifier

MAX_DEPTH=3
    
def get_model():
    clf = LogisticRegression(max_iter=5000, solver="liblinear")
    return clf


def train_model_single(
    x_train,
    x_test,
    y_train,
    y_test,
    feature_group=None,
    n_components=10,
    n_components_per_group=2,
):
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    clf = get_model()
    clf.fit(x_train_scaled, y_train)
    pred_probs = clf.predict_proba(x_test_scaled)[:, 1]
    pred_probs_train = clf.predict_proba(x_train_scaled)[:, 1]
    all_auc_train = roc_auc_score(y_train, pred_probs_train)
    all_auc_test = roc_auc_score(y_test, pred_probs)

    fpr, tpr, _ = roc_curve(y_test, pred_probs)
    all_tpr = tpr[np.where(fpr <= 0.01)[0][-1]] * 100
    fpr_train, tpr_train, _ = roc_curve(y_train, pred_probs_train)
    all_tpr_train = tpr_train[np.where(fpr_train <= 0.01)[0][-1]] * 100
    
    

    pca = PCA(n_components=n_components)
    pca.fit(x_train)
    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)
    clf = get_model()
    clf.fit(x_train_pca, y_train)
    pred_probs = clf.predict_proba(x_test_pca)[:, 1]
    pred_probs_train = clf.predict_proba(x_train_pca)[:, 1]
    all_pca_train = roc_auc_score(y_train, pred_probs_train)
    all_pca_test = roc_auc_score(y_test, pred_probs)
    fpr_train, tpr_train, _ = roc_curve(y_train, pred_probs_train)
    all_pca_tpr_train = tpr_train[np.where(fpr_train <= 0.01)[0][-1]] * 100
    fpr, tpr, _ = roc_curve(y_test, pred_probs)
    all_pca_tpr = tpr[np.where(fpr <= 0.01)[0][-1]] * 100

    all_results = {
        "auc_test": all_auc_test,
        "pca_auc_test": all_pca_test,
        "auc_train": all_auc_train,
        "pca_auc_train": all_pca_train,
        "tpr": all_tpr,
        "pca_tpr": all_pca_tpr,
        "tpr_train": all_tpr_train,
        "pca_tpr_train": all_pca_tpr_train,
        # "group_pca_tpr": all_group_pca_tpr,
    }

    if feature_group is not None:

        grouppca = GroupPCA(
            n_components=n_components_per_group, feature_group=feature_group
        )
        x_train_pca = grouppca.fit_transform(x_train)
        x_test_pca = grouppca.transform(x_test)
        # clf = LogisticRegression(random_state=0, max_iter=5000, solver="liblinear")
        clf = get_model()
        clf.fit(x_train_pca, y_train)
        pred_probs = clf.predict_proba(x_test_pca)[:, 1]
        pred_probs_train = clf.predict_proba(x_train_pca)[:, 1]
        group_pca_train = roc_auc_score(y_train, pred_probs_train)
        group_pca_test = roc_auc_score(y_test, pred_probs)

        fpr, tpr, _ = roc_curve(y_test, pred_probs)
        all_group_pca_tpr = tpr[np.where(fpr <= 0.01)[0][-1]] * 100
        fpr_train, tpr_train, _ = roc_curve(y_train, pred_probs_train)
        all_group_pca_tpr_train = tpr_train[np.where(fpr_train <= 0.01)[0][-1]] * 100
        
        all_results["group_pca_auc_test"] = group_pca_test
        all_results["group_pca_tpr"] = all_group_pca_tpr
        all_results["group_pca_auc_train"] = group_pca_train
        all_results["group_pca_tpr_train"] = all_group_pca_tpr_train
    return all_results


if __name__ == "__main__":
    num_copies = 9
    pop_ratio = 0.3
    num_repeat = 10
    n_components = 10
    n_components_per_group = 1

    split = "7_0.2"

    model_size = "2.8b"

    model_name = f"EleutherAI_pythia-{model_size}-deduped"
    dataset_list = [
        f"arxiv_ngram_{split}",
        f"dm_mathematics_ngram_{split}",
        f"github_ngram_{split}",
        f"hackernews_ngram_{split}",
        f"pile_cc_ngram_{split}",
        f"pubmed_central_ngram_{split}",
        
    ]

    # Initialize dictionary to store the results
    data = {
        "Method": [
            "Agg",
            "Fisher",
            "Pearson",
            "George",
        ]
    }
    for source in dataset_list:
        data[f"AUC_{source}"] = []
        data[f"TPR_{source}"] = []

    for dataset in dataset_list:
        if "pythia" in model_name and "deduped" in model_name:
            if model_size == "160m":
                res_path = f"v3_extracted_features_{dataset}.pkl"
            else:
                res_path = f"v3_extracted_features_{dataset}_{model_size}.pkl"
        else:
            res_path = f"v3_extracted_features_{dataset}_{model_name}.pkl"
        
        if os.path.exists(res_path) == False:
            print(f"{res_path} does not exist")
        else:
            with open(res_path, "rb") as f:
                all_features = pickle.load(f)
            x_approximate_entropy = all_features["x_approximate_entropy"]
            x_slope_signal = all_features["x_slope_signal"]
            extracted_features_raw_single = all_features["extracted_features_single"]
            extracted_features_copied = all_features["extracted_features_copied"]
            y_all = all_features["y_all"]

        feature_idx = 0
        feature_groups = []
        extracted_features_single = {}
        filter_features = ["find_t"]

        for keys in extracted_features_copied[0]:
            if keys in filter_features:
                print("Filtering features:", keys)
                continue
            extracted_features_single[keys] = extracted_features_raw_single[keys]
            length_feature = extracted_features_copied[1][keys].shape[0]

            for idx in range(length_feature):
                feature_groups.append(feature_idx)

            if keys not in ["token_diversity"]:
                extracted_features_single[f"overall_diff_1_{keys}"] = (
                    extracted_features_copied[0][keys]
                    - extracted_features_copied[1][keys]
                )
                for idx in range(length_feature):
                    feature_groups.append(feature_idx)

                extracted_features_single[f"overall_diff_2_{keys}"] = (
                    extracted_features_copied[0][keys]
                    - extracted_features_copied[2][keys]
                )
                for idx in range(length_feature):
                    feature_groups.append(feature_idx)
                
            feature_idx += 1

        extracted_features_single["slope"] = x_slope_signal.T
        for _ in range(extracted_features_single["slope"].shape[0]):
            feature_groups.append(feature_idx)
        feature_idx += 1

        extracted_features_single["approximate_entropy"] = x_approximate_entropy
        for _ in range(x_approximate_entropy.shape[0]):
            feature_groups.append(feature_idx)
        feature_idx += 1

        ## add one more features: the fraction of tokens which are greater than the mean....
        data_path = f"results_new/construct_dataset_with_repeat_10/{model_name}/all_features_{dataset}.pkl"
        data_dict = load_data_from_model_history(data_path)
        signals = []
        for cut_off in [1000000000, 200,300]:
            x_pre_mean = [
                    [
                        np.mean(data_dict["x_all"][sample_idx][:t])
                        for t in range(1, min(cut_off, len(data_dict["x_all"][sample_idx])))
                    ]
                    
                    for sample_idx in range(len(data_dict["x_all"]))
                ]
            
            x_cur = [
                    [
                        data_dict["x_all"][sample_idx][t]
                        for t in range(1, min(cut_off, len(data_dict["x_all"][sample_idx])))
                    ]
                    for sample_idx in range(len(data_dict["x_all"]))
                ]
            signals.append(np.array([np.mean(np.array(x) > np.array(x_pre)) for x, x_pre in zip(x_cur, x_pre_mean)]))
            feature_groups.append(feature_idx)
        extracted_features_single["count_above_pre"] = np.array(signals)
        feature_idx += 1
    
        for keys in extracted_features_single.keys():
            if (
                extracted_features_single[keys].T[y_all == 1].mean()
                > extracted_features_single[keys].T[y_all == 0].mean()
            ):
                extracted_features_single[keys] = -extracted_features_single[keys]

        auc_agg = AverageMeter()
        tpr_agg = AverageMeter()
        auc_fisher = AverageMeter()
        tpr_fisher = AverageMeter()
        auc_perason = AverageMeter()
        tpr_perason = AverageMeter()
        auc_george = AverageMeter()
        tpr_george = AverageMeter()

        for i in range(num_repeat):
            np.random.seed(i)
            non_member_idx = np.where(y_all == 0)[0]
            member_idx = np.where(y_all == 1)[0]

            pop_idx = np.random.choice(
                non_member_idx, int(len(non_member_idx) * pop_ratio), replace=False
            )
            remove_idx = np.random.choice(
                member_idx, int(len(member_idx) * pop_ratio), replace=False
            )
            target_idx = np.array(
                [
                    i
                    for i in range(len(y_all))
                    if i not in pop_idx and i not in remove_idx
                ]
            )

            def compute_p_value(target_value, observed_distribution):
                return (
                    scipy.stats.percentileofscore(observed_distribution, target_value)
                    / 100
                )

            p_value_list = []
            max_auc_list = []
            max_tpr_list = []
            feature_name = []
            all_target_signal = []

            x_attack_train_features = []
            y_attack_train = np.concatenate([y_all[pop_idx], y_all[remove_idx]], axis=0)
            x_attack_test_features = []
            y_attack_test = y_all[target_idx]
            
            for keys in extracted_features_single.keys():
                target_test = extracted_features_single[keys].T[target_idx]
                target_y = y_all[target_idx]
                pop_signal = extracted_features_single[keys].T[pop_idx]
                
                p_group_list = []
                max_auc_per_group = []
                for j in range(pop_signal.shape[1]):
                    max_auc_list.append(roc_auc_score(target_y, -target_test[:, j]))
                    max_auc_per_group.append(
                        roc_auc_score(target_y, -target_test[:, j])
                    )
                    pvalue = compute_p_value(target_test[:, j], pop_signal[:, j])
                    fpr, tpr, _ = roc_curve(target_y, -target_test[:, j])
                    max_tpr_list.append(tpr[np.where(fpr < 0.01)[0][-1]] * 100)
                    feature_name.append(keys + f"_{j}")
                    p_value_list.append(pvalue)
                    
        
            
                x_train_features = np.concatenate(
                    [
                        extracted_features_single[keys].T[pop_idx],
                        extracted_features_single[keys].T[remove_idx],
                    ],
                    axis=0,
                )
                x_attack_train_features.append(
                    np.concatenate(
                        [
                            extracted_features_single[keys].T[pop_idx],
                            extracted_features_single[keys].T[remove_idx],
                        ],
                        axis=0,
                    )
                )
                x_attack_test_features.append(
                    extracted_features_single[keys].T[target_idx]
                )
            
            p_values = np.array(p_value_list)
            for agg_method in ["agg", "fisher", "pearson", "george"]:
                if agg_method == "agg":
                    combined_p = -np.mean(p_values, axis=0)
                    fpr, tpr, _ = roc_curve(target_y, combined_p)
                    auc_agg.update(roc_auc_score(target_y, combined_p))
                    tpr_agg.update(tpr[np.where(fpr <= 0.01)[0][-1]] * 100)

                elif agg_method == "fisher":
                    p_values[p_values == 0] = 1e-10
                    p_values[p_values == 1] = 1 - 1e-10
                    method = "fisher"
                    p_value_combined = combine_pvalues(p_values, method=method).pvalue
                    fpr, tpr, _ = roc_curve(target_y, -p_value_combined)
                    auc_fisher.update(roc_auc_score(target_y, -p_value_combined))
                    tpr_fisher.update(tpr[np.where(fpr <= 0.01)[0][-1]] * 100)
                elif agg_method == "pearson":
                    p_values[p_values == 0] = 1e-10
                    p_values[p_values == 1] = 1 - 1e-10
                    method = "pearson"
                    p_value_combined = combine_pvalues(p_values, method=method).pvalue
                    fpr, tpr, _ = roc_curve(target_y, -p_value_combined)
                    auc_perason.update(roc_auc_score(target_y, -p_value_combined))
                    tpr_perason.update(tpr[np.where(fpr <= 0.01)[0][-1]] * 100)
                elif agg_method == "george":
                    p_values[p_values == 0] = 1e-10
                    p_values[p_values == 1] = 1 - 1e-10
                    method = "mudholkar_george"
                    p_value_combined = combine_pvalues(p_values, method=method).pvalue
                    fpr, tpr, _ = roc_curve(target_y, -p_value_combined)
                    auc_george.update(roc_auc_score(target_y, -p_value_combined))
                    tpr_george.update(tpr[np.where(fpr <= 0.01)[0][-1]] * 100)   

            
        data[f"AUC_{dataset}"].extend(
            [
                auc_agg.avg,
                auc_fisher.avg,
                auc_perason.avg,
                auc_george.avg,
            ]
        )
        data[f"TPR_{dataset}"].extend(
            [
                tpr_agg.avg,
                tpr_fisher.avg,
                tpr_perason.avg,
                tpr_george.avg,
            ]
        )


    df = pd.DataFrame(data)
    print(df.round(2))
    df.to_csv(
        f"results_data_paper/results_{pop_ratio}_{n_components}_{n_components_per_group}_{model_name}_{split}.csv",
        index=False,
    )

