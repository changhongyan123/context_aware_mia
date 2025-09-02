import pickle
import numpy as np
from util_features import load_data_from_model_history, GroupPCA, approximate_entropy, get_slope
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.preprocessing import MinMaxScaler
from util_features import AverageMeter, collect_all_features, load_data_from_model_history
import os
import time
import argparse
import scipy
from scipy.stats import percentileofscore

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


    grouppca = GroupPCA(
        n_components=n_components_per_group, feature_group=feature_group
    )
    x_train_pca = grouppca.fit_transform(x_train)
    x_test_pca = grouppca.transform(x_test)
    clf = get_model()
    clf.fit(x_train_pca, y_train)
    pred_probs = clf.predict_proba(x_test_pca)[:, 1]
    pred_probs_train = clf.predict_proba(x_train_pca)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, pred_probs)
    
    return fpr, tpr, pred_probs


if __name__ == "__main__":
    num_copies = 9
    pop_ratio = 0.3
    num_repeat = 10
    n_components = 10
    n_components_per_group = 1

    # split = "7_0.2"

    # accepte the model name and model size from the command line 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-160m-deduped")
    parser.add_argument("--split", type=str, default="7_0.2")
    parser.add_argument("--model_size", type=str, default="160m")
    
    args = parser.parse_args()
    model_name = args.model_name.replace("/", "_")
    
    split = args.split
    model_size = args.model_size

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
            "Loss",
            "Max",
            "Average",
            "Pearson",
            "LR",
            "LR(PCA)",
            "LR(Group PCA)",
        ]
    }
    for source in dataset_list:
        data[f"AUC_{source}"] = []
        data[f"TPR_{source}"] = []
    all_res_features = {}
    for dataset in dataset_list:
        # if dataset == "github_ngram_13_0.2" or dataset == "github_ngram_13_0.8":
        #     continue

        if "pythia" in model_name and "deduped" in model_name:
            if  "160m" in model_name:
                res_path = f"v3_extracted_features_{dataset}.pkl"
            else:
                res_path = f"v3_extracted_features_{dataset}_{model_size}.pkl"
        else:
            res_path = f"v3_extracted_features_{dataset}_{model_name}.pkl"

        if os.path.exists(res_path) == False:

            data_path = f"results_new/construct_dataset_with_repeat_10/{model_name}/all_features_{dataset}.pkl"
            data_dict = load_data_from_model_history(data_path)

            x_all = data_dict["x_all"]
            label_all = data_dict["label_all"]
            y_all = data_dict["y_all"]
            x_copied_all = data_dict["x_copied_all"]
            num_sample = len(y_all)
            # print(len(x_copied_all), len(x_copied_all[0]))

            start_time = time.time()
            extracted_features_single = collect_all_features(x_all, label_all)
            print(f"Time taken for single: {time.time() - start_time}")

            extracted_features_copied = []
            for i in range(min(len(x_copied_all[0]), num_copies)):
                start_time = time.time()
                for j in range(len(x_copied_all)):
                    if len(x_copied_all[j]) == 0:
                        import pdb; pdb.set_trace()

                extracted_signal = collect_all_features(
                    [x_copied_all[j][i] for j in range(len(x_copied_all))], label_all
                )
                extracted_features_copied.append(extracted_signal)
                print(
                    f"Time taken for copied {i}/{min(len(x_copied_all[0]), num_copies)}: {time.time() - start_time}"
                )

            m = 8
            r = 0.8
            start_time = time.time()
            x_approxiamte_entropy = np.array(
                [
                    np.array(
                        [
                            approximate_entropy(
                                data_dict["all_preds_copies"][idx][:cut_off], m, r
                            )
                            for idx in range(num_sample)
                        ]
                    )
                    for cut_off in [600, 800, 1000]
                ]
            )
            print(f"Time taken for approximate entropy: {time.time() - start_time}")
            start_time = time.time()
            x_slope_signal = np.array(
                [
                    get_slope(data_dict["all_preds_copies"], end_time=end_time).T
                    for end_time in [600, 800, 1000]
                ]
            ).T
            print(f"Time taken for slope signal: {time.time() - start_time}")
            with open(res_path, "wb") as f:
                pickle.dump(
                    {
                        "x_approxiamte_entropy": x_approxiamte_entropy,
                        "x_slope_signal": x_slope_signal,
                        "extracted_features_single": extracted_features_single,
                        "extracted_features_copied": extracted_features_copied,
                        "y_all": y_all,
                    },
                    f,
                )
            extracted_features_raw_single = extracted_features_single
        else:
            with open(res_path, "rb") as f:
                all_features = pickle.load(f)
            x_approxiamte_entropy = all_features["x_approxiamte_entropy"]
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

        extracted_features_single["approximate_entropy"] = x_approxiamte_entropy
        for _ in range(x_approxiamte_entropy.shape[0]):
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




    
        np.random.seed(0)
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
        # import pdb; pdb.set_trace()
        for keys in extracted_features_single.keys():
            target_test = extracted_features_single[keys].T[target_idx]
            target_y = y_all[target_idx]
            pop_signal = extracted_features_single[keys].T[pop_idx]
            
            p_group_list = []
            max_auc_per_group = []
            for j in range(pop_signal.shape[1]):
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
        max_auc_list = np.array(max_auc_list)
        max_tpr_list = np.array(max_tpr_list)
        p_values = np.array(p_values)
        combined_p = -p_values.mean(axis=0)
        fpr, tpr, _ = roc_curve(target_y, combined_p)
        
        all_res_features[f"fpr_agg_{dataset}"] = fpr
        all_res_features[f"tpr_agg_{dataset}"] = tpr
        all_res_features[f"signal_agg_{dataset}"] = combined_p


        ### Training a LR models based on the exrtacted values.
        x_attack_train_features = np.concatenate(x_attack_train_features, axis=1)
        x_attack_test_features = np.concatenate(x_attack_test_features, axis=1)
        fpr, tpr, signal = train_model_single(
            x_attack_train_features,
            x_attack_test_features,
            y_attack_train,
            y_attack_test,
            feature_group=feature_groups,
            n_components=n_components,
            n_components_per_group=n_components_per_group,
        )
        all_res_features[f"fpr_lr_{dataset}"] = fpr
        all_res_features[f"tpr_lr_{dataset}"] = tpr
        all_res_features[f"signal_lr_{dataset}"] = signal

    with open(f"results_data_paper/results_{pop_ratio}_{n_components_per_group}_{model_name}_{split}.pkl", "wb") as f:
        pickle.dump(all_res_features, f)
