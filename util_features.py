import numpy as np
from sklearn.feature_selection import (
    RFE,
    RFECV,
    SelectFromModel,
    SelectKBest,
    f_classif,
    mutual_info_classif,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score
import pickle
import copy
import pandas as pd
from sklearn.decomposition import PCA


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()
        self.values = []


    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.std = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count
        # self.std = np.std(self.values)
        self.values.append(val)
    


def lempel_ziv_complexity(x, bins):
    """
    Calculate a complexity estimate based on the Lempel-Ziv compression
    algorithm.

    The complexity is defined as the number of dictionary entries (or sub-words) needed
    to encode the time series when viewed from left to right.
    For this, the time series is first binned into the given number of bins.
    Then it is converted into sub-words with different prefixes.
    The number of sub-words needed for this divided by the length of the time
    series is the complexity estimate.

    For example, if the time series (after binning in only 2 bins) would look like "100111",
    the different sub-words would be 1, 0, 01 and 11 and therefore the result is 4/6 = 0.66.

    Ref: https://github.com/Naereen/Lempel-Ziv_Complexity/blob/master/src/lempel_ziv_complexity.py

    """
    x = np.asarray(x)
    bins_edges = np.linspace(min(x), 0, 100)
    # Use numpy.digitize to bin the values
    x = np.digitize(x, bins=bins_edges, right=True)

    bins = np.linspace(np.min(x), np.max(x), bins + 1)[1:]
    sequence = np.searchsorted(bins, x, side="left")

    sub_strings = set()
    n = len(sequence)

    ind = 0
    inc = 1
    while ind + inc <= n:
        # convert to tuple in order to make it hashable
        sub_str = tuple(sequence[ind : ind + inc])
        if sub_str in sub_strings:
            inc += 1
        else:
            sub_strings.add(sub_str)
            ind += inc
            inc = 1
    return len(sub_strings) / n


def approximate_entropy(x, m, r):
    """
    Implements a vectorized Approximate entropy algorithm.

        https://en.wikipedia.org/wiki/Approximate_entropy

    For short time-series this method is highly dependent on the parameters,
    but should be stable for N > 2000, see:

        Yentes et al. (2012) -
        *The Appropriate Use of Approximate Entropy and Sample Entropy with Short Data Sets*


    Other shortcomings and alternatives discussed in:

        Richman & Moorman (2000) -
        *Physiological time-series analysis using approximate entropy and sample entropy*

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param m: Length of compared run of data
    :type m: int
    :param r: Filtering level, must be positive
    :type r: float

    :return: Approximate entropy
    :return type: float
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)

    N = x.size
    r *= np.std(x)
    if r < 0:
        raise ValueError("Parameter r must be positive.")
    if N <= m + 1:
        return 0

    def _phi(m):
        x_re = np.array([x[i : i + m] for i in range(N - m + 1)])
        C = np.sum(
            np.max(np.abs(x_re[:, np.newaxis] - x_re[np.newaxis, :]), axis=2) <= r,
            axis=0,
        ) / (N - m + 1)
        return np.sum(np.log(C)) / (N - m + 1.0)

    return np.abs(_phi(m) - _phi(m + 1))


def count_above_mean(x):
    """
    Returns the number of values in x that are higher than the mean of x

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    m = np.mean(x)
    return np.where(x > m)[0].size


def find_t(probabilities, tau, alpha):
    T = len(probabilities)
    all_alpha = np.mean(np.array(probabilities) > tau)
    # cumsum in the reverse order
    t = np.where(
        np.cumsum(np.array(probabilities[::-1]) > tau) / T > alpha * all_alpha
    )[0][0]

    # t = np.where(np.cumsum(np.array(probabilities)>tau)/T > alpha*all_alpha)[0][0]
    # if len(t) == 0:
    #     return -1

    return t


def get_roc(signal, membership):
    # Calculate false positive rates, true positive rates, and thresholds
    fpr_list, tpr_list, thresholds = roc_curve(membership, signal)

    # Calculate AUC using roc_auc_score for comparison
    auc = roc_auc_score(membership, signal)

    # Find indices for FPR thresholds 0.001, 0.01, 0.05
    idx01 = np.searchsorted(fpr_list, 0.001)
    idx1 = np.searchsorted(fpr_list, 0.01)
    idx5 = np.searchsorted(fpr_list, 0.05)

    # Ensure indices are within bounds
    tpr_001 = tpr_list[idx01 - 1] if idx01 > 0 else 0.0
    tpr_01 = tpr_list[idx1 - 1] if idx1 > 0 else 0.0
    tpr_05 = tpr_list[idx5 - 1] if idx5 > 0 else 0.0

    # Initialize variables to track the best accuracy and threshold
    best_accuracy = 0
    best_threshold = 0

    # Iterate over each threshold to find the one that maximizes accuracy
    for threshold in thresholds:
        # Generate predictions based on the current threshold
        predictions = (signal >= threshold).astype(int)

        # Calculate the accuracy for the current threshold
        accuracy = accuracy_score(membership, predictions)

        # Update the best accuracy and threshold if the current one is better
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return (
        np.array(tpr_list),
        np.array(fpr_list),
        auc,
        tpr_001,
        tpr_01,
        tpr_05,
        best_accuracy,
        best_threshold,
    )


def get_loss(x, start_time=0, end_time=-1):
    return np.array([np.mean(x[i][start_time:end_time]) for i in range(len(x))])


def get_ppl(x, start_time=0, end_time=-1):
    return np.array(
        [np.exp(-(np.mean(x[i][start_time:end_time]))) for i in range(len(x))]
    )


def get_count_above(x, threshold, start_time=0, end_time=-1):
    return np.array(
        [
            np.mean(np.array(x[i][start_time:end_time]) >= threshold)
            for i in range(len(x))
        ]
    )


def get_count_mean(x, start_time=0, end_time=-1):
    return np.array(
        [count_above_mean(x[i][start_time:end_time]) for i in range(len(x))]
    )


def get_lz_complexity(x, bins,start_time=0, end_time=200):
    return np.array([lempel_ziv_complexity(x[i][start_time:end_time], bins) for i in range(len(x))])


def get_approximate_entropy(x, m, r, cut_off=200):
    return np.array([approximate_entropy(x[i][:cut_off], m, r) for i in range(len(x))])


def get_find_t(x, tau, alpha):
    return np.array([find_t(x[i], tau, alpha) for i in range(len(x))])


def get_token_diversity(labels, start_time=0, end_time=-1):
    return np.array(
        [
            len(set(labels[i][start_time:end_time]))
            / len(labels[i][start_time:end_time])
            for i in range(len(labels))
        ]
    )


def get_slope(x, start_time=0, end_time=-1):
    return np.array(
        [
            np.polyfit(
                range(len(x[i][start_time:end_time])), x[i][start_time:end_time], 1
            )[0]
            for i in range(len(x))
        ]
    )


def load_data_from_model_history(path):
    with open(
        path,
        "rb",
    ) as f:
        all_features = pickle.load(f)
    repeated_str = (
        "labels_repeated_10"
        if "labels_repeated_10" in all_features["member_preds"][0]
        else "labels_repeated_5"
    )
    num_repeat = 10 if "labels_repeated_10" in all_features["member_preds"][0] else 5
    if (
        "labels_repeated_10" not in all_features["member_preds"][0]
        and "labels_repeated_5" not in all_features["member_preds"][0]
    ):
        raise ValueError("The data does not have the data with repeated patterns.")

    labels_mem = [
        all_features["member_preds"][i]["labels"][0][0].numpy()
        for i in range(len(all_features["member_preds"]))
    ]
    labels_nonmem = [
        all_features["nonmember_preds"][i]["labels"][0][0].numpy()
        for i in range(len(all_features["nonmember_preds"]))
    ]
    x_nonmember = [
        all_features["nonmember_preds"][i]["tk_probs"][0]
        for i in range(len(all_features["nonmember_preds"]))
    ]
    x_member = [
        all_features["member_preds"][i]["tk_probs"][0]
        for i in range(len(all_features["member_preds"]))
    ]
    x_all = x_nonmember + x_member
    label_all = labels_nonmem + labels_mem
    y_all = np.concatenate([np.zeros(len(x_nonmember)), np.ones(len(x_member))], axis=0)
    data_dict = {"x_all": x_all, "label_all": label_all, "y_all": y_all}

    x_copied_members = []
    for idx in range(len(all_features["member_preds"])):
        target_labels = np.array(all_features["member_preds"][idx]["labels"])[0][0][2:]
        length = len(target_labels)
        target_signal = all_features["member_preds"][idx][repeated_str][0][0]
        all_si = find_sublist_indices(target_signal, target_labels)
        if len(all_si) <= 2:
            all_si = find_sublist_indices(target_signal, target_labels[1:])
            if len(all_si) <= 2:
                all_si = find_sublist_indices(target_signal, target_labels[2:])
                if len(all_si) <= 2:
                    all_si = find_sublist_indices(target_signal, target_labels[3:])
                    if len(all_si) <= 2:
                        all_si = find_sublist_indices(target_signal, target_labels[4:])
                        if len(all_si) <= 2:
                            import pdb; pdb.set_trace()
            
            
            
        # if len(all_si) <= 1:
        #     target_labels = np.array(all_features["member_preds"][idx]["labels"])[0][0][3:]
        #     length = len(target_labels)
        #     target_signal = all_features["member_preds"][idx]["labels_repeated_10"][0][0]
        x_copied_members.append(
            np.array(
                [
                    all_features["member_preds"][idx][
                        f"tk_probs_repeated_{num_repeat}"
                    ][0][si - 1 : si + length - 1]
                    for si in all_si[:-2]
                ]
            )
        )

    x_copied_nonmembers = []
    for idx in range(len(all_features["nonmember_preds"])):
        target_labels = np.array(all_features["nonmember_preds"][idx]["labels"])[0][0][
            2:
        ]
        length = len(target_labels)
        target_signal = all_features["nonmember_preds"][idx][repeated_str][0][0]
        all_si = find_sublist_indices(target_signal, target_labels)
        # if len(all_si) <= 1:
        #     target_labels = np.array(all_features["nonmember_preds"][idx]["labels"])[0][0][3:]
        #     length = len(target_labels)
        #     target_signal = all_features["nonmember_preds"][idx]["labels_repeated_10"][0][0]
        #     all_si = find_sublist_indices(target_signal, target_labels)
        x_copied_nonmembers.append(
            np.array(
                [
                    all_features["nonmember_preds"][idx][
                        f"tk_probs_repeated_{num_repeat}"
                    ][0][si - 1 : si + length - 1]
                    for si in all_si[:-2]
                ]
            )
        )
    x_copied_all = x_copied_nonmembers + x_copied_members
    data_dict["x_copied_all"] = x_copied_all

    semantic = {}
    for keys in all_features["member_preds"][0].keys():
        if "conf" in keys:
            semantic[keys] = []
            for idx in range(len(all_features["nonmember_preds"])):
                semantic[keys].append(all_features["nonmember_preds"][idx][keys])
            for idx in range(len(all_features["member_preds"])):
                semantic[keys].append(all_features["member_preds"][idx][keys])

    data_dict["semantic"] = semantic

    all_preds_copies = [
        all_features["nonmember_preds"][idx][f"tk_probs_repeated_{num_repeat}"][0]
        for idx in range(len(all_features["nonmember_preds"]))
    ] + [
        all_features["member_preds"][idx][f"tk_probs_repeated_{num_repeat}"][0]
        for idx in range(len(all_features["member_preds"]))
    ]

    data_dict["all_preds_copies"] = all_preds_copies

    return data_dict


def collect_all_features(x, labels):
    loss_features = np.array(
                [
                    get_loss(x, start_time=0, end_time=end_time)
                    for end_time in [-1, 200, 300]
                ]
            )

    ppl_features = np.array(
                [
                    get_ppl(x, start_time=0, end_time=end_time)
                    for end_time in [-1, 200, 300]
                ]
            )
    count_above_features = np.array(
        [
            get_count_above(x, threshold, start_time=0, end_time=200)
            for threshold in [-1, -2, -3]
        ]
    )
    lz_complexity_features = np.array(
        [get_lz_complexity(x, bins) for bins in [3, 4, 5]]
    )

    find_t_features = np.array(
        [
            get_find_t(x, tau, beta)
            for tau in [-1]
            for beta in [0.4, 0.5, 0.6]
        ]
    )
    token_diversity_features = np.array(
        [
            get_token_diversity(labels, start_time=0, end_time=end_time)
            for end_time in [-1, 200, 300,]
        ]
    )
    count_mean_features = np.array(
                [
                    get_count_mean(x, start_time=0, end_time=end_time)
                    for end_time in [-1, 200, 300]
                ]
            )
   
    calibrated_loss = loss_features / token_diversity_features
    calibrated_ppl = ppl_features / token_diversity_features
    
    return {
        "loss": loss_features,
        "ppl": ppl_features,
        "count_above": count_above_features,
        "lz_complexity": lz_complexity_features,
        # "approximate_entropy": approximate_entropy_features,
        "find_t": find_t_features,
        "token_diversity": token_diversity_features,
        # "slope": slope_features,
        "count_mean": count_mean_features,
        "calibrated_loss": calibrated_loss,
        "calibrated_ppl": calibrated_ppl,
        # "calibrate_above_mean": calibrate_above_mean,
        # "calibrated_slope": calibrated_slope,
    }


def find_sublist_indices(a, b):
    
    a = np.array(a)
    b = np.array(b)
    len_a = len(a)
    len_b = len(b)
    indices = []
    while len(indices) <= 1:
        
      # List to store the starting indices
        # Loop through each possible starting index in list a
        for i in range(len_a - len_b + 1):
            # Check if the sublist of a starting at index i matches list b
            if np.mean(a[i : i + len_b] == b) == 1:
                indices.append(i)
        b = b[1:]
        len_b = len(b)

    return indices  # Return the list of starting indices


class GroupPCA:
    def __init__(self, n_components, feature_group):
        self.n_components = n_components
        self.pca_list = {
            group: PCA(n_components=n_components) for group in np.unique(feature_group)
        }
        self.feature_group = feature_group

    def fit(self, X):
        for group in np.unique(self.feature_group):
            self.pca_list[group].fit(X[:, self.feature_group == group])

    def transform(self, X):
        return np.concatenate(
            [
                self.pca_list[group].transform(X[:, self.feature_group == group])
                for group in np.unique(self.feature_group)
            ],
            axis=1,
        )

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def select_features_based_on_fs(
    x_attack_train, y_attack_train, num_features=3, method="f_classif"
):
    if method == "f_classif":
        selector = SelectKBest(f_classif, k=num_features)
        selector.fit(x_attack_train, y_attack_train)

    elif method == "mutual_info_classif":
        selector = SelectKBest(mutual_info_classif, k=num_features)
        selector.fit(x_attack_train, y_attack_train)

    elif method == "rfe":
        estimator = LogisticRegression()
        selector = RFE(estimator, n_features_to_select=num_features)
        selector.fit(x_attack_train, y_attack_train)
    elif method == "lasso":
        lasso = LogisticRegression()
        lasso.fit(x_attack_train, y_attack_train)
        selector = SelectFromModel(lasso, prefit=True, max_features=num_features)
    elif method == "rfecv":
        estimator = LogisticRegression()
        selector = RFECV(estimator, step=1, cv=5, min_features_to_select=num_features)
        selector.fit(x_attack_train, y_attack_train)
    elif method == "all":
        # return True for all features
        return np.ones(x_attack_train.shape[1], dtype=bool)
    return selector.get_support()
