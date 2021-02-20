import pandas
import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from collections import Counter, defaultdict
from sklearn.utils import check_random_state

class RepeatedStratifiedGroupKFold():

    def __init__(self, n_splits=5, n_repeats=1, random_state=None):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        k = self.n_splits

        def eval_y_counts_per_fold(y_counts, fold):
            y_counts_per_fold[fold] += y_counts
            std_per_label = []
            for label in range(labels_num):
                label_std = np.std(
                    [y_counts_per_fold[i][label] / y_distr[label] for i in range(k)]
                )
                std_per_label.append(label_std)
            y_counts_per_fold[fold] -= y_counts
            return np.mean(std_per_label)

        rnd = check_random_state(self.random_state)
        for repeat in range(self.n_repeats):
            labels_num = np.max(y) + 1
            y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
            y_distr = Counter()
            for label, g in zip(y, groups):
                y_counts_per_group[g][label] += 1
                y_distr[label] += 1

            y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
            groups_per_fold = defaultdict(set)

            groups_and_y_counts = list(y_counts_per_group.items())
            rnd.shuffle(groups_and_y_counts)

            for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
                best_fold = None
                min_eval = None
                for i in range(k):
                    fold_eval = eval_y_counts_per_fold(y_counts, i)
                    if min_eval is None or fold_eval < min_eval:
                        min_eval = fold_eval
                        best_fold = i
                y_counts_per_fold[best_fold] += y_counts
                groups_per_fold[best_fold].add(g)

            all_groups = set(groups)
            for i in range(k):
                train_groups = all_groups - groups_per_fold[i]
                test_groups = groups_per_fold[i]

                train_indices = [i for i, g in enumerate(groups) if g in train_groups]
                test_indices = [i for i, g in enumerate(groups) if g in test_groups]

                yield train_indices, test_indices




nfold=5
train = pd.read_csv('../train.csv')
train.head()



# let's first concat all the labels
# e.g 00000000010
target_cols = train.drop(['StudyInstanceUID', 'PatientID'],axis=1).columns.values.tolist()
targets = train[target_cols].astype(str)
# create a new col to store the label
train['combined_tar'] = ''
for i in tqdm(range(targets.shape[1])):
    train['combined_tar'] += targets.iloc[:,i]
# take a look at it
train.combined_tar.value_counts()

train['combined_tar'] = LabelEncoder().fit_transform(train['combined_tar'])




train['fold'] = -1
rskf = RepeatedStratifiedGroupKFold(n_splits=nfold, random_state=42)
for i, (train_idx, valid_idx) in enumerate(rskf.split(train, train.combined_tar, train.PatientID)): #(df, targets, group)
    train.loc[valid_idx, 'fold'] = int(i)

train.drop('combined_tar', axis=1).to_csv('train_folds.csv', index=False)