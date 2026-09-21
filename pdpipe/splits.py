"""Version-independent stratified group splitting.

``sklearn.model_selection.StratifiedGroupKFold`` is not guaranteed to produce
identical partitions across library versions, and on a cohort of 32 subjects a
single subject moving between folds visibly shifts the reported AUC. Because
the review requires that every figure be reproducible from the released code,
fold assignment is performed here instead, using only ``numpy.random.Generator``
semantics, which are stable across versions.

The rule is deliberately simple and fully specified:

1. Reduce the data to one row per subject, carrying that subject's label.
2. Within each class stratum, sort subject identifiers lexicographically so
   that the starting order never depends on row order in the source file.
3. Permute each stratum with a seeded ``numpy.random.Generator``.
4. Deal subjects to folds round-robin within each stratum, so every fold
   receives as even a share of each class as the counts allow.

With 8 control subjects and 5 folds this guarantees every fold receives at
least one control, which sklearn's implementation does not.
"""
from __future__ import annotations

import numpy as np


class DeterministicStratifiedGroupKFold:
    """Stratified group k-fold whose partitions depend only on the seed."""

    def __init__(self, n_splits: int = 5, *, shuffle: bool = True,
                 random_state: int = 0):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def _subject_folds(self, y: np.ndarray, groups: np.ndarray) -> dict:
        groups = np.asarray(groups)
        y = np.asarray(y)

        # one label per subject; reject heterogeneous subjects loudly
        subjects, first_idx = np.unique(groups, return_index=True)
        labels = y[first_idx]
        for s, lab in zip(subjects, labels):
            if not np.all(y[groups == s] == lab):
                raise ValueError(f"Subject {s} has heterogeneous labels.")

        rng = np.random.default_rng(self.random_state)
        assignment: dict = {}
        for cls in np.unique(labels):
            members = np.sort(subjects[labels == cls])     # stable start order
            if self.shuffle:
                members = members[rng.permutation(members.size)]
            for position, subject in enumerate(members):
                assignment[subject] = position % self.n_splits
        return assignment

    def split(self, X=None, y=None, groups=None):
        if y is None or groups is None:
            raise ValueError("y and groups are required.")
        assignment = self._subject_folds(y, groups)
        fold_of_row = np.array([assignment[g] for g in np.asarray(groups)])
        indices = np.arange(fold_of_row.size)
        for k in range(self.n_splits):
            test = indices[fold_of_row == k]
            train = indices[fold_of_row != k]
            if test.size == 0:
                raise ValueError(
                    f"Fold {k} is empty; n_splits exceeds the number of "
                    "subjects in some class.")
            yield train, test
