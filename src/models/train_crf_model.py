#!/usr/bin/env python3
import os
import sys
import pandas as pd
from joblib import dump
import sklearn_crfsuite
from sklearn_crfsuite import metrics

import os
cwd = os.getcwd()

if __name__ == '__main__':
    print("TRAINING CRF MODEL")
    X_train = pd.read_hdf(os.path.join(os.path.dirname(__file__), "../../data/interim/crf_training_features.h5"), 'df')
    y_train = pd.read_hdf(os.path.join(os.path.dirname(__file__), "../../data/interim/crf_training_labels.h5"), 'df')

    X_test = pd.read_hdf(os.path.join(os.path.dirname(__file__), "../../data/interim/crf_test_features.h5"), 'df')
    y_test = pd.read_hdf(os.path.join(os.path.dirname(__file__), "../../data/interim/crf_test_labels.h5"), 'df')

    print(X_train[0])

    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True,
    )
    crf.fit(X_train, y_train)

    labels = list(crf.classes_)

    y_pred = crf.predict(X_test)
    print(metrics.flat_f1_score(y_test, y_pred, average="weighted", labels=labels))

    # group B and I results
    sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
    print(
        metrics.flat_classification_report(
            y_test, y_pred, labels=sorted_labels, digits=3
        )
    )
    dump(crf, os.path.join(os.path.dirname(__file__), '../../models/crf_model.joblib')) 