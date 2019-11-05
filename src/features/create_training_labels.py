#!/usr/bin/env python3
import os
import sys
from decimal import Decimal, InvalidOperation
import spacy
import pandas as pd


def match_up(df):
    """
    Returns our best guess of the match between the tags and the
    words from the display text.
    This problem is difficult for the following reasons:
        * not all the words in the display name have associated tags
        * the quantity field is stored as a number, but it appears
          as a string in the display name
        * the comment is often a compilation of different comments in
          the display name
    """
    labels = []

    for token in df["input"]:
        decimal_token = None
        try:
            decimal_token = Decimal(token)
        except InvalidOperation:
            pass
        if token in df["name"]:
            labels.append("NAME")
        elif token in df["unit"]:
            labels.append("UNIT")
        elif decimal_token is not None and decimal_token == df["qty"]:
            labels.append("QTY")
        elif token in df["comment"]:
            labels.append("COMMENT")
        elif decimal_token is not None and decimal_token == df["range_end"]:
            labels.append("RANGE_END")
        else:
            labels.append("OTHER")
    return labels


def add_prefixes(data):
    """
    We use BIO tagging/chunking to differentiate between tags
    at the start of a tag sequence and those in the middle. This
    is a common technique in entity recognition.

    Reference: http://www.kdd.cis.ksu.edu/Courses/Spring-2013/CIS798/Handouts/04-ramshaw95text.pdf
    """
    prev_tag = None
    new_data = []

    for token, tag in data:

        new_tag = ""

        p = "B" if ((prev_tag is None) or (tag != prev_tag)) else "I"
        new_tag = "%s-%s" % (p, tag)
        new_data.append(new_tag)
        prev_tag = tag

    return new_data


if __name__ == "__main__":

    nlp = spacy.load("en_core_web_lg", disable=["tagger", "parser", "ner", "textcat"])

    # Load raw data and do some preprocessing
    training_data = pd.read_hdf(os.path.join(os.path.dirname(__file__), "../../data/interim/crf_training_data.h5"))
    test_data = pd.read_hdf(os.path.join(os.path.dirname(__file__), "../../data/interim/crf_test_data.h5"))

    # have spacy parse the input string with the full pipeline to generate features this will take some time
    training_data["input"] = list(
        nlp.pipe(training_data["input"].astype("unicode").values, batch_size=50)
    )
    test_data["input"] = list(
        nlp.pipe(test_data["input"].astype("unicode").values, batch_size=50)
    )

    # for assigining labels we only need lemmas
    training_data["name"] = list(
        nlp.pipe(training_data["name"].astype("unicode").values, batch_size=50)
    )
    training_data["unit"] = list(
        nlp.pipe(training_data["unit"].astype("unicode").values, batch_size=50)
    )
    training_data["comment"] = list(
        nlp.pipe(training_data["comment"].astype("unicode").values, batch_size=50)
    )

    test_data["name"] = list(
        nlp.pipe(test_data["name"].astype("unicode").values, batch_size=50)
    )
    test_data["unit"] = list(
        nlp.pipe(test_data["unit"].astype("unicode").values, batch_size=50)
    )
    test_data["comment"] = list(
        nlp.pipe(test_data["comment"].astype("unicode").values, batch_size=50)
    )

    training_data["input"] = training_data["input"].apply(
        lambda doc: [token.lemma_ for token in doc]
    )
    training_data["name"] = training_data["name"].apply(
        lambda doc: [token.lemma_ for token in doc]
    )
    training_data["unit"] = training_data["unit"].apply(
        lambda doc: [token.lemma_ for token in doc]
    )
    training_data["comment"] = training_data["comment"].apply(
        lambda doc: [token.lemma_ for token in doc]
    )

    test_data["input"] = test_data["input"].apply(
        lambda doc: [token.lemma_ for token in doc]
    )
    test_data["name"] = test_data["name"].apply(
        lambda doc: [token.lemma_ for token in doc]
    )
    test_data["unit"] = test_data["unit"].apply(
        lambda doc: [token.lemma_ for token in doc]
    )
    test_data["comment"] = test_data["comment"].apply(
        lambda doc: [token.lemma_ for token in doc]
    )

    training_data["labels"] = training_data.apply(match_up, axis=1)
    test_data["labels"] = test_data.apply(match_up, axis=1)

    crf_training_labels = pd.Series(
        training_data.apply(
            lambda row: add_prefixes(zip(row["input"], row["labels"])), axis=1
        )
    )
    crf_test_labels = pd.Series(
        test_data.apply(
            lambda row: add_prefixes(zip(row["input"], row["labels"])), axis=1
        )
    )

    crf_training_labels.to_hdf(os.path.join(os.path.dirname(__file__), "../../data/interim/crf_training_labels.h5"), mode='w', format="fixed")
    crf_test_labels.to_hdf(os.path.join(os.path.dirname(__file__), "../../data/interim/crf_test_labels.h5"), mode='w', format="fixed")
