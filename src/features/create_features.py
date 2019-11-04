#!/usr/bin/env python3
import sys
import os
import argparse
import spacy
import pandas as pd


def word2features(sent, i):

    features = {
        "bias": 1.0,
        "lemma": sent[i].lemma_,
        "pos": sent[i].pos_,
        "tag": sent[i].tag_,
        "dep": sent[i].dep_,
        "shape": sent[i].shape_,
        "is_alpha": sent[i].is_alpha,
        "is_stop": sent[i].is_stop,
        "is_title": sent[i].is_title,
        "is_punct": sent[i].is_punct,
    }
    if i > 0:
        features.update(
            {
                "-1:lemma": sent[i - 1].lemma_,
                "-1:pos": sent[i - 1].pos_,
                "-1:tag": sent[i - 1].tag_,
                "-1:dep": sent[i - 1].dep_,
                "-1:shape": sent[i - 1].shape_,
                "-1:is_alpha": sent[i - 1].is_alpha,
                "-1:is_stop": sent[i - 1].is_stop,
                "-1:is_title": sent[i - 1].is_title,
                "-1:is_left_punct": sent[i - 1].is_left_punct,
            }
        )
        if i > 1:
            features.update(
                {
                    "-2:lemma": sent[i - 2].lemma_,
                    "-2:pos": sent[i - 2].pos_,
                    "-2:tag": sent[i - 2].tag_,
                    "-2:dep": sent[i - 2].dep_,
                    "-2:shape": sent[i - 2].shape_,
                    "-2:is_alpha": sent[i - 2].is_alpha,
                    "-2:is_stop": sent[i - 2].is_stop,
                    "-2:is_title": sent[i - 2].is_title,
                    "-2:is_left_punct": sent[i - 2].is_left_punct,
                }
            )
    else:
        features["BOS"] = True

    if i < len(sent) - 1:
        features.update(
            {
                "+1:lemma": sent[i + 1].lemma_,
                "+1:pos": sent[i + 1].pos_,
                "+1:tag": sent[i + 1].tag_,
                "+1:dep": sent[i + 1].dep_,
                "+1:shape": sent[i + 1].shape_,
                "+1:is_alpha": sent[i + 1].is_alpha,
                "+1:is_stop": sent[i + 1].is_stop,
                "+1:is_title": sent[i + 1].is_title,
                "+1:is_right_punct": sent[i + 1].is_right_punct,
            }
        )
        if i < len(sent) - 2:
            features.update(
                {
                    "+2:lemma": sent[i + 2].lemma_,
                    "+2:pos": sent[i + 2].pos_,
                    "+2:tag": sent[i + 2].tag_,
                    "+2:dep": sent[i + 2].dep_,
                    "+2:shape": sent[i + 2].shape_,
                    "+2:is_alpha": sent[i + 2].is_alpha,
                    "+2:is_stop": sent[i + 2].is_stop,
                    "+2:is_title": sent[i + 2].is_title,
                    "+2:is_right_punct": sent[i + 2].is_right_punct,
                }
            )
    else:
        features["EOS"] = True

    return features

def process_data(input_data):
    # have spacy parse the input string with the full pipeline to generate features this will take some time
    input_data["input"] = list(
        nlp.pipe(input_data["input"].astype("unicode").values, batch_size=50)
    )


    # Create our features dict
    input_data = input_data["input"].apply(
        lambda doc: [word2features(doc, i) for i in range(len(doc))]
    )

    return input_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--crf", action="store_true", help="CRF Training Data")
    parser.add_argument("--epi", action="store_true", help="Epicurious Data")
    args = parser.parse_args()

    # Load spacy NLP model
    nlp = spacy.load("en_core_web_lg", disable=["ner", "textcat"])
    
    if args.crf:
        # Load cleaned data
        training_data = pd.read_pickle(os.path.join(os.path.dirname(__file__), "../../data/interim/crf_training_data.pickle"))
        test_data = pd.read_pickle(os.path.join(os.path.dirname(__file__), "../../data/interim/crf_test_data.pickle"))

        training_data = process_data(training_data)
        test_data = process_data(test_data)

        # Save features to file
        training_data.to_pickle(os.path.join(os.path.dirname(__file__), "../../data/interim/crf_training_features.pickle"))
        test_data.to_pickle(os.path.join(os.path.dirname(__file__), "../../data/interim/crf_test_features.pickle"))

    if args.epi:
        # Load cleaned data
        epi_data = pd.read_pickle(os.path.join(os.path.dirname(__file__), "../../data/interim/epi_data.pickle"))
        epi_data = process_data(epi_data)
        epi_data.to_pickle(os.path.join(os.path.dirname(__file__), "../../data/interim/epi_features.pickle"))

    
