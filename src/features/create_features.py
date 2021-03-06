#!/usr/bin/env python3
import sys
import os
import uuid
import argparse
import spacy
import pandas as pd
from joblib import load
import re
import numpy as np

def qty2float(qty):
    try:
        qty = float(qty)
    except ValueError:
        qty = np.nan

    return qty


def smartJoin(words):
    """
    Joins list of words with spaces, but is smart about not adding spaces
    before commas.
    """

    input = " ".join(words)

    # replace " , " with ", "
    input = input.replace(" , ", ", ")

    # replace " ( " with " ("
    input = input.replace("( ", "(")

    # replace " ) " with ") "
    input = input.replace(" )", ")")

    return input


def format_ingredient_output(tokens, tags, display=False):
    """Formats the tagger output into a more convenient dictionary"""
    data = [{}]
    display = [[]]
    prevTag = None

    for token, tag in zip(tokens, tags):
        # turn B-NAME/123 back into "name"
        tag = re.sub(r"^[BI]\-", "", tag).lower()
        # ---- DISPLAY ----
        # build a structure which groups each token by its tag, so we can
        # rebuild the original display name later.

        if prevTag != tag:
            display[-1].append((tag, [token]))
            prevTag = tag
        else:
            display[-1][-1][1].append(token)
            #               ^- token
            #            ^---- tag
            #        ^-------- ingredient

            # ---- DATA ----
            # build a dict grouping tokens by their tag

            # initialize this attribute if this is the first token of its kind
        if tag not in data[-1]:
            data[-1][tag] = []

        data[-1][tag].append(token)

    # reassemble the output into a list of dicts.
    output = [
        dict([(k, smartJoin(tokens)) for k, tokens in ingredient.items()])
        for ingredient in data
        if len(ingredient)
    ]

    # Add the raw ingredient phrase
    for i, v in enumerate(output):
        output[i]["input"] = smartJoin([" ".join(tokens) for k, tokens in display[i]])

    return output[0]

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
    features = input_data["input"].apply(
        lambda doc: [word2features(doc, i) for i in range(len(doc))]
    )
    return features


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--crf", action="store_true", help="CRF Training Data")
    parser.add_argument("--epi", action="store_true", help="Epicurious Data")
    parser.add_argument("--mba", action="store_true", help="Market Basket Analysis")
    parser.add_argument("-v", action="store_true", help="Verbose")
    args = parser.parse_args()

    # Load spacy NLP model
    nlp = spacy.load("en_core_web_lg", disable=["ner", "textcat"])
    
    if args.crf:
        # Load cleaned data
        training_data = pd.read_hdf(os.path.join(os.path.dirname(__file__), "../../data/interim/crf_training_data.h5"), 'df')
        test_data = pd.read_hdf(os.path.join(os.path.dirname(__file__), "../../data/interim/crf_test_data.h5"), 'df')

        if args.v:
            print(training_data.head())
            print(test_data.head())

        training_features = process_data(training_data)
        test_features = process_data(test_data)

        if args.v:
            print(type(training_features))
            print(training_features.head())
            print(test_features.head())

        # Save features to file
        training_features.to_hdf(os.path.join(os.path.dirname(__file__), "../../data/interim/crf_training_features.h5"), key="df", mode='w', format="fixed")
        test_features.to_hdf(os.path.join(os.path.dirname(__file__), "../../data/interim/crf_test_features.h5"), key="df", mode='w', format="fixed")

    if args.epi:
        print("CREATING FEATURES FOR EPI DATA")
        # Load cleaned data
        epi_ingredients = pd.read_hdf(os.path.join(os.path.dirname(__file__), "../../data/interim/epi_ingredients.h5"), 'df')
        if args.v:
            print(epi_ingredients.head())

        epi_ingredients["input"] = list(
            nlp.pipe(epi_ingredients["input"].astype("unicode").values, batch_size=50)
        )

        

        features = epi_ingredients["input"].apply(
        lambda doc: [word2features(doc, i) for i in range(len(doc))]
        )

        epi_ingredients["features"] = pd.DataFrame.from_dict(features)


        #lemmas = epi_ingredients["input"].apply(pd.Series, 1).stack()
        #lemmas.index = lemmas.index.droplevel(-1)
        #lemmas.name = "lemmas"
        #epi_ingredients = epi_ingredients.join(lemmas)
        #print(epi_ingredients.head())

        # Create Features
        #epi_features = pd.DataFrame(process_data(epi_ingredients))
        #if args.v:
        #    print(epi_features.head())

        #epi_ingredients.to_hdf(os.path.join(os.path.dirname(__file__), "../../data/interim/epi_features.h5"), key="df", mode='w', format="table")


         # Load our trained CRF model
        crf = load(os.path.join(os.path.dirname(__file__), "../../models/crf_model.joblib"))

        epi_ingredients["predicted"] = crf.predict(epi_ingredients["features"].values)
        epi_ingredients["input"] = epi_ingredients["input"].apply(
            lambda doc: [token.lemma_ for token in doc]
        )

        parsed = epi_ingredients.apply(
            lambda x: format_ingredient_output(x.input, x.predicted), axis=1
        )

        epi_ingredients = pd.DataFrame(parsed.tolist(), index=parsed.index)
        if args.v:
            print(epi_ingredients.head())
        epi_ingredients["qty"] = epi_ingredients["qty"].apply(lambda x: qty2float(x))

        # Now we convert as many units as possible to metric
        epi_ingredients.loc[epi_ingredients.unit == "pound", "qty"] *= 453.592
        epi_ingredients.loc[epi_ingredients.unit == "pound", "unit"] = "grams"

        epi_ingredients.loc[epi_ingredients.unit == "teaspoon", "qty"] *= 4.92892
        epi_ingredients.loc[epi_ingredients.unit == "teaspoon", "unit"] = "milliliters"

        epi_ingredients.loc[epi_ingredients.unit == "tablespoon", "qty"] *= 14.7868
        epi_ingredients.loc[
            epi_ingredients.unit == "tablespoon", "unit"
        ] = "milliliters"

        epi_ingredients.loc[epi_ingredients.unit == "cup", "qty"] *= 236.588
        epi_ingredients.loc[epi_ingredients.unit == "cup", "unit"] = "milliliters"

        epi_ingredients.loc[epi_ingredients.unit == "pinch", "qty"] *= 4.92892 * (
            1 / 16
        )
        epi_ingredients.loc[epi_ingredients.unit == "pinch", "unit"] = "milliliters"

        epi_ingredients.loc[epi_ingredients.unit == "dash", "qty"] *= 4.92892 * (1 / 8)
        epi_ingredients.loc[epi_ingredients.unit == "dash", "unit"] = "milliliters"

        epi_ingredients.loc[epi_ingredients.unit == "ounce", "qty"] *= 28.3495
        epi_ingredients.loc[epi_ingredients.unit == "ounce", "unit"] = "grams"

        epi_ingredients.loc[epi_ingredients.unit == "fluid ounce", "qty"] *= 29.5735
        epi_ingredients.loc[
            epi_ingredients.unit == "fluid ounce", "unit"
        ] = "milliliters"

        epi_ingredients.loc[epi_ingredients.unit == "pint", "qty"] *= 473.176
        epi_ingredients.loc[epi_ingredients.unit == "pint", "unit"] = "milliliters"

        epi_ingredients.loc[epi_ingredients.unit == "quart", "qty"] *= 946.353
        epi_ingredients.loc[epi_ingredients.unit == "quart", "unit"] = "milliliters"

        epi_ingredients.loc[epi_ingredients.unit == "liter", "qty"] *= 1000
        epi_ingredients.loc[epi_ingredients.unit == "liter", "unit"] = "milliliters"

        epi_ingredients.loc[epi_ingredients.unit == "gallon", "qty"] *= 3785.41
        epi_ingredients.loc[epi_ingredients.unit == "gallon", "unit"] = "milliliters"

        epi_ingredients.loc[epi_ingredients.unit == "drop", "qty"] *= 0.05
        epi_ingredients.loc[epi_ingredients.unit == "drop", "unit"] = "milliliters"

        epi_ingredients.loc[epi_ingredients.unit == "jigger", "qty"] *= 44.3603
        epi_ingredients.loc[epi_ingredients.unit == "jigger", "unit"] = "milliliters"


        if args.v:
            print(epi_ingredients.head())

        epi_data = pd.read_hdf(
            os.path.join(os.path.dirname(__file__), "../../data/interim/epi_data.h5"), 'df')
        if args.v:
            print(epi_data.head())

        epi_ingredients["qty"] = epi_ingredients['qty'] / epi_data["yields"]
        epi_vec = epi_ingredients.pivot_table(
            index=epi_ingredients.index, columns="name", values="qty", aggfunc=np.mean
        )
        epi_vec = epi_vec.join(
            epi_data[
                [
                    "avg_rating",
                    "best_rating",
                    "worst_rating",
                    "prepare_again_rating",
                    "num_reviews",
                    "total_time",
                    "tags",
                    "title",
                ]
            ]
        )

        epi_vec.fillna(0, inplace=True)
        if args.v:
            print(epi_vec.head())

        # Let's save our dataframe so we can look at it without having to reload and recompute everything later.
        epi_vec.to_hdf(
            os.path.join(
                os.path.dirname(__file__), "../../data/processed/epi_vector.h5"),
            key="df",
            mode="w",
            format="fixed")
    

    if args.mba:
        if os.path.isfile(os.path.join(os.path.dirname(__file__), "../../data/processed/epi_vector.h5")):

            epi_df = pd.read_hdf(os.path.join(os.path.dirname(__file__), "../../data/processed/epi_vector.h5"))
            dat = epi_df.values[:, :-8]
            columns = list(epi_df)
            new_vec = []
            for recipe in dat:
                ind = uuid.uuid4().hex
                for i in range(len(recipe)):
                    if recipe[i] != 0:
                        new_vec.append([ind, columns[i]])

            basketized = pd.DataFrame(new_vec, columns=["index", "name"]).set_index('index', drop=False)
            del basketized["index"]
            print(basketized.head())

            basketized.to_hdf(os.path.join(
                os.path.dirname(__file__), "../../data/processed/basketized.h5"),
                key="df",
                mode="w",
                format="fixed")
        else:
            print("First, generate epi_vector file.")
