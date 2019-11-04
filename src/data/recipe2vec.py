#!/usr/bin/env python3
import sys
import os
import argparse
import pandas as pd
from itertools import chain
from joblib import load
import numpy as np
import re

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epi", action="store_true", help="Epicurious Data")
    parser.add_argument("-v", action="store_true", help="Verbose")
    args = parser.parse_args()

    # Load our trained CRF model
    crf = load( os.path.join(
                os.path.dirname(__file__), "../../models/crf_model.joblib"
            ))
    
    if args.epi:
        print("CONVERTING EPI DATA TO VECTORS")
        # Load features
        epi_ingredients = pd.read_pickle(
            os.path.join(
                os.path.dirname(__file__), "../../data/interim/epi_features.pickle"
            )
        )
        if args.v:
            print(epi_ingredients.head(10))

        epi_ingredients["predicted"] = crf.predict(epi_ingredients["features"].values)
        epi_ingredients["input"] = epi_ingredients["input"].apply(lambda doc: [token.lemma_ for token in doc])


        parsed = epi_ingredients.apply(lambda x: format_ingredient_output(x.input, x.predicted), axis=1)

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
        epi_ingredients.loc[epi_ingredients.unit == "tablespoon", "unit"] = "milliliters"

        epi_ingredients.loc[epi_ingredients.unit == "cup", "qty"] *= 236.588
        epi_ingredients.loc[epi_ingredients.unit == "cup", "unit"] = "milliliters"

        epi_ingredients.loc[epi_ingredients.unit == "pinch", "qty"] *= 4.92892 * (1 / 16)
        epi_ingredients.loc[epi_ingredients.unit == "pinch", "unit"] = "milliliters"

        epi_ingredients.loc[epi_ingredients.unit == "dash", "qty"] *= 4.92892 * (1 / 8)
        epi_ingredients.loc[epi_ingredients.unit == "dash", "unit"] = "milliliters"

        epi_ingredients.loc[epi_ingredients.unit == "ounce", "qty"] *= 28.3495
        epi_ingredients.loc[epi_ingredients.unit == "ounce", "unit"] = "grams"

        epi_ingredients.loc[epi_ingredients.unit == "fluid ounce", "qty"] *= 29.5735
        epi_ingredients.loc[epi_ingredients.unit == "fluid ounce", "unit"] = "milliliters"

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

        epi_data = pd.read_pickle(
            os.path.join(
                os.path.dirname(__file__), "../../data/interim/epi_data.pickle"
            )
        )
        if args.v:
            print(epi_data.head())

        epi_vec = epi_ingredients.pivot_table(
            index=epi_ingredients.index, columns="name", values="qty", aggfunc=np.mean
        )
        epi_vec = epi_vec.join(epi_data[["avg_rating", "best_rating", "worst_rating", "prepare_again_rating",
                                            "num_reviews", "total_time", "tags", "title"]])

        epi_vec.fillna(0, inplace=True)
        if args.v:
            print(epi_vec.head())

        # Let's save our dataframe so we can look at it without having to reload and recompute everything later.
        epi_vec.to_pickle(
            os.path.join(
                os.path.dirname(__file__), "../../data/processed/epi_vector.pickle"
            )
        )
