#!/usr/bin/env python3
import sys
import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

import data_cleaning_util

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def process_data(input_data):
    # Remove random HTML tags
    if args.crf:
        input_data = input_data.apply(data_cleaning_util.clean_nyt_html, axis=1)
    elif args.epi:
        input_data["input"] = input_data["input"].apply(
            data_cleaning_util.clean_epi_html
        )

    # Drop rows that have no input
    num_rows = input_data.shape[0]
    input_data.dropna(axis=0, subset=["input"], inplace=True)
    if args.v:
        print(
            "Dropped {} rows with no ingredients".format(num_rows - input_data.shape[0])
        )

    # Unicode has numerous characters to represent fractions like ¾, we remove these
    input_data["input"] = input_data["input"].apply(
        data_cleaning_util.clean_unicode_fractions
    )

    # Many ingredient quantities are written as 1 1/2 to represent 1.5
    # The quantity label however is always written as 1.5 so we need to
    # convert these fractions so that the crf can match it
    input_data["input"] = input_data["input"].apply(data_cleaning_util.merge_fractions)

    if args.crf:
        input_data = input_data.apply(data_cleaning_util.merge_quantities, axis=1)
        input_data["input"] = input_data["input"].apply(data_cleaning_util.fix_abbreviations)
        input_data["unit"] = input_data["unit"].apply(data_cleaning_util.fix_abbreviations)
        input_data = input_data.apply(data_cleaning_util.fix_inconsistencies, axis=1)
        input_data = input_data.apply(data_cleaning_util.fix_individual_rows, axis=1)
    elif args.epi:
        input_data["input"] = input_data["input"].apply(
            data_cleaning_util.fix_abbreviations
        )

    return input_data


if __name__ == "__main__":
    print(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument("--crf", action="store_true", help="CRF Training Data")
    parser.add_argument("--epi", action="store_true", help="Epicurious Data")
    parser.add_argument("-v", action="store_true", help="Verbose")
    args = parser.parse_args()
    if args.crf:
        # Load crf training data
        print("Cleaning NYT Data")
        input_data = pd.read_csv(
            os.path.join(
                os.path.dirname(__file__),
                "../../data/raw/nyt-ingredients-snapshot-2015.csv",
            ),
            index_col="index",
        )
        if args.v:
            print(input_data.head())
        input_data = process_data(input_data)
        if args.v:
            print(input_data.head())

        input_data["unit"] = input_data["unit"].fillna('')
        input_data["comment"] = input_data["comment"].fillna('')
        input_data["name"] = input_data["name"].fillna('')
        # Split data into training and testing set
        training_data, test_data = train_test_split(input_data, test_size=0.2)
        training_data.to_hdf(
            os.path.join(
                os.path.dirname(__file__), "../../data/interim/crf_training_data.h5"
            ),
            key="df",
            mode="w",
            format="fixed",
        )
        test_data.to_hdf(
            os.path.join(
                os.path.dirname(__file__), "../../data/interim/crf_test_data.h5"
            ),
            key="df",
            mode="w",
            format="fixed",
        )
    if args.epi:
        # Load data scrapped from epicurious
        print("CLEANING EPI DATA")
        epi_raw = pd.read_json(
            os.path.join(
                os.path.dirname(__file__), "../../data/raw/recipes_raw_epi.json"
            )
        )
        if args.v:
            print("RAW EPI HEAD")
            print(epi_raw.head())

        # We need to transpose the matrix so that we have a the recipes as rows and ingredients as columns
        epi_data = epi_raw.transpose()

        # Select a subset so processing happend faster
        if args.v:
            print("TRANSPOSED EPI HEAD")
            print(epi_data.head())
        #epi_data = epi_data.iloc[:100]
        # Drop rows with empty ingredients
        num_rows = epi_data.shape[0]
        epi_data = epi_data[epi_data.astype(str)["ingredients"] != "[]"]
        if args.v:
            print(
                "Dropped {} rows with no ingredients".format(
                    num_rows - epi_data.shape[0]
                )
            )

        # We do two things next, we create a new dataframe to contain all the ingredient info
        # and rearrange it so that it has one ingredient per row. All the non-ingredient columns
        # are kept in the original dataframe.
        ingredients = epi_data["ingredients"].apply(pd.Series, 1).stack()
        ingredients.index = ingredients.index.droplevel(-1)
        ingredients.name = "input"
        epi_ingredients = epi_data.join(ingredients)

        # Drop non ingredient/yield columns from ingredient dataframe
        epi_ingredients = epi_ingredients[["input"]]
        num_rows = epi_ingredients.shape[0]
        epi_ingredients = epi_ingredients[epi_ingredients["input"] != ""]
        if args.v:
            print(
                "Dropped {} rows with no ingredients".format(
                    num_rows - epi_ingredients.shape[0]
                )
            )
        print(epi_ingredients.loc["http://www.epicurious.com/recipes/food/views/roast-leg-of-lamb-with-tarragon-mint-butter-352043"])
        # Drop ingredient/yield columns from data dataframe
        del epi_data["ingredients"]

        print(epi_data["yields"].unique())

        # Convert column dtypes
        epi_data["avg_rating"] = epi_data["avg_rating"].astype('float64')
        epi_data["best_rating"] = epi_data["best_rating"].astype('float64')
        epi_data["num_reviews"] = epi_data["num_reviews"].astype('float64')
        epi_data["prepare_again_rating"] = epi_data["prepare_again_rating"].astype('float64')
        epi_data["worst_rating"] = epi_data["worst_rating"].astype('float64')
        epi_data["instructions"] = epi_data["instructions"].astype(str)
        epi_data["title"] = epi_data["title"].astype(str)
        epi_data["yields"] = epi_data["yields"].apply(lambda x: int(x.split()[0]) if x!='' else float('nan'))
        epi_data["yields"] = epi_data["yields"].astype('float64')

        epi_data["total_time"] = epi_data["total_time"].astype('float64')
        epi_data["tags"] = epi_data["tags"].apply(lambda d: d if isinstance(d, list) else [])

        if args.v:
            print("TRANSPOSED EPI HEAD")
            print(epi_data.dtypes)
            print(epi_data.head())
            print(epi_data["yields"].isnull().values.any())

            print("EPI INGREDIENTS")
            print(epi_ingredients.dtypes)
            print(epi_ingredients.head())



        # Clean the ingredients
        epi_ingredients = process_data(epi_ingredients)

        # Save both dataframes to pickle
        epi_data.to_hdf(
            os.path.join(os.path.dirname(__file__), "../../data/interim/epi_data.h5"),
            key="df",
            mode="w",
            format="fixed",
        )
        epi_ingredients.to_hdf(
            os.path.join(
                os.path.dirname(__file__), "../../data/interim/epi_ingredients.h5"
            ),
            key="df",
            mode="w",
            format="fixed",
        )
