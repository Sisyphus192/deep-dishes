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
    input_data = input_data.apply(data_cleaning_util.clean_html, axis=1)

    # Drop rows that have no input
    input_data.dropna(axis=0, subset=["input"], inplace=True)

    # Unicode has numerous characters to represent fractions like ¾, we remove these
    input_data["input"] = input_data["input"].apply(data_cleaning_util.clean_unicode_fractions)

    # Many ingredient quantities are written as 1 1/2 to represent 1.5
    # The quantity label however is always written as 1.5 so we need to
    # convert these fractions so that the crf can match it
    input_data["input"] = input_data["input"].apply(data_cleaning_util.merge_fractions)

    input_data = input_data.apply(data_cleaning_util.fix_abbreviations, axis=1)

    return input_data

if __name__ == '__main__':
    print(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument("--crf", action="store_true", help="CRF Training Data")
    parser.add_argument("--epi", action="store_true", help="Epicurious Data")
    args = parser.parse_args()
    if args.crf:
        # Load crf training data
        print("Cleaning NYT Data")
        input_data = pd.read_csv(
            os.path.join(os.path.dirname(__file__), "../../data/raw/nyt-ingredients-snapshot-2015.csv"), index_col="index"
        )
        input_data = process_data(input_data)
        # Split data into training and testing set
        training_data, test_data = train_test_split(input_data, test_size=0.2)
        training_data.to_pickle(os.path.join(os.path.dirname(__file__), "../../data/interim/crf_training_data.pickle"))
        test_data.to_pickle(os.path.join(os.path.dirname(__file__), "../../data/interim/crf_test_data.pickle"))
    if args.epi:
        # Load data scrapped from epicurious
        print("CLEANING EPI DATA")
        raw_df = pd.read_json(os.path.join(os.path.dirname(__file__), "../../data/raw/recipes_raw_epi.json"))
        trans_df = raw_df.transpose()
        trans_df = trans_df[trans_df.astype(str)['ingredients'] != '[]']
        s = trans_df['ingredients'].apply(pd.Series, 1).stack()
        s.index = s.index.droplevel(-1)
        s.name = 'ingredient'
        del trans_df['ingredients']
        trans_df = trans_df.join(s)
        trans_df = process_data(trans_df)
        trans_df.to_pickle(os.path.join(os.path.dirname(__file__), "../../data/interim/epi_data.pickle"))

    

    

        

        
