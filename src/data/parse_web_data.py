from joblib import load
import numpy as np
import pandas as pd
import spacy
import re
from fractions import Fraction
from decimal import Decimal, InvalidOperation
import unicodedata

if __name__ == "__main__":
    crf = load("../../models/crf_model.joblib")
    nlp = spacy.load("en_core_web_lg", disable=["ner", "textcat"])

    raw_df = pd.read_json("../data/raw/recipes_raw_epi.json")
    trans_df = raw_df.transpose()
    trans_df = trans_df[trans_df.astype(str)["ingredients"] != "[]"]
