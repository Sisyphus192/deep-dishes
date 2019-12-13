from collections import Counter
import decimal
import re
from fractions import Fraction
import sys
import unicodedata
import pandas as pd
from sklearn.model_selection import train_test_split
import spacy

sys.path.append("..")
from src.features import create_features
import unidecode

decimal.getcontext().rounding = decimal.ROUND_HALF_UP


def clean_nyt_html(row, verbose=False):
    """
    This will replace all html tags that were not stripped
    from the NYT data
    """
    columns = ["input", "name", "comment"]
    for col in columns:
        # This filters out NaN values so they wont get caught in the try except
        if row[col] == row[col]:
            try:
                # this will remove all: <a href=...>see recipe</a>
                match = re.findall(r"\(?<.*see\s*recipe.*>\)?", row[col])
                if match:
                    for m in match:
                        row[col] = re.sub(r"\(?<.*see\s*recipe.*>\)?", "", row[col])
                        if col == "input" and row["comment"] == row["comment"]:
                            row["comment"] = re.sub(r"see recipe", "", row["comment"])
            except TypeError:
                print("ERROR: Removing <see recipe>, " + col + " ", row)
            try:
                # this will remove all: see <a href=...>recipe</a>
                match = re.findall(r"\(?\s*(see)\s*?<.*recipe.*>\)?", row[col])
                if match:
                    for m in match:
                        row[col] = re.sub(
                            r"\(?\s*(see)\s*?<.*recipe.*>\)?", "", row[col]
                        )
                        if col == "input" and row["comment"] == row["comment"]:
                            row["comment"] = re.sub(r"see recipe", "", row["comment"])
            except TypeError:
                print("ERROR: Removing see <recipe>, " + col + " ", row)

            # This will remove all <span> and misc <a href=...>...</a>
            match = re.findall(r"<.*?>", row[col])
            if match:
                for m in match:
                    row[col] = re.sub(r"<.*?>", "", row[col])
            # this will remove all un-escapped '\n' from the original html
            match = re.findall(r"\\n", row[col])
            if match:
                for m in match:
                    row[col] = re.sub(r"\\n", " ", row[col])
            # this will remove all un-escapped '\t' from the original html
            match = re.findall(r"\\t", row[col])
            if match:
                for m in match:
                    row[col] = re.sub(r"\\t", " ", row[col])
            # if the column is now blank becasue of what we removed, set it
            # to NaN so pandas can handle it easier
            if not row[col]:
                row[col] = float("nan")
            else:
                row[col] = row[col].strip()
    return row


def clean_epi_html(ingredient):
    # this will remove all: epi:recipelink stuff
    match = re.findall(r"epi\:recipeLink id\=\"\"\d+\"\"<", ingredient)
    if match:
        for m in match:
            ingredient = re.sub(r"epi\:recipeLink id\=\"\"\d+\"\"<", "", ingredient)

    # this will remove all un-escapped '\n' from the original html
    match = re.findall(r"\\n", ingredient)
    if match:
        for m in match:
            ingredient = re.sub(r"\\n", " ", ingredient)
    # this will remove all un-escapped '\t' from the original html
    match = re.findall(r"\\t", ingredient)
    if match:
        for m in match:
            ingredient = re.sub(r"\\t", " ", ingredient)
    # if the column is now blank becasue of what we removed, set it
    # to NaN so pandas can handle it easier
    if not ingredient:
        ingredient = ""
    else:
        ingredient = ingredient.strip()
    return ingredient


def fix_spelling(string):
    if string == string:
        string = re.sub(r"([Cc])(hipolte|hipottle)", r"\1hipotle", string)
        string = re.sub(
            r"([Ff])(ritata|rittatta|ritatta|ritartar)", r"\1rittata", string
        )
        string = re.sub(r"([Cc])reme\s[Ff](resh|raishe)", r"\1reme fraiche", string)
        string = re.sub(r"([Mm])(ascapone|ascaprone)", r"\1ascarpone", string)
        string = re.sub(
            r"([Bb])(russel|russle)\s[Ss]prout", r"\1russels sprout", string
        )
        string = re.sub(r"([Gg])nocci", r"\1nocchi", string)
        string = re.sub(r"([Mm])(accaroni|acarroni)", r"\1acaroni", string)
        string = re.sub(r"([Mm])(acaroon|accaron|acarron)", r"\1acaron", string)
        string = re.sub(
            r"([Ff])(ettuccini|ettucine|ettucchine)", r"\1ettuccine", string
        )
        string = re.sub(r"([Ee])xpresso", r"\1spresso", string)
        string = re.sub(r"([Mm])(ozzarrella|ozarela|ozzarela )", r"\1ozzarella", string)
        string = re.sub(r"([Ss])herbert", r"\1herbet", string)
        string = re.sub(r"([Cc])ardamon", r"\1ardamom", string)
        string = re.sub(r"([Ll])inguini", r"\1inguine", string)
        string = re.sub(r"([Ll])iquer", r"\1iqueur", string)
        string = re.sub(r"([Ww])on\ston", r"\1onton", string)
        string = re.sub(r"([Cc])hile", r"\1hili", string)
        string = re.sub(r"([Cc])hilies", r"\1hilis", string)
        string = re.sub(r"(\&amp\;|\&)e(acute|grave)\;", "e", string)
        string = re.sub(r"(\&amp\;|\&)icirc\;", "i", string)
        string = re.sub(r"(\&amp\;|\&)ucirc\;", "u", string)
        string = re.sub(r"(\&amp\;|\&)\#231\;", "c", string)
        string = re.sub(r"(\&amp\;|\&)rsquo\;", "'", string)
        string = re.sub(r"(\&amp\;|\&)ntilde\;", "n", string)
        string = re.sub(r"redpepper", "red pepper", string)
        string = re.sub(r"blackpepper", "black pepper", string)
        string = re.sub(r"roastedalmonds", "roasted almonds", string)
        string = re.sub(r"XXshiitake", "shiitake", string)
        # Handling misc edge case
        string = re.sub(r"1 1\/2\½", "1 1/2", string)
        string = re.sub(r"1\#3", "1/3", string)
        string = re.sub(r"1\#12", "1 12", string)

    return string


def fix_characters(string):
    if string == string:
        if "\xa0" in string:
            string = string.replace("\xa0", " ")
        if "\x90" in string:
            string = string.replace("\x90", "")
        if "×" in string:
            string = string.replace("×", "x")
        # Wait to process hyphens until after ingredient ranges are processed.
        # string = re.sub(r"(?<!(?:[^\d]))[\–\—\‐\‑\-](?=(?:[^\d]))", " ", string)
        string = re.sub(r"[\!\*\|\`\@\+\?\�\™\‿\•\®\§\¤\[\]\u2028]", "", string)
        string = re.sub(r"[\‘|\’]", "''", string)
        string = re.sub(r"[\“\”\″\‟]", '"', string)
        string = re.sub(r"\&", "and", string)
        if "⁄" in string:
            string = string.replace("⁄", "/")
        # The following characters only appear a very small number of times each in the data and are removed

        if "‱" in string:
            string = string.replace("‱", "n")
        strin = string.replace("  ", " ")
    return string


# Qty in data are rounded up to two decimal places


def fix_abbreviations(string):
    """
    Converts instances of oz., ml., and g. to ounce and gram respectively
    """
    if string == string:
        match = re.findall(r"([^\w])oz\.?([^\w])?", string)
        if match:
            for m in match:
                if len(m) == 1:
                    string = re.sub(r"([^\w])oz\.?([^\w])", m[0] + "ounce", string, 1)
                else:
                    string = re.sub(
                        r"([^\w])oz\.?([^\w])", m[0] + "ounce" + m[1], string, 1
                    )
        match = re.findall(r"([^\w])lbs?\.?([^\w])?", string)
        if match:
            for m in match:
                if len(m) == 1:
                    string = re.sub(
                        r"([^\w])lbs?\.?([^\w])?", m[0] + "pound", string, 1
                    )
                else:
                    string = re.sub(
                        r"([^\w])lbs?\.?([^\w])?", m[0] + "pound" + m[1], string, 1
                    )
        # replace ml. with milliliter
        match = re.findall(r"([^\w])ml\.?([^\w])?", string)
        if match:
            for m in match:
                if len(m) == 1:
                    string = re.sub(
                        r"([^\w])ml\.?([^\w])", m[0] + "milliliter", string, 1
                    )
                else:
                    string = re.sub(
                        r"([^\w])ml\.?([^\w])", m[0] + "milliliter" + m[1], string, 1
                    )
        # replace g. with gram
        match = re.findall(r"(\d+)\s?g\.?([^\w])", string)
        if match:
            for m in match:
                string = re.sub(
                    r"(\d+)\s?g\.?([^\w])", m[0] + " gram" + m[1], string, 1
                )

        # replace tbsp with tablespoon
        match = re.findall(r"[Tt]bsp\.*", string)
        if match:
            for m in match:
                string = re.sub(r"[Tt]bsp\.*", "tablespoon", string, 1)

        # replace tsp with teaspoon
        match = re.findall(r"[Tt]sp\.*", string)
        if match:
            for m in match:
                string = re.sub(r"[Tt]sp\.*", "teaspoon", string, 1)
    return string


numbers = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "dozen": 12,
}


def fix_numeric_words(ingredient):

    ingredient = re.sub(
        r"(\sone and a half|\sone and one[\s\-]half)", " 1.5", ingredient
    )
    ingredient = re.sub(r"one and one[\s\-]quarter", "1.25", ingredient)
    ingredient = re.sub(r"two and one[\s\-]quarter", "2.25", ingredient)
    ingredient = re.sub(r"two and one[\s\-]half", "2.5", ingredient)
    ingredient = re.sub(r"three and a half", "3.5", ingredient)
    match = re.findall(
        r"(?<!(?:[^\w]))([Oo]ne|[Tt]wo|[Tt]hree|[Ff]our|[Ff]ive|[Ss]ix|[Ss]even|[Ee]ight|[Nn]ine|[Tt]en|[Dd]ozen)(?=(?:[^\w]))",
        ingredient,
    )
    for m in match:
        ingredient = re.sub(
            r"(?<!(?:[^\w]))([Oo]ne|[Tt]wo|[Tt]hree|[Ff]our|[Ff]ive|[Ss]ix|[Ss]even|[Ee]ight|[Nn]ine|[Tt]en|[Dd]ozen)(?=(?:[^\w]))",
            str(numbers[m.lower()]),
            ingredient,
        )
    return ingredient


decimal.getcontext().rounding = decimal.ROUND_HALF_UP


def clean_unicode_fractions(string):
    """
    Replace unicode fractions with ascii representation, preceded by a
    space.

    "1\x215e" => "1 7/8"
    """

    # match all mixed fractions with a unicode fraction (e.g. 1 ¾ or 1¾) and add them together
    # UNHANDLED EDGE CASE: There are a handful of ingredients in which the whole number is a quantity
    # mulitplier and not part of the fraction, e.g. 2 1/4 in cinnamon sticks, should be 0.5 not 2.25
    match = re.findall(r"(\d+\s?)?([\u2150-\u215E\u00BC-\u00BE])", string)
    if match:
        for m in match:
            if not m[0]:  # single unicode fraction e.g. ¾
                num = float(Fraction(unicodedata.numeric(m[1])))
            else:  # mixed unicode fraction e.g. 1¾
                num = float(m[0]) + float(Fraction(unicodedata.numeric(m[1])))
            num = decimal.Decimal(num)
            num = round(num, 2)
            num = str(num.normalize())
            string = re.sub(r"(\d+\s?)?([\u2150-\u215E\u00BC-\u00BE])", num, string, 1)

    return string


# Qty in data are rounded up to two decimal places


def merge_fractions(string):
    """
    Merges mixed fractions: 1 2/3 => 1.67
    """
    # This filters out NaN values so they wont get caught in the try except
    decimal.getcontext().rounding = decimal.ROUND_HALF_UP
    if string == string:
        match = re.findall(r"(\d+)[\-\s](\d+\/\d+)", string)
        if match:
            for m in match:
                num = float(m[0]) + float(Fraction(m[1]))
                num = decimal.Decimal(num)
                num = round(num, 2)
                if "E" in str(num.normalize()):
                    num = str(num.quantize(decimal.Decimal("1")))
                else:
                    num = str(num.normalize())
                string = re.sub(r"(\d+)[\-\s](\d+\/\d+)", num, string, 1)

        match = re.findall(r"(\d+\/\d+)", string)
        if match:
            for m in match:
                num = float(Fraction(m))
                num = decimal.Decimal(num)
                num = round(num, 2)
                if "E" in str(num.normalize()):
                    num = str(num.quantize(decimal.Decimal("1")))
                else:
                    num = str(num.normalize())
                string = re.sub(r"(\d+\/\d+)", num, string, 1)

    return string


def merge_quantities(string):
    """
    Many ingredients are written in the form 2 8.5-ounce cans...
    This is both tricky for the model to parse and made worse because
    the labeled data incosistently labels the quanity as 2, 8.5, or 17.
    We want to reuce all these to a single value:
    2 8.5-ounce => 17.0-ounce
    and update the quantity label as appropriate
    """
    decimal.getcontext().rounding = decimal.ROUND_HALF_UP
    if string == string:
        # Ok first we need to average any number ranges, e.g. "3 to 4 pounds" becomes "3.5 pounds"

        match = re.findall(r"(\d+\.?\d*)[\s\-]*[tor\-]+[\s\-]*(\d+\.?\d*)", string)
        if match:
            for m in match:
                num = (float(m[0]) + float(m[1])) / 2
                num = decimal.Decimal(num)
                num = round(num, 2)
                if "E" in str(num.normalize()):
                    num = str(num.quantize(decimal.Decimal("1")))
                else:
                    num = str(num.normalize())
                string = re.sub(
                    r"(\d+\.?\d*)[\s\-]*[tor\-]+[\s\-]*(\d+\.?\d*)", num, string, 1
                )

        # now we do quantity multipliers
        match = re.findall(r"(\d+)\s+(\d+\.*\d*)", string)
        if match:
            for m in match:
                num = float(m[0]) * float(m[1])
                num = decimal.Decimal(num)
                num = round(num, 2)
                if "E" in str(num.normalize()):
                    num = str(num.quantize(decimal.Decimal("1")))
                else:
                    num = str(num.normalize())
                string = re.sub(r"(\d+)\s+(\d+\.*\d*)", num, string, 1)
        # Remove hyphens we skipped before
        string = re.sub(r"[\–\—\‐\‑\-]", " ", string)

    return string


def fix_measurements(string, nlp = None):
    input_tokens = nlp(string)
    lemmatized_input = " ".join([x.lemma_ for x in input_tokens])

    match = re.findall(
        r"^(\d+\.?\d*) (\w+) (plus|and) (\d+\.?\d*) (\w+)", lemmatized_input
    )
    for m in match:
        if m[1] in measurementUnit and m[4] in measurementUnit:
            num = 0
            unit = ""
            # sticks and tablespoons of butter
            if (
                m[1] == "stick"
                and m[4] == "tablespoon"
                and "butter" in lemmatized_input
            ):
                num += float(m[0]) * 113.398
                num += float(m[3]) * 14.18
                unit = "gram"
            elif (
                m[1] == "pound"
                and m[4] == "tablespoon"
                and "butter" in lemmatized_input
            ):
                num += float(m[0]) * 453.592
                num += float(m[3]) * 14.18
                unit = "gram"
            elif (
                m[1] == "ounce"
                and m[4] == "tablespoon"
                and "butter" in lemmatized_input
            ):
                num += float(m[0]) * 28.3495
                num += float(m[3]) * 14.18
                unit = "gram"
            elif m[1] == "pound" and m[4] == "ounce":
                num += float(m[0]) * 453.592
                num += float(m[3]) * 28.3495
                unit = "gram"
            elif m[1] == "head" and m[4] == "clove" and "garlic" in lemmatized_input:
                num += float(m[0]) * 56.699
                num += float(m[3]) * 5.15
                unit = "gram"
            elif (
                m[1] == "tablespoon"
                and m[4] == "clove"
                and "garlic" in lemmatized_input
            ):
                num += float(m[0]) * (5.15 * 3)
                num += float(m[3]) * 5.15
                unit = "gram"
            elif (
                m[1] == "packet"
                and m[4] == "teaspoon"
                and "gelatin" in lemmatized_input
            ):
                num += float(m[0]) * 7
                num += float(m[3]) * 3.08
                unit = "gram"
            else:
                if m[1] == "teaspoon":
                    num += float(m[0]) * 4.92892
                elif m[1] == "tablespoon":
                    num += float(m[0]) * 14.7868
                elif m[1] == "ounce":
                    num += float(m[0]) * 29.5735
                elif m[1] == "cup":
                    num += float(m[0]) * 236.588
                elif m[1] == "quart":
                    num += float(m[0]) * 946.353
                elif m[1] == "bottle" and "wine" in lemmatized_input:
                    num += float(m[0]) * 750
                else:
                    print(string)
                if m[4] == "teaspoon":
                    num += float(m[3]) * 4.92892
                elif m[4] == "tablespoon":
                    num += float(m[3]) * 14.7868
                elif m[4] == "ounce":
                    num += float(m[3]) * 29.5735
                elif m[4] == "cup":
                    num += float(m[3]) * 236.588
                elif m[4] == "pinch":
                    num += float(m[3]) * 0.31
                else:
                    print(string)
                    print(lemmatized_input)
                unit = "milliliter"
            string = re.sub(
                r"^(\d+\.?\d*) (\w+) (plus|and) (\d+\.?\d*) (\w+)",
                str(round(num, 2)) + " " + unit,
                string,
            )
    return string
