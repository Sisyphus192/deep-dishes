from collections import Counter
import decimal
import re
from fractions import Fraction
import sys
import unicodedata
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append("..")
from src.features import create_features
import unidecode

decimal.getcontext().rounding = decimal.ROUND_HALF_UP


def fix_characters(string):
    if "\xa0" in string:
        string = string.replace("\xa0", " ")
    if "\x90" in string:
        string = string.replace("\x90", "")
    if "×" in string:
        string = string.replace("×", "x")
    string = re.sub(r"–|—|‐|‑", "-", string)
    if "‘" in string:
        string = string.replace("‘", "'")
    if "’" in string:
        string = string.replace("’", "'")
    string = re.sub(r"“|”|″|‟", '"', string)
    if "\u2028" in string:
        string = string.replace("\u2028", "")
    if "⁄" in string:
        string = string.replace("⁄", "/")
    # The following characters only appear a very small number of times each in the data and are removed
    if "|" in string:
        string = string.replace("|", "")
    if "!" in string:
        string = string.replace("!", "")
    if "`" in string:
        string = string.replace("`", "")
    if "@" in string:
        string = string.replace("@", "")
    if "+" in string:
        string = re.sub(r"\+{2,}", "", string)
    if "[" in string:
        string = string.replace("[", "")
    if "]" in string:
        string = string.replace("]", "")
    if "?" in string:
        string = string.replace("?", "")
    if "�" in string:
        string = string.replace("™", "")
    if "™" in string:
        string = string.replace("™", "")
    if "‿" in string:
        string = string.replace("‿", "")
    # for whatever reason the n in jalepeno is scraped as this character
    if "‱" in string:
        string = string.replace("‱", "n")
    if "•" in string:
        string = string.replace("•", "")
    if "®" in string:
        string = string.replace("®", "")
    if "§" in string:
        string = string.replace("§", "")
    if "¤" in string:
        string = string.replace("¤", "")
    if "-" in string:
        string = re.sub(r"(\d+)\-(\w)", r"\1 - \2", string)

    return string


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
        string = re.sub(r"(\&amp\;|\&\;?)e?ntilde\;?", "n", string)
        # Handling misc edge case
        string = re.sub(r"1\#3", "1/3", string)
        string = re.sub(r"1\#12", "1 12", string)

    return string


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
    match = re.findall(r'epi\:recipeLink id\=\"\"\d+\"\"<', ingredient)
    if match:
        for m in match:
            ingredient = re.sub(r'epi\:recipeLink id\=\"\"\d+\"\"<', "", ingredient)

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


def clean_unicode_fractions(ingredient):
    """
    Replace unicode fractions with ascii representation, preceded by a
    space.

    "1\x215e" => "1 7/8"
    """
    try:
        # match all mixed fractions with a unicode fraction (e.g. 1 ¾ or 1¾) and add them together
        match = re.findall(r"(\d+)\s?([\u2150-\u215E\u00BC-\u00BE])", ingredient)
        if match:
            for m in match:
                num = float(m[0]) + float(Fraction(unicodedata.numeric(m[1])))
                ingredient = re.sub(
                    r"(\d+)\s?([\u2150-\u215E\u00BC-\u00BE])",
                    str(round(num, 3)),
                    ingredient,
                )

        # match all other unicode fractions
        match = re.findall(r"([\u2150-\u215E\u00BC-\u00BE])", ingredient)
        if match:
            for m in match:
                ingredient = re.sub(
                    r"([\u2150-\u215E\u00BC-\u00BE])",
                    str(round(float(Fraction(unicodedata.numeric(m))), 3)),
                    ingredient,
                )
    except TypeError:
        print("ERROR CLEANING UNICODE: ", ingredient)
    return ingredient


def merge_fractions(ingredient):
    """
    Merges mixed fractions: 1 2/3 => 1.67
    """

    match = re.findall(r"(\d+)[\-\s]?(\d+\/\d+)", ingredient)
    if match:
        for m in match:
            num = float(m[0]) + float(Fraction(m[1]))
            num = decimal.Decimal(num)
            ingredient = re.sub(
                r"(\d+)[\-\s]?(\d+\/\d+)", str(round(num, 2)), ingredient, 1
            )

    match = re.findall(r"(\d+\/\d+)", ingredient)
    if match:
        for m in match:
            num = float(Fraction(m))
            num = decimal.Decimal(num)
            ingredient = re.sub(r"(\d+\/\d+)", str(round(num, 2)), ingredient, 1)

    return ingredient


def merge_quantities(ingredient):
    """
    Many ingredients are written in the form 2 8.5-ounce cans...
    This is both tricky for the model to parse and made worse because
    the labeled data incosistently labels the quanity as 2, 8.5, or 17.
    We want to reuce all these to a single value:
    2 8.5-ounce => 17.0-ounce
    and update the quantity label as appropriate
    """
    try:

        # Ok first we need to average any number ranges, e.g. "3 to 4 pounds" becomes "3.5 pounds"
        match = re.findall(r"(\d+\.?\d*)[\s\-]*[tor]+[\s\-]*(\d+\.?\d*)", ingredient)
        if match:
            for m in match:
                num = (float(m[0]) + float(m[1])) / 2
                num = decimal.Decimal(num)
                ingredient = re.sub(
                    r"(\d+\.?\d*)[\s\-]*[tor]+[\s\-]*(\d+\.?\d*)",
                    str(round(num, 2)),
                    ingredient,
                    1,
                )

        # now we do quantity multipliers
        match = re.findall(r"(\d+)\s+(\d+\.*\d*)", ingredient)
        if match:
            for m in match:
                num = float(m[0]) * float(m[1])
                num = decimal.Decimal(num)
                ingredient = re.sub(
                    r"(\d+)\s+(\d+\.*\d*)", str(round(num, 2)), ingredient, 1
                )

    except TypeError:
        print("Error Merging Ranges: ", ingredient)
    return ingredient


def fix_abbreviations(row):
    """
    Converts instances of oz. and g. to ounce and gram respectively
    """
    columns = ["input", "unit", "comment"]
    for col in columns:
        # replace oz. with ounce
        if row[col] == row[col]:
            match = re.findall(r"([^\w])oz\.?([^\w])?", row[col])
            if match:
                for m in match:
                    if len(m) == 1:
                        row[col] = re.sub(
                            r"([^\w])oz\.?([^\w])", m[0] + "ounce", row[col], 1
                        )
                    else:
                        row[col] = re.sub(
                            r"([^\w])oz\.?([^\w])", m[0] + "ounce" + m[1], row[col], 1
                        )
            # replace g. with gram
            match = re.findall(r"(\d+)\s?g\.?([^\w])", row[col])
            if match:
                for m in match:
                    row[col] = re.sub(
                        r"(\d+)\s?g\.?([^\w])", m[0] + " gram" + m[1], row[col], 1
                    )

            # replace tbsp with tablespoon
            match = re.findall(r"[Tt]bsp\.*", row[col])
            if match:
                for m in match:
                    row[col] = re.sub(r"[Tt]bsp\.*", "tablespoon", row[col], 1)

            # replace tsp with teaspoon
            match = re.findall(r"[Tt]sp\.*", row[col])
            if match:
                for m in match:
                    row[col] = re.sub(r"[Tt]sp\.*", "teaspoon", row[col], 1)
                # there are a handful of instances when "tsp."" is present that the unit is either missing or incorrectly labeled: "tablespoon"
                row["unit"] = "teaspoon"
    return row


def fix_epi_abbreviations(ingredient):
    """
    Converts instances of oz., ml., and g. to ounce and gram respectively
    """

    match = re.findall(r"([^\w])oz\.?([^\w])?", ingredient)
    if match:
        for m in match:
            if len(m) == 1:
                ingredient = re.sub(
                    r"([^\w])oz\.?([^\w])", m[0] + "ounce", ingredient, 1
                )
            else:
                ingredient = re.sub(
                    r"([^\w])oz\.?([^\w])", m[0] + "ounce" + m[1], ingredient, 1
                )
    # replace ml. with milliliter
    match = re.findall(r"([^\w])ml\.?([^\w])?", ingredient)
    if match:
        for m in match:
            if len(m) == 1:
                ingredient = re.sub(
                    r"([^\w])ml\.?([^\w])", m[0] + "milliliter", ingredient, 1
                )
            else:
                ingredient = re.sub(
                    r"([^\w])ml\.?([^\w])", m[0] + "milliliter" + m[1], ingredient, 1
                )
    # replace g. with gram
    match = re.findall(r"(\d+)\s?g\.?([^\w])", ingredient)
    if match:
        for m in match:
            ingredient = re.sub(
                r"(\d+)\s?g\.?([^\w])", m[0] + " gram" + m[1], ingredient, 1
            )

    # replace tbsp with tablespoon
    match = re.findall(r"[Tt]bsp\.*", ingredient)
    if match:
        for m in match:
            ingredient = re.sub(r"[Tt]bsp\.*", "tablespoon", ingredient, 1)

    # replace tsp with teaspoon
    match = re.findall(r"[Tt]sp\.*", ingredient)
    if match:
        for m in match:
            ingredient = re.sub(r"[Tt]sp\.*", "teaspoon", ingredient, 1)

    return ingredient


def fix_inconsistencies(row):
    if row["unit"] == row["unit"]:
        match = re.findall(r"(\w+)\s(sprigs?)", row["unit"])
        if match:
            for m in match:
                if row["comment"] == row["comment"]:
                    row["comment"] += " " + m[0]
                else:
                    row["comment"] = m[0]
                row["unit"] = m[1]
    return row


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
}


def fix_numeric_words(ingredient):

    ingredient = re.sub(r"(\sone and a half|\sone and one-half)", " 1.5", ingredient)
    ingredient = re.sub(r"one and one-quarter", "1.25", ingredient)
    ingredient = re.sub(r"two and one-quarter", "2.25", ingredient)
    ingredient = re.sub(r"two and one-half", "2.5", ingredient)
    ingredient = re.sub(r"three and a half", "3.5", ingredient)
    match = re.findall(
        r"\s(one|two|three|four|five|six|seven|eight|nine|ten)\s", ingredient
    )
    for m in match:
        ingredient = re.sub(
            r"\s(one|two|three|four|five|six|seven|eight|nine|ten)\s",
            " " + str(numbers[m]) + " ",
            ingredient,
        )
    return ingredient


def clean_epi_unicode_fractions(ingredient):
    """
    Replace unicode fractions with ascii representation, preceded by a
    space.

    "1\x215e" => "1 7/8"
    """

    # match all mixed fractions with a unicode fraction (e.g. 1 ¾ or 1¾) and add them together
    # UNHANDLED EDGE CASE: There are a handful of ingredients in which the whole number is a quantity
    # mulitplier and not part of the fraction, e.g. 2 1/4 in cinnamon sticks, should be 0.5 not 2.25
    match = re.findall(r"(\d+\s?)?([\u2150-\u215E\u00BC-\u00BE])", ingredient)
    if match:
        for m in match:
            if not m[0]:  # single unicode fraction e.g. ¾
                num = float(Fraction(unicodedata.numeric(m[1])))
            else:  # mixed unicode fraction e.g. 1¾
                num = float(m[0]) + float(Fraction(unicodedata.numeric(m[1])))
            num = decimal.Decimal(num)
            num = str(round(num, 2))
            ingredient = re.sub(
                r"(\d+\s?)?([\u2150-\u215E\u00BC-\u00BE])", num, ingredient, 1
            )

    return ingredient


def fix_individual_rows(row):
    if row["input"] == "3 crushed red peppers":
        row["unit"] = float("nan")
        row["comment"] = "crushed"
    if row["input"] == "1 small-to-medium daikon radish (cut into 1-inch cubes)":
        row["comment"] = row["unit"] + " " + row["comment"]
        row["unit"] = float("nan")
    if row["unit"] == "chopped":
        row["comment"] = "chopped"
        row["unit"] = float("nan")
    if row["input"] == "1 heaping teaspoon black peppercorns":
        row["comment"] = "heaping"
        row["unit"] = "teaspoon"
    if row["input"] == "a 10-pound piece of pork belly with the skin":
        row["qty"] = 10
        row["unit"] = "pound"
    if row["input"] == "1 long soft baguette or loaf Cuban bread":
        row["unit"] = float("nan")
        row["comment"] = "long"
    if row["input"] == "2 long red chilies, seeded and finely sliced":
        row["unit"] = float("nan")
        row["comment"] = "long " + row["comment"]
    if row["input"] == "2 scant cups all-purpose flour":
        row["unit"] = "cups"
        row["comment"] = "scant"
    if row["input"] == "4 ounces (1 stick) unsalted butter":
        row["unit"] = "ounces"
        row["comment"] = "(1 stick)"
    return row
