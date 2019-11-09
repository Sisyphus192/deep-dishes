import re
from fractions import Fraction
import unicodedata


def clean_nyt_html(row):
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
                match = re.findall(r"\s*\(?<.*see\s*recipe.*>\)?", row[col])
                if match:
                    for m in match:
                        row[col] = re.sub(r"\s*\(?<.*see\s*recipe.*>\)?", "", row[col])
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
            match = re.findall(r"\s*\\n\s*", row[col])
            if match:
                for m in match:
                    row[col] = re.sub(r"\\n", " ", row[col])
            # if the column is now blank becasue of what we removed, set it
            # to NaN so pandas can handle it easier
            if not row[col]:
                row[col] = float("nan")
            else:
                row[col] = row[col].strip()
    return row


def clean_epi_html(ingredient):
    # this will remove all: epi:recipelink stuff
    match = re.findall(r"\<?\/?epi:recipelink\>?", ingredient)
    if match:
        for m in match:
            ingredient = re.sub(r"\<?\/?epi:recipelink\>?", "", ingredient)
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
    try:
        match = re.findall(r"(\d+)\s+(\d\/\d)", ingredient)
        if match:
            for m in match:
                num = float(m[0]) + float(Fraction(m[1]))
                ingredient = re.sub(r"(\d+)\s+(\d\/\d)", str(round(num, 3)), ingredient)

        match = re.findall(r"(\d\/\d)", ingredient)
        if match:
            for m in match:
                num = float(Fraction(m))
                ingredient = re.sub(r"(\d\/\d)", str(round(num, 3)), ingredient)
    except ZeroDivisionError:
        print(ingredient)
    return ingredient


def merge_quantities(row):
    try:
        match = re.findall(r"(\d+\.?\d*)\-?[\s\-][tor]+[\s\-](\d+\.?\d*)", row["input"])
        if match:
            for m in match:
                num = (float(m[0]) + float(m[1])) / 2
                row["input"] = re.sub(
                    r"(\d+\.?\d*)\-?[\s\-][tor]+[\s\-](\d+\.?\d*)",
                    str(round(num, 3)),
                    row["input"],
                )
        match = re.findall(r"(\d+)\s+(\d+\.*\d*)", row["input"])
        if match:
            for m in match:
                num = float(m[0]) * float(m[1])
                row["input"] = re.sub(
                    r"(\d+)\s+(\d+\.*\d*)", str(round(num, 3)), row["input"]
                )
                if float(m[0]) == row["qty"] or float(m[1]) == row["qty"]:
                    # probably a pretty good guess that the qty was only one of these two numbers, update it with the new num
                    row["qty"] = round(num, 3)
    except TypeError:
        print("Error Merging Ranges: ", row)
    return row


def fix_abbreviations(ingredient):
    """
    Converts instances of oz. and g. to ounce and gram respectively
    """

    # replace oz. with ounce
    if ingredient == ingredient:
        match = re.findall(r"([0-9])\s*oz\.*", ingredient)
        if match:
            for m in match:
                ingredient = re.sub(r"([0-9])\s*oz\.*", m + " ounce", ingredient)
        # replace g. with gram
        match = re.findall(r"([0-9])\s*g([^a-z])", ingredient)
        if match:
            for m in match:
                ingredient = re.sub(
                    r"([0-9])\s*g([^a-z])", m[0] + " gram" + m[1], ingredient
                )
        # replace tbsp with tablespoon
        match = re.findall(r"[Tt]bsp\.*", ingredient)
        if match:
            for m in match:
                ingredient = re.sub(r"[Tt]bsp\.*", "tablespoon", ingredient)
        # replace tsp with teaspoon
        match = re.findall(r"[Tt]sp\.*", ingredient)
        if match:
            for m in match:
                ingredient = re.sub(r"[Tt]sp\.*", "teaspoon", ingredient)
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
