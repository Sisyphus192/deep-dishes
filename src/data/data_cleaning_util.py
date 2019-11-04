import re
from fractions import Fraction
import unicodedata


def clean_html(s):
    """
    This will replace all html tags that were not stripped
    from the NYT data
    """
    columns = ["input", "name", "comment"]
    for col in columns:
        # This filters out NaN values so they wont get caught in the try except
        if col in s.keys() and s[col] == s[col]:
            try:
                # this will remove all: <a href=...>see recipe</a>
                match = re.findall(r"\s*\(?<.*see\s*recipe.*>\)?", s[col])
                if match:
                    for m in match:
                        s[col] = re.sub(r"\s*\(?<.*see\s*recipe.*>\)?", "", s[col])
                        if col == "input" and s["comment"] == s["comment"]:
                            s["comment"] = re.sub(r"see recipe", "", s["comment"])
                # this will remove all: see <a href=...>recipe</a>
                match = re.findall(r"\(?\s*(see)\s*?<.*recipe.*>\)?", s[col])
                if match:
                    for m in match:
                        s[col] = re.sub(r"\(?\s*(see)\s*?<.*recipe.*>\)?", "", s[col])
                        if col == "input" and s["comment"] == s["comment"]:
                            s["comment"] = re.sub(r"see recipe", "", s["comment"])
                # this will remove all: epi:recipelink stuff
                match = re.findall(r"\<?\/?epi:recipelink\>?", s[col]) 
                if match:
                    for m in match:
                        s[col] = re.sub(r"\<?\/?epi:recipelink\>?", "", s)
                # This will remove all <span> and misc <a href=...>...</a>
                match = re.findall(r"<.*?>", s[col])
                if match:
                    for m in match:
                        s[col] = re.sub(r"<.*?>", "", s[col])
                # this will remove all un-escapped '\n' from the original html
                match = re.findall(r"\s*\\n\s*", s[col])
                if match:
                    for m in match:
                        s[col] = re.sub(r"\\n", " ", s[col])
                # if the column is now blank becasue of what we removed, set it
                # to NaN so pandas can handle it easier
                if not s[col]:
                    s[col] = float("nan")
                else:
                    s[col] = s[col].strip()

            except TypeError:
                print("ERROR CLEANING HTML: " + col + ": ", s)
    return s


def clean_unicode_fractions(s):
    """
    Replace unicode fractions with ascii representation, preceded by a
    space.

    "1\x215e" => "1 7/8"
    """
    try:
        # match all mixed fractions with a unicode fraction (e.g. 1 ¾ or 1¾) and add them together
        match = re.findall(r"(\d+)\s?([\u2150-\u215E\u00BC-\u00BE])", s)
        if match:
            for m in match:
                num = float(m[0]) + float(Fraction(unicodedata.numeric(m[1])))
                s = re.sub(
                    r"(\d+)\s?([\u2150-\u215E\u00BC-\u00BE])", str(round(num, 3)), s
                )

        # match all other unicode fractions
        match = re.findall(r"([\u2150-\u215E\u00BC-\u00BE])", s)
        if match:
            for m in match:
                s = re.sub(
                    r"([\u2150-\u215E\u00BC-\u00BE])",
                    str(round(float(Fraction(unicodedata.numeric(m))), 3)),
                    s,
                )
    except TypeError:
        print("ERROR CLEANING UNICODE: ", s)
    return s


def merge_ranges(s):
    """
    Many ingredients are written "1 2-2 1/2 pound" this represents
    an acceptable quantity range of 2 to 2.5. Because this will
    make parseing harder we will replace the range with the average.
    """
    try:
        match = re.findall(r"\d+\s\d+\-\d+\s*\d*\/*\d*", s["input"])

    except TypeError:
        print("error parsing: ", s)


def merge_fractions(s):
    """
    Merges mixed fractions: 1 2/3 => 1.67
    """
    match = re.findall(r"(\d+)\s+(\d\/\d)", s)
    if match:
        for m in match:
            num = float(m[0]) + float(Fraction(m[1]))
            s = re.sub(r"(\d+)\s+(\d\/\d)", str(round(num, 3)), s)

    match = re.findall(r"(\d\/\d)", s)
    if match:
        for m in match:
            num = float(Fraction(m))
            s = re.sub(r"(\d\/\d)", str(round(num, 3)), s)
    return s


def fix_abbreviations(s):
    """
    Converts instances of oz. and g. to ounce and gram respectively
    """
    columns = ["input", "unit"]
    for col in columns:
        # replace oz. with ounce
        if s[col] == s[col]:
            match = re.findall(r"([0-9])\s*oz\.*", s[col])
            if match:
                for m in match:
                    s[col] = re.sub(r"([0-9])\s*oz\.*", m + " ounce", s[col])
            # replace g. with gram
            match = re.findall(r"([0-9])\s*g([^a-z])", s[col])
            if match:
                for m in match:
                    s[col] = re.sub(
                        r"([0-9])\s*g([^a-z])", m[0] + " gram" + m[1], s[col]
                    )
            # replace tbsp with tablespoon
            match = re.findall(r"[Tt]bsp\.*", s[col])
            if match:
                for m in match:
                    s[col] = re.sub(r"[Tt]bsp\.*", "tablespoon", s[col])
            # replace tsp with teaspoon
            match = re.findall(r"[Tt]sp\.*", s[col])
            if match:
                for m in match:
                    s[col] = re.sub(r"[Tt]sp\.*", "teaspoon", s[col])
    return s
