import pandas as pd



dicct = {
    "French": [1, 2, 3, 10],
    "Spanish": [1, 6, 4, 100],
    "Italian": [1, 4, 3, 55],
    "Englishk": [3, 3, 3, 27],
    }

df = pd.DataFrame.from_dict(dicct, orient="index", columns=["Total Tokens", "Unique Lemmas", "Lexical Diversity", "Number of Sentences"])

print(df)
