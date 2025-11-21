# Manually remove headers and footers. Then, load and clean all language files and tokenize using the appropriate spaCy pipeline for each language.
# Compute and compare basic corpus stats:
# total tokens
# unique lemmas
# lexical diversity (type-token ratio), which measures how varied the vocabulary is
# number of sentences
# Summarize these metrics in a small comparative table.
import spacy
from nltk.corpus import wordnet as wn
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import wordnet as wn
from collections import Counter
from spacy.matcher import PhraseMatcher
from spacy_wordnet.wordnet_annotator import WordnetAnnotator
#nltk.download('wordnet')
#nlp = spacy.load("en_core_web_sm")
#nlp.add_pipe("spacy_wordnet", after="tagger")

with open("english.txt", "r", encoding="utf-8") as f:
  lines = f.readlines()
with open("english.txt", "r", encoding="utf-8") as f:
    text = f.read()

def lemma_table(doc, counter):
    data = [(t.lemma_, t.pos_, counter[t.lemma_.lower()]) for t in doc if t.is_alpha]
    data_unique = []
    seen_lemmas = []
    for item in data:
        if item[0].lower() in seen_lemmas:
            continue
        seen_lemmas.append(item[0].lower())
        data_unique.append(item)
    return pd.DataFrame(data_unique, columns=["lemma", "pos", "count"]).sort_values("count", ascending=False)
    doc = nlp(text)
doc = [t for t in doc
        if not (t.is_stop or t.is_punct or t.is_space or t.like_num)]
tokens = [t.text for t in doc]
lemmas = [t.lemma_ for t in doc]
content_lemmas = [t.lemma_.lower() for t in doc]
freq = Counter(content_lemmas)
print(freq.items())
print(lemma_table(doc, freq))
sentence_df = pd.DataFrame(freq.items(), columns=["lemma", "count"]).sort_values("count", ascending=False)
print(sentence_df.head())