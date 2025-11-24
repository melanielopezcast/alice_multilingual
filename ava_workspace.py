import spacy
from nltk.corpus import wordnet as wn
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from spacy.matcher import PhraseMatcher
from spacy_wordnet.wordnet_annotator import WordnetAnnotator

nlp_en = spacy.load("en_core_web_sm")
nlp_fr = spacy.load("fr_core_news_sm")
nlp_es = spacy.load("es_core_news_sm")
nlp_it = spacy.load("it_core_news_sm")

with open("data/english.txt", "r", encoding="utf-8") as f:
    text_en_lines = f.readlines()
with open("data/english.txt", "r", encoding="utf-8") as f:
    text_en = f.read()

with open("data/french.txt", "r", encoding="utf-8") as f:
    text_fr_lines = f.readlines()
with open("data/french.txt", "r", encoding="utf-8") as f:
    text_fr = f.read()

with open("data/spanish.txt", "r", encoding="utf-8") as f:
    text_es_lines = f.readlines()
with open("data/spanish.txt", "r", encoding="utf-8") as f:
    text_es = f.read()

with open("data/italian.txt", "r", encoding="utf-8") as f:
    text_it_lines = f.readlines()
with open("data/italian.txt", "r", encoding="utf-8") as f:
    text_it = f.read()

def ensure_tokens_unique(tokens):
    unique_tokens = []
    for token in tokens:
        if token.lower() in unique_tokens:
            continue
        unique_tokens.append(token.lower())
    return unique_tokens

def ensure_unique_lemmas(lemmas):
    counter = Counter(lemmas)
    data = [(l, counter[l]) for l in lemmas]
    data_unique = []
    unique_lemmas = []
    for item in data:
        if item[0].lower() in unique_lemmas:
            continue
        unique_lemmas.append(item[0].lower())
        data_unique.append(item)
    return data_unique

def info(lang, text, nlp, text_lines):
    doc = nlp(text)
    doc = [t for t in doc
        if (not (t.is_punct or t.is_space or t.like_num or t.is_stop)) and t.is_alpha]
    tokens = [t.text for t in doc]
    lemmas = [t.lemma_.lower() for t in doc]
    unique_lemmas = ensure_unique_lemmas(lemmas)
    unique_tokens = ensure_tokens_unique(tokens)
    unique_tokens_count = len(unique_tokens)
    total_tokens = len([t.text for t in doc])
    unique_lemmas_count = len(unique_lemmas)
    lexical_diversity = unique_tokens_count/float(total_tokens)
    num_sentences = len(text_lines)
    return [total_tokens, unique_lemmas_count, lexical_diversity, num_sentences, unique_lemmas]

df_dict = {}
en_info = info("English", text_en, nlp_en, text_en_lines)
df_dict["English"] = [en_info[0], en_info[1], en_info[2], en_info[3]]

fr_info = info("French", text_fr, nlp_fr, text_fr_lines)
df_dict["French"] = [fr_info[0], fr_info[1], fr_info[2], fr_info[3]]

es_info = info("Spanish", text_es, nlp_es, text_es_lines)
df_dict["Spanish"] = [es_info[0], es_info[1], es_info[2], es_info[3]]

it_info = info("Italian", text_it, nlp_it, text_it_lines)
df_dict["Italian"] = [it_info[0], it_info[1], it_info[2], it_info[3]]

df = pd.DataFrame.from_dict(df_dict, orient="index", columns=["Total Tokens", "Unique Lemmas", "Lexical Diversity", "Number of Sentences"])
print(df)

# part two

en_lemma_counts = en_info[4]
fr_lemma_counts = fr_info[4]
es_lemma_counts = es_info[4]
it_lemma_counts = it_info[4]

en_lemma_counts_sort = sorted(en_lemma_counts, key=lambda l: l[1], reverse=True)
fr_lemma_counts_sort = sorted(fr_lemma_counts, key=lambda l: l[1], reverse=True)
es_lemma_counts_sort = sorted(es_lemma_counts, key=lambda l: l[1], reverse=True)
it_lemma_counts_sort = sorted(it_lemma_counts, key=lambda l: l[1], reverse=True)

print(en_lemma_counts_sort[:20])
print(fr_lemma_counts_sort[:20])
print(es_lemma_counts_sort[:20])
print(it_lemma_counts_sort[:20])