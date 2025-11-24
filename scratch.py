from collections import Counter
import matplotlib.pyplot as plt

# Make sure to install the package once:
# pip install wordcloud

from wordcloud import WordCloud

def read_book(path):
    with open(path, mode="r", encoding="utf-8") as book_file:
        book_lines = book_file.readlines()
    clean_lines = [line.strip().lower() for line in book_lines]
    
    # Split each line into words and flatten the list
    words = [word for line in clean_lines for word in line.split() if word]
    return words

def wordcloud_plot(tokens, title="Word Cloud", width=800, height=400, background_color="white"):
    """
    Draws a word cloud from a list of tokens.
    - tokens: list of strings
    - width, height: size of the generated image
    - background_color: 'white', 'black', etc.
    """
    if not tokens:
        print("No data.")
        return

    c = Counter(tokens)
    freqs = dict(c)

    wc = WordCloud(width=width, height=height, background_color=background_color)
    wc = wc.generate_from_frequencies(freqs)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    print("Showing plot!")
    plt.show()

en_tokens = read_book("data/english.txt")
wordcloud_plot(en_tokens)

while True:
    x = 3