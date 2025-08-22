#importing libraries
import streamlit as st
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
print(nltk.__version__)
import re

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Function to summarize text
def summarize_text(text, num_sentences):
    # Initialize the LSA summarizer
    lsa = LsaSummarizer(Stemmer("english"))
    lsa.stop_words = get_stop_words("english")
    # Parse the text
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    # Generate the summary
    lsa_summary = lsa(parser.document, num_sentences)
    # Convert the summary to a string
    lsa_summary_list = [str(sentence) for sentence in lsa_summary]
    summary_novel = " ".join(lsa_summary_list)
    return summary_novel 