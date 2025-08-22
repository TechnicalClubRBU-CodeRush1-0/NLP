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

#nltk.download('punkt_tab')
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

# Function to generate word cloud
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white', 
                          stopwords=set(stopwords.words('english')), 
                          max_words=200, colormap='viridis').generate(text)
    return wordcloud

# Function to plot word frequency
def plot_word_frequency(text):
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    word_counts = Counter(tokens)
    most_common_words = word_counts.most_common(10)
    words, counts = zip(*most_common_words)

    plt.figure(figsize=(10, 5))
    plt.bar(words, counts, color='skyblue')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Top 10 Most Common Words')
    plt.xticks(rotation=45)
    st.pyplot(plt)

# Function for sentiment analysis

# def SentimentAnalysis(text):
#     sentiment_analyzer = SentimentIntensityAnalyzer()
#     sentiment_score = sentiment_analyzer.polarity_scores(text)
#     return sentiment_score
    

#testing  for sentiment analysis 
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

def SentimentAnalysis(text):
    # Preprocess the input text
    cleaned_text = preprocess_text(text)
    sentiment_analyzer = SentimentIntensityAnalyzer()
    sentiment_score = sentiment_analyzer.polarity_scores(cleaned_text)
    return sentiment_score

# Streamlit UI
st.title("Text Summarization and Analysis ")

# File upload or text input
uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
text_input = st.text_area("Or enter text directly:")

if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")
elif text_input:
    text = text_input
else:
    st.warning("Please upload a file or enter text.")
    st.stop()

# Sidebar for controls
st.sidebar.header("Options")


# Number of lines for summary
num_lines = st.sidebar.number_input("Number of lines for summary:", min_value=1, max_value=20, value=3)


# Buttons in the sidebar
if st.sidebar.button("Summarize"):
    summary = summarize_text(text, num_lines)
    st.subheader("Summary:")
    st.write(summary)


if st.sidebar.button("Generate Word Cloud"):
    wordcloud = generate_wordcloud(text)
    st.subheader("Word Cloud:")
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)


if st.sidebar.button("Show Word Frequency Graph"):
    st.subheader("Word Frequency Graph:")
    plot_word_frequency(text)


if st.sidebar.button("Analyze Sentiment"):
    sentiment_score = SentimentAnalysis(text)
    st.subheader("Sentiment Analysis:")
    st.write("Sentiment Score:", sentiment_score)

    a=sentiment_score['pos']
    b=sentiment_score['neg']
    c=sentiment_score['neu']

    if(a>b and c<0.75):
        st.write("The text is Positive")
    elif(b>a and c<0.75):
        st.write("The text is Negative")
    else:
        st.write("The text is Neutral")  
    
    st.write("Compound Score:", sentiment_score['compound'])

    #st.write("Positive:", sentiment_score['pos'])
    #st.write("Negative:", sentiment_score['neg'])
    #st.write("Neutral:", sentiment_score['neu'])

    
