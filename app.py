import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px

st.set_page_config(page_title="Sentiment Analyzer", layout="wide")

st.title("😊 Sentiment Analyzer Dashboard")
st.write("Analyze product reviews or tweets and classify them as Positive, Neutral, or Negative.")

# Function to analyze sentiment
def get_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0:
        return "Positive", polarity
    elif polarity < 0:
        return "Negative", polarity
    else:
        return "Neutral", polarity

# Sidebar
st.sidebar.header("Input Options")
option = st.sidebar.radio("Choose Input Type", ["Single Text", "CSV Upload"])

# Single text analysis
if option == "Single Text":
    user_text = st.text_area("Enter your text here:")

    if st.button("Analyze Sentiment"):
        if user_text.strip() != "":
            sentiment, score = get_sentiment(user_text)
            st.subheader("Result")
            st.write(f"**Sentiment:** {sentiment}")
            st.write(f"**Polarity Score:** {score:.2f}")

# CSV upload analysis
elif option == "CSV Upload":
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file, quotechar='"')

        if "review" in df.columns:
            results = df["review"].apply(get_sentiment)
            df["Sentiment"] = results.apply(lambda x: x[0])
            df["Score"] = results.apply(lambda x: x[1])

            st.subheader("Analyzed Data")
            st.dataframe(df)

            # Metrics
            positive_count = (df["Sentiment"] == "Positive").sum()
            negative_count = (df["Sentiment"] == "Negative").sum()
            neutral_count = (df["Sentiment"] == "Neutral").sum()

            col1, col2, col3 = st.columns(3)
            col1.metric("Positive Reviews", positive_count)
            col2.metric("Negative Reviews", negative_count)
            col3.metric("Neutral Reviews", neutral_count)

            # Pie Chart
            sentiment_counts = df["Sentiment"].value_counts().reset_index()
            sentiment_counts.columns = ["Sentiment", "Count"]

            fig = px.pie(sentiment_counts, values="Count", names="Sentiment", hole=0.4,
                         title="Overall Sentiment Distribution")
            st.plotly_chart(fig, use_container_width=True)

            # Bar Chart
            fig2 = px.bar(df, x=df.index, y="Score", color="Sentiment",
                          title="Sentiment Score Per Review")
            st.plotly_chart(fig2, use_container_width=True)

            # Word Cloud
            text = " ".join(df["review"].astype(str))
            wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

            st.subheader("Word Cloud")
            fig_wc, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig_wc)

        else:
            st.error("CSV must contain a column named 'review'")