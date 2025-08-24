import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline

# ==============================
# Load Hugging Face Models (free)
# ==============================
sentiment_pipeline = pipeline("sentiment-analysis")

# ==============================
# Sentiment Analysis Function
# ==============================
def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]
    label = result['label']
    score = result['score']

    explanation = {
        "POSITIVE": "The text expresses a positive opinion, likely showing satisfaction or enthusiasm.",
        "NEGATIVE": "The text expresses a negative opinion, likely showing dissatisfaction or frustration.",
        "NEUTRAL": "The text seems neutral, with no strong emotional leaning."
    }

    return f"Sentiment: {label}\nConfidence: {score:.2f}\nExplanation: {explanation.get(label, 'General sentiment detected.')}"

# ==============================
# Lead Scoring Function (rule-based, free)
# ==============================
def lead_scoring(name, email, company, engagement_level, purchase_intent):
    score = 0

    if engagement_level == "High":
        score += 40
    elif engagement_level == "Medium":
        score += 25
    else:
        score += 10

    if purchase_intent == "High":
        score += 50
    elif purchase_intent == "Medium":
        score += 30
    else:
        score += 10

    reasoning = f"""
    Lead {name} from {company} shows {engagement_level} engagement
    and {purchase_intent} purchase intent.
    Final lead score: {score}/100.
    """

    return reasoning

# ==============================
# Streamlit App
# ==============================
st.title("🤖 AI-Driven Sentiment & Lead Scoring App (Free & Local)")

# ------------------------------
# Sentiment Analysis Section
# ------------------------------
st.header("Sentiment Analysis")
st.write("Enter a sentence below to analyze its sentiment:")

text_input = st.text_area("Input Text")

if st.button("Analyze Sentiment"):
    if text_input.strip():
        with st.spinner("Analyzing..."):
            sentiment_result = analyze_sentiment(text_input)
        st.success("Analysis Complete!")
        st.write("### Sentiment Analysis Result:")
        st.write(sentiment_result)
    else:
        st.warning("⚠️ Please enter some text before analyzing.")

# ------------------------------
# Bulk Sentiment Analysis from CSV
# ------------------------------
st.header("📂 Bulk Sentiment Analysis from CSV")
file = st.file_uploader("Upload a CSV file", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    # Detect text column automatically
    text_column = None
    for col in df.columns:
        if "text" in col.lower():
            text_column = col
            break

    if text_column:
        st.write("### Preview of Uploaded Data:")
        st.write(df.head())

        if st.button("Analyze Sentiments in CSV"):
            with st.spinner("Analyzing sentiments..."):
                df_subset = df.head(5)  # Limit to 5 rows for demo
                df_subset["Sentiment Analysis"] = df_subset[text_column].apply(analyze_sentiment)

                # Extract main sentiment (POS/NEG/NEUTRAL)
                df_subset["Sentiment"] = df_subset["Sentiment Analysis"].apply(
                    lambda x: "Positive" if "POSITIVE" in x else "Negative" if "NEGATIVE" in x else "Neutral"
                )

            st.success("Sentiment Analysis Complete!")
            st.write(df_subset)

            # Plot sentiment distribution
            st.write("### Sentiment Distribution")
            fig, ax = plt.subplots()
            sns.countplot(x="Sentiment", data=df_subset, palette="viridis", ax=ax)
            st.pyplot(fig)
    else:
        st.error("CSV must contain a column with 'text' in its name!")

# ------------------------------
# Lead Scoring Section
# ------------------------------
st.header("🎯 AI-Driven Lead Scoring")
st.write("Enter lead details to get a score:")

name = st.text_input("Name")
email = st.text_input("Email")
company = st.text_input("Company")
engagement_level = st.selectbox("Engagement Level", ["Low", "Medium", "High"])
purchase_intent = st.selectbox("Purchase Intent", ["Low", "Medium", "High"])

if st.button("Get Lead Score"):
    if name and email and company:
        with st.spinner("Scoring lead..."):
            lead_score = lead_scoring(name, email, company, engagement_level, purchase_intent)
        st.success("Lead Scoring Complete!")
        st.write("### Lead Score & Insights:")
        st.write(lead_score)
    else:
        st.warning("⚠️ Please fill all fields before scoring.")
