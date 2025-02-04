# Import necessary libraries
import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob
import yake
import nltk

# Initialize NLTK
nltk.download("stopwords")

# Define themes and associated keywords
THEME_DICT = {
    "Spacious": ["spacious", "large", "open", "airy", "big"],
    "Lighting": ["bright", "dark", "lighting", "sunlight", "dim"],
    "Comfort": ["comfortable", "seats", "warm", "cozy", "cold", "ac", "ventilation"],
    "Accessibility": ["stairs", "wheelchair", "elevator", "distance", "accessible"],
    "Collaborative": ["teamwork", "group", "interactive", "discussion"],
}

# Function to correct grammar and reframe sentences
def correct_sentence(text):
    if isinstance(text, str):
        return str(TextBlob(text).correct())
    return text

# Function to extract keywords using YAKE
def extract_keywords(text, num_keywords=5):
    if isinstance(text, str):
        kw_extractor = yake.KeywordExtractor(lan="en", n=1, dedupLim=0.9, top=num_keywords)
        keywords = kw_extractor.extract_keywords(text)
        return ", ".join([kw[0] for kw in keywords])
    return ""

# Function to detect themes
def detect_themes(text):
    if isinstance(text, str):
        detected_themes = [theme for theme, words in THEME_DICT.items() if any(word in text.lower() for word in words)]
        return ", ".join(detected_themes) if detected_themes else "No clear theme"
    return "No clear theme"

# Main Streamlit app function
def main():
    # App title
    st.title("Classroom Sentiment Analysis")

    # Upload CSV File
    data_file = st.file_uploader("Upload Classroom Data CSV", type=["csv"])

    if data_file is not None:
        try:
            # Attempt to read the file with utf-8, fallback to ISO-8859-1
            try:
                df = pd.read_csv(data_file, encoding="utf-8", errors="ignore")
            except UnicodeDecodeError:
                df = pd.read_csv(data_file, encoding="ISO-8859-1", errors="ignore")

            # Validate required columns
            required_columns = {"Tell us about your classroom", "Latitude", "Longitude", "Buildings Name"}
            if not required_columns.issubset(df.columns):
                st.error("CSV file is missing required columns.")
                st.stop()

            # Preprocess data
            df = df.dropna(subset=["Tell us about your classroom", "Latitude", "Longitude", "Buildings Name"])
            df["Buildings Name"] = df["Buildings Name"].str.strip().str.title()
            df["Tell us about your classroom"] = df["Tell us about your classroom"].str.strip().str.lower()
            df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
            df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

            # Apply transformations
            df["Corrected Response"] = df["Tell us about your classroom"].apply(correct_sentence)
            df["Extracted Keywords"] = df["Tell us about your classroom"].apply(extract_keywords)
            df["Themes"] = df["Tell us about your classroom"].apply(detect_themes)

            # Visualization: Map
            st.header("Sentiment Map")
            map_fig = px.scatter_mapbox(
                df,
                lat="Latitude",
                lon="Longitude",
                hover_name="Buildings Name",
                hover_data={"Themes": True, "Corrected Response": True},
                color_dis
