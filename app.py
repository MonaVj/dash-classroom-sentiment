
# Import necessary libraries
import streamlit as st
import pandas as pd
import plotly.express as px
import nltk
from textblob import TextBlob
import yake

# Initialize NLTK
nltk.download("stopwords")

# Define themes and associated words
THEME_DICT = {
    "Spacious": ["spacious", "large", "open", "airy", "big"],
    "Lighting": ["bright", "dark", "lighting", "sunlight", "dim"],
    "Comfort": ["comfortable", "seats", "warm", "cozy", "cold", "ac", "ventilation"],
    "Accessibility": ["stairs", "wheelchair", "elevator", "distance", "accessible"],
    "Collaborative": ["teamwork", "group", "interactive", "discussion"],
}

# Function to correct grammar and reframe sentence
def correct_sentence(text):
    if isinstance(text, str):
        return str(TextBlob(text).correct())
    return text  # Return original if not a valid string

# Function to extract keywords using YAKE
def extract_keywords(text, num_keywords=5):
    if isinstance(text, str):
        kw_extractor = yake.KeywordExtractor(lan="en", n=1, dedupLim=0.9, top=num_keywords)
        keywords = kw_extractor.extract_keywords(text)
        return ", ".join([kw[0] for kw in keywords])  # Extract keywords only
    return ""

# Function to detect themes
def detect_themes(text):
    if isinstance(text, str):
        detected_themes = [theme for theme, words in THEME_DICT.items() if any(word in text.lower() for word in words)]
        return ", ".join(detected_themes) if detected_themes else "No clear theme"
    return "No clear theme"

# Streamlit UI
def main():
    st.title("Classroom Sentiment Analysis")

    # Upload CSV File
    data_file = st.file_uploader("Upload Classroom Data CSV", type=["csv"])

    if data_file is not None:
        try:
            # Read and preprocess data
           df = pd.read_csv(data_file, encoding="utf-8")

            # Validate required columns
            required_columns = {"Tell us about your classroom", "Latitude", "Longitude", "Buildings Name"}
            if not required_columns.issubset(df.columns):
                st.error("CSV file is missing required columns.")
                st.stop()

            # Data Cleaning
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
                color_continuous_scale=px.colors.diverging.RdYlGn,
                title="Classroom Feedback Map",
                zoom=12,
            )
            map_fig.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 0, "l": 0, "b": 0})
            st.plotly_chart(map_fig)

            # Dropdown for Building Details
            st.header("Building Details")
            selected_building = st.selectbox("Select a Building", df["Buildings Name"].unique())
            if selected_building:
                building_data = df[df["Buildings Name"] == selected_building]
                themes_highlighted = ", ".join(building_data["Themes"].unique())
                corrected_responses = "\n".join(building_data["Corrected Response"].tolist())

                st.subheader(f"Details for {selected_building}")
                st.write(f"**Themes Highlighted:** {themes_highlighted}")
                st.write("**Corrected Responses:**")
                st.text_area("", corrected_responses, height=200)

            # Filter by Theme
            st.header("Filter by Themes")
            theme_selected = st.radio("Select a Theme", list(THEME_DICT.keys()))
            if theme_selected:
                filtered_data = df[df["Themes"].str.contains(theme_selected, na=False)]
                st.write(f"Buildings mentioning '{theme_selected}':")
                st.dataframe(filtered_data[["Buildings Name", "Themes", "Corrected Response"]])

        except Exception as e:
            st.error(f"Error processing file: {e}")

# Run the main function
if __name__ == "__main__":
    main()
