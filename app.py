# Install required libraries
!pip install streamlit pandas plotly nltk transformers torch sentencepiece --quiet

# Import necessary libraries
import streamlit as st
import pandas as pd
import plotly.express as px
import nltk
import torch
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# Initialize NLTK
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# Check for GPU availability
device = 0 if torch.cuda.is_available() else -1  # Use GPU if available, else CPU

# Load AI models with error handling and lightweight alternatives
try:
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6", device=device)
    grammar_correction = pipeline("text2text-generation", model="textattack/bert-base-uncased-CoLA", device=device)
except Exception as e:
    st.error(f"Error loading AI models: {e}")
    summarizer = None
    grammar_correction = None

# Write a main function for Streamlit
def main():
    st.title("Classroom Sentiment Analysis")

    # Upload CSV File
    data_file = st.file_uploader("Upload Classroom Data CSV", type=["csv"])

    if data_file is not None:
        try:
            # Read and preprocess data with encoding fallback
            df = pd.read_csv(data_file, encoding="utf-8", errors="ignore")

            # Validate required columns
            required_columns = {"Tell us about your classroom", "Latitude", "Longitude", "Buildings Name"}
            if not required_columns.issubset(df.columns):
                st.error("CSV file is missing required columns.")
                st.stop()

            df = df.dropna(subset=["Tell us about your classroom", "Latitude", "Longitude", "Buildings Name"])
            df["Buildings Name"] = df["Buildings Name"].str.strip().str.title()
            df["Tell us about your classroom"] = df["Tell us about your classroom"].str.strip().str.lower()
            df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
            df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

            # Sentiment Analysis
            df["Sentiment Score"] = df["Tell us about your classroom"].apply(
                lambda x: sia.polarity_scores(x)["compound"] if isinstance(x, str) and x.strip() else 0
            )

            # Theme Extraction
            themes = ["spacious", "lighting", "comfort", "accessibility", "collaborative"]
            df["Themes"] = df["Tell us about your classroom"].apply(
                lambda x: ", ".join([theme for theme in themes if isinstance(x, str) and theme in x]) if isinstance(x, str) else ""
            )

            # AI-based Summarization for each building
            def summarize_building(building):
                comments = df[df["Buildings Name"] == building]["Tell us about your classroom"].tolist()
                text = " ".join(comments)[:400]  # Reduce length to prevent memory issues
                if summarizer and text:
                    try:
                        summary = summarizer(text, max_length=50, min_length=10, do_sample=False)
                        return summary[0]["summary_text"]
                    except Exception:
                        return "No summary available."
                return "No summary available."

            df["Summary"] = df["Buildings Name"].apply(summarize_building)

            # Grammar correction and top quotes
            def correct_grammar(quote):
                if grammar_correction and quote:
                    try:
                        corrected_text = grammar_correction(quote, max_length=100)
                        return corrected_text[0]["generated_text"]
                    except Exception:
                        return "No correction available."
                return "No data available."

            def get_top_quotes(building):
                building_data = df[df["Buildings Name"] == building]
                if not building_data.empty:
                    try:
                        positive_quote = building_data.sort_values(by="Sentiment Score", ascending=False).iloc[0]["Tell us about your classroom"]
                        negative_quote = building_data.sort_values(by="Sentiment Score", ascending=True).iloc[0]["Tell us about your classroom"]
                        return correct_grammar(positive_quote), correct_grammar(negative_quote)
                    except Exception:
                        return "No data available", "No data available"
                return "No data available", "No data available"

            # Visualization: Map
            st.header("Sentiment Map")
            map_fig = px.scatter_mapbox(
                df,
                lat="Latitude",
                lon="Longitude",
                color="Sentiment Score",
                hover_name="Buildings Name",
                hover_data={"Sentiment Score": ":.2f", "Themes": True},
                color_continuous_scale=px.colors.diverging.RdYlGn,
                title="Sentiment Scores by Classroom Location",
                zoom=12,  # Reduce zoom for better visibility
            )
            map_fig.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 0, "l": 0, "b": 0})
            st.plotly_chart(map_fig)

            # Dropdown for Building Details
            st.header("Building Details")
            selected_building = st.selectbox("Select a Building", df["Buildings Name"].unique())
            if selected_building:
                building_data = df[df["Buildings Name"] == selected_building]
                avg_sentiment = building_data["Sentiment Score"].mean()
                total_responses = len(building_data)
                themes_highlighted = ", ".join(building_data["Themes"].unique())
                building_summary = df[df["Buildings Name"] == selected_building]["Summary"].iloc[0]

                st.subheader(f"Details for {selected_building}")
                st.write(f"**Average Sentiment Score:** {avg_sentiment:.2f}")
                st.write(f"**Total Responses:** {total_responses}")
                st.write(f"**Themes Highlighted:** {themes_highlighted}")
                st.write(f"**Building Summary:** {building_summary}")

                # Top Quotes
                positive_quote, negative_quote = get_top_quotes(selected_building)
                st.markdown(f"**Positive:** :green[{positive_quote}]")
                st.markdown(f"**Negative:** :red[{negative_quote}]")

            # Filter by Theme
            st.header("Filter by Themes")
            theme_selected = st.radio("Select a Theme", themes)
            if theme_selected:
                filtered_data = df[df["Themes"].str.contains(theme_selected, na=False)]
                st.write(f"Buildings mentioning '{theme_selected}':")
                st.dataframe(filtered_data[["Buildings Name", "Sentiment Score", "Themes"]])

        except Exception as e:
            st.error(f"Error processing file: {e}")

# Run the main function
if __name__ == "__main__":
    main()
