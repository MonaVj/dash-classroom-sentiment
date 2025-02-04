# Import necessary libraries
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Initialize NLTK
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# App title
st.title("Campus Classroom Sentiment Dashboard")

# File upload
data_file = st.file_uploader("Upload Classroom Data CSV", type=["csv"])

if data_file is not None:
    try:
        # Read the file with fixed encoding
        df = pd.read_csv(data_file, encoding="ISO-8859-1")
        
        # Check for required columns
        required_columns = {"Tell us about your classroom", "Latitude", "Longitude", "Buildings Name"}
        if not required_columns.issubset(df.columns):
            st.error("CSV file is missing one or more required columns.")
            st.stop()

        # Preprocess data
        df = df.drop_duplicates()
        df = df.dropna(subset=["Tell us about your classroom", "Latitude", "Longitude", "Buildings Name"])
        df["Buildings Name"] = df["Buildings Name"].str.strip().str.title()
        df["Tell us about your classroom"] = df["Tell us about your classroom"].str.strip()

        # Grammar correction
        def correct_grammar(text):
            if isinstance(text, str):
                return str(TextBlob(text).correct())
            return text

        df["Corrected Response"] = df["Tell us about your classroom"].apply(correct_grammar)

        # Sentiment analysis
        df["Sentiment Score"] = df["Corrected Response"].apply(
            lambda text: sia.polarity_scores(text)["compound"] if isinstance(text, str) else 0
        )

        # Aggregate sentiment scores at the building level
        building_sentiments = df.groupby("Buildings Name").agg(
            Latitude=("Latitude", "mean"),
            Longitude=("Longitude", "mean"),
            Average_Sentiment=("Sentiment Score", "mean"),
            Count=("Corrected Response", "count")
        ).reset_index()

        # Visualization: Color-Coded Sentiment Map
        st.subheader("Sentiment Map by Building")
        map_center = [building_sentiments["Latitude"].mean(), building_sentiments["Longitude"].mean()]
        m = folium.Map(location=map_center, zoom_start=15)

        # Add building points with color based on sentiment
        for _, row in building_sentiments.iterrows():
            sentiment = row["Average_Sentiment"]
            color = "green" if sentiment > 0.2 else "red" if sentiment < -0.2 else "orange"
            popup_text = f"Building: {row['Buildings Name']}<br>Avg Sentiment: {sentiment:.2f}<br>Responses: {row['Count']}"
            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=8,
                popup=popup_text,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7
            ).add_to(m)

        folium_static(m)

        # Display building sentiment summary
        st.subheader("Building Sentiment Summary")
        st.dataframe(building_sentiments)

        # Detailed Analysis for Selected Building
        st.subheader("Detailed Analysis")
        selected_building = st.selectbox("Select a Building for Analysis:", building_sentiments["Buildings Name"])

        if selected_building:
            building_data = df[df["Buildings Name"] == selected_building]
            st.write(f"### Details for {selected_building}")
            st.write(f"**Average Sentiment Score:** {building_sentiments[building_sentiments['Buildings Name'] == selected_building]['Average_Sentiment'].values[0]:.2f}")
            st.write(f"**Total Responses:** {len(building_data)}")

            st.write("### Corrected Responses:")
            for response in building_data["Corrected Response"].tolist():
                st.write(f"- {response}")

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.warning("Please upload a CSV file to proceed.")
