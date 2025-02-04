# Import necessary libraries
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import plotly.express as px
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Initialize NLTK
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# Grammar correction function
def correct_grammar(text):
    if isinstance(text, str):
        return str(TextBlob(text).correct())
    return text

# Function to get top quotes
def get_top_quotes(data, top_n=3):
    data = data.sort_values(by="Sentiment Score", ascending=False)
    top_quotes = data.head(top_n)[["Corrected Response", "Sentiment Score"]]
    return top_quotes

# Function to add color-coded sentiment dots
def get_sentiment_color(score):
    if score > 0.2:
        return "🟢"  # Positive sentiment
    elif score < -0.2:
        return "🔴"  # Negative sentiment
    else:
        return "🟠"  # Neutral sentiment

# Main Streamlit app function
def main():
    # Set up the page configuration
    st.set_page_config(page_title="Classroom Sentiment Dashboard", layout="wide")

    # Header Section
    st.title("📚 Classroom Sentiment and Theme Dashboard")
    st.markdown("""
        Analyze classroom sentiment and discover insights from your data. Upload your dataset, explore the interactive map, and view trends in building preferences!
    """)

    # File upload
    data_file = st.file_uploader("Upload Classroom Data CSV", type=["csv"])

    if data_file is not None:
        try:
            # Read the file
            df = pd.read_csv(data_file, encoding="ISO-8859-1")

            # Validate required columns
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
            df["Corrected Response"] = df["Tell us about your classroom"].apply(correct_grammar)

            # Sentiment analysis
            df["Sentiment Score"] = df["Corrected Response"].apply(
                lambda text: sia.polarity_scores(text)["compound"] if isinstance(text, str) else 0
            )

            # Aggregate data at the building level
            building_sentiments = df.groupby("Buildings Name").agg(
                Latitude=("Latitude", "mean"),
                Longitude=("Longitude", "mean"),
                Average_Sentiment=("Sentiment Score", "mean"),
                Count=("Corrected Response", "count")
            ).reset_index()

            # Layout with map and treemap
            col1, col2 = st.columns([2, 1])

            # Left column: Sentiment Map
            with col1:
                st.subheader("🗺️ Sentiment Map")
                map_center = [building_sentiments["Latitude"].mean(), building_sentiments["Longitude"].mean()]
                m = folium.Map(location=map_center, zoom_start=15)

                # Add building points with color based on sentiment
                for _, row in building_sentiments.iterrows():
                    sentiment = row["Average_Sentiment"]
                    color = "green" if sentiment > 0.2 else "red" if sentiment < -0.2 else "orange"
                    popup_text = f"""
                    <b>{row['Buildings Name']}</b><br>
                    <b>Avg Sentiment:</b> {row['Average_Sentiment']:.2f} | <b>Responses:</b> {row['Count']}
                    """
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

            # Right column: Treemap
            with col2:
                st.subheader("📊 Sentiment Treemap")
                treemap_fig = px.treemap(
                    building_sentiments,
                    path=["Buildings Name"],
                    values="Count",
                    color="Average_Sentiment",
                    color_continuous_scale=["red", "orange", "green"],
                    title="Building Sentiment Treemap",
                )
                treemap_fig.update_layout(margin=dict(t=25, l=0, r=0, b=0))
                st.plotly_chart(treemap_fig, use_container_width=True)

            # Detailed Analysis
            st.subheader("🏛️ Detailed Building Analysis")
            selected_building = st.selectbox("Select a Building for Analysis:", building_sentiments["Buildings Name"])

            if selected_building:
                building_data = df[df["Buildings Name"] == selected_building]
                st.write(f"### {selected_building}")
                st.write(f"**Average Sentiment Score:** {building_sentiments[building_sentiments['Buildings Name'] == selected_building]['Average_Sentiment'].values[0]:.2f}")
                st.write(f"**Total Responses:** {len(building_data)}")

                # Top Quotes Section
                st.write("### Top Quotes:")
                top_quotes = get_top_quotes(building_data)
                for _, row in top_quotes.iterrows():
                    st.markdown(f"{get_sentiment_color(row['Sentiment Score'])} **{row['Corrected Response']}**")

        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.warning("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()
