# Install required libraries if not already installed
# Uncomment the below lines to install
# !pip install streamlit pandas plotly nltk folium streamlit-folium --quiet

# Import necessary libraries
import streamlit as st
import pandas as pd
import plotly.express as px
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
from streamlit_folium import folium_static
import folium

# Download NLTK resources
nltk.download("vader_lexicon")
nltk.download("stopwords")
nltk.download("punkt")

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Set Streamlit page configuration
st.set_page_config(page_title="Classroom Sentiment Analysis", layout="wide")
st.title("Classroom Sentiment Analysis Dashboard")

# Predefined themes and keywords
themes_dict = {
    "Spacious": ["spacious", "large", "roomy", "open", "wide", "airy", "expansive"],
    "Lighting": ["bright", "natural light", "well-lit", "light", "sunny", "vibrant"],
    "Comfort": ["comfortable", "cozy", "relaxed", "ergonomic", "inviting", "soft"],
    "Accessibility": ["accessible", "easy access", "inclusive", "ramps", "barrier-free"],
    "Collaborative": ["collaborative", "teamwork", "interactive", "group work", "shared space"]
}

# Upload CSV File
uploaded_file = st.file_uploader("Upload Classroom Data CSV", type=["csv"])

if uploaded_file:
    # Load the dataset
    df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")

    # Preprocess the dataset
    df["Corrected Response"] = df["Tell us about your classroom"].str.strip().str.lower()
    df["Sentiment Score"] = df["Corrected Response"].apply(
        lambda x: sia.polarity_scores(x)["compound"] if isinstance(x, str) else 0
    )

    # Assign themes to responses
    def assign_themes(response):
        matched_themes = []
        for theme, keywords in themes_dict.items():
            for keyword in keywords:
                if keyword in response:
                    matched_themes.append(theme)
                    break
        return ", ".join(matched_themes) if matched_themes else None

    df["Themes"] = df["Corrected Response"].apply(lambda x: assign_themes(x))

    # Aggregate data for buildings
    building_sentiments = df.groupby("Buildings Name").agg(
        Average_Sentiment=("Sentiment Score", "mean"),
        Count=("Corrected Response", "count"),
        Themes=("Themes", lambda x: ", ".join(set(", ".join(x.dropna()).split(", "))))
    ).reset_index()

    # Create sentiment map
    st.subheader("🌍 Overall Sentiment Analysis of Classrooms By Building")
    map_center = [df["Latitude"].mean(), df["Longitude"].mean()]
    folium_map = folium.Map(location=map_center, zoom_start=15)

    for _, row in building_sentiments.iterrows():
        sentiment_color = (
            "green" if row["Average_Sentiment"] > 0.2
            else "red" if row["Average_Sentiment"] < -0.2
            else "orange"
        )
        popup_content = f"""
        <strong>{row['Buildings Name']}</strong><br>
        Average Sentiment: {row['Average_Sentiment']:.2f}<br>
        Responses: {row['Count']}<br>
        Themes: {row['Themes']}
        """
        folium.CircleMarker(
            location=[df.loc[df["Buildings Name"] == row["Buildings Name"], "Latitude"].mean(),
                      df.loc[df["Buildings Name"] == row["Buildings Name"], "Longitude"].mean()],
            radius=10,
            color=sentiment_color,
            fill=True,
            fill_color=sentiment_color,
            popup=folium.Popup(popup_content, max_width=250)
        ).add_to(folium_map)

    folium_static(folium_map)

    # Theme selection and related buildings
    st.sidebar.subheader("🎯 Key Themes in Responses")
    theme_selection = st.radio("Select a Theme:", list(themes_dict.keys()))

    if theme_selection:
        theme_buildings = df[df["Themes"].str.contains(theme_selection, na=False)]
        if not theme_buildings.empty:
            st.sidebar.write(f"Buildings Mentioning Theme '{theme_selection}':")
            for building in theme_buildings["Buildings Name"].unique():
                avg_sentiment = building_sentiments.loc[
                    building_sentiments["Buildings Name"] == building, "Average_Sentiment"
                ].values[0]
                sentiment_color = (
                    "🟢" if avg_sentiment > 0.2 else "🔴" if avg_sentiment < -0.2 else "🟠"
                )
                st.sidebar.markdown(f"{sentiment_color} {building}")
        else:
            st.sidebar.write(f"No buildings mention the theme '{theme_selection}'.")

    # Sentiment Treemap
    st.subheader("📊 Sentiment Treemap (Click a Building to Learn More)")
    treemap_fig = px.treemap(
        building_sentiments,
        path=["Buildings Name"],
        values="Count",
        color="Average_Sentiment",
        color_continuous_scale="RdYlGn",
        title="Building Sentiment Treemap"
    )
    st.plotly_chart(treemap_fig, use_container_width=True)

    # Key Responses from Students
    clicked_building = st.selectbox(
        "Select a Building for Details:", building_sentiments["Buildings Name"].unique()
    )

    if clicked_building:
        building_data = df[df["Buildings Name"] == clicked_building]
        st.subheader(f"🏛️ Details for {clicked_building}")
        st.write(f"**Average Sentiment Score:** {building_sentiments.loc[building_sentiments['Buildings Name'] == clicked_building, 'Average_Sentiment'].values[0]:.2f}")
        st.write(f"**Total Responses:** {len(building_data)}")

        # Key Responses
        st.write("### Key Responses from Students:")
        positive_responses = building_data[building_data["Sentiment Score"] > 0]["Corrected Response"].tolist()
        negative_responses = building_data[building_data["Sentiment Score"] < 0]["Corrected Response"].tolist()
        neutral_responses = building_data[(building_data["Sentiment Score"] == 0)]["Corrected Response"].tolist()

        # Balance responses
        displayed_responses = positive_responses[:2] + negative_responses[:2] + neutral_responses[:1]
        for response in displayed_responses:
            sentiment = sia.polarity_scores(response)["compound"]
            sentiment_color = "🟢" if sentiment > 0.2 else "🔴" if sentiment < -0.2 else "🟠"
            st.markdown(f"{sentiment_color} {response}")

else:
    st.info("Please upload a CSV file to proceed.")
