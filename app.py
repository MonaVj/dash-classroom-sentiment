import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import folium_static
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# Page Configuration
st.set_page_config(page_title="University of Alabama, Huntsville Engagement Analysis", layout="wide")
st.title("University of Alabama, Huntsville Engagement Analysis Using Machine Learning")

# Define themes and associated keywords
themes_dict = {
    "Spacious": ["spacious", "large", "roomy", "open", "airy"],
    "Lighting": ["bright", "natural light", "well-lit", "sunny"],
    "Comfort": ["comfortable", "cozy", "relaxed", "ergonomic"],
    "Accessibility": ["accessible", "inclusive", "easy access", "barrier-free"],
    "Collaborative": ["collaborative", "interactive", "teamwork", "group"]
}

# File Upload
uploaded_file = st.file_uploader("Upload Classroom Data CSV", type=["csv"])

if uploaded_file:
    @st.cache_data
    def load_data(file):
        df = pd.read_csv(file, encoding="ISO-8859-1")
        df["Corrected Response"] = df["Tell us about your classroom"].str.strip().str.lower()
        df["Sentiment Score"] = df["Corrected Response"].apply(
            lambda x: sia.polarity_scores(x)["compound"] if isinstance(x, str) else 0
        )
        return df

    df = load_data(uploaded_file)

    def assign_themes(response):
        matched_themes = []
        for theme, keywords in themes_dict.items():
            if any(keyword in response for keyword in keywords):
                matched_themes.append(theme)
        return ", ".join(matched_themes) if matched_themes else None

    df["Themes"] = df["Corrected Response"].apply(lambda x: assign_themes(x))

    building_data = df.groupby("Buildings Name").agg(
        Avg_Sentiment=("Sentiment Score", "mean"),
        Count=("Corrected Response", "count"),
        Themes=("Themes", lambda x: ", ".join(set(", ".join(x.dropna()).split(", "))))
    ).reset_index()

    # Section 1: Map
    st.subheader("Section 1: Overall Sentiment Map")
    map_center = [df["Latitude"].mean(), df["Longitude"].mean()]
    folium_map = folium.Map(location=map_center, zoom_start=15)

    for _, row in building_data.iterrows():
        sentiment_color = (
            "green" if row["Avg_Sentiment"] > 0.2
            else "red" if row["Avg_Sentiment"] < -0.2
            else "orange"
        )
        popup_content = f"""
        <strong>{row['Buildings Name']}</strong><br>
        Average Sentiment: {row['Avg_Sentiment']:.2f}<br>
        Responses: {row['Count']}<br>
        Themes: {row['Themes']}
        """
        folium.CircleMarker(
            location=[
                df.loc[df["Buildings Name"] == row["Buildings Name"], "Latitude"].mean(),
                df.loc[df["Buildings Name"] == row["Buildings Name"], "Longitude"].mean(),
            ],
            radius=10,
            color=sentiment_color,
            fill=True,
            fill_color=sentiment_color,
            popup=folium.Popup(popup_content, max_width=250),
        ).add_to(folium_map)

    col1, col2 = st.columns([4, 1])
    with col1:
        folium_static(folium_map)
    with col2:
        st.markdown("**Legend**")
        st.markdown("🟢 Positive (> 0.2)")
        st.markdown("🟠 Neutral (-0.2 to 0.2)")
        st.markdown("🔴 Negative (< -0.2)")
        st.markdown(f"**Total Responses:** {len(df)}")

    # Section 2: Analyze by Themes
    st.subheader("Section 2: Analyze by Themes")
    selected_theme = st.radio("Select a Theme", list(themes_dict.keys()))

    if selected_theme:
        theme_data = df[df["Themes"].str.contains(selected_theme, na=False)]
        st.markdown(f"### Buildings Mentioning '{selected_theme}'")
        for _, row in building_data.iterrows():
            if selected_theme in row["Themes"]:
                sentiment_color = "🟢" if row["Avg_Sentiment"] > 0.2 else "🔴" if row["Avg_Sentiment"] < -0.2 else "🟠"
                st.write(f"{sentiment_color} {row['Buildings Name']}")

        st.markdown(f"### Key Responses for '{selected_theme}'")
        for _, response in theme_data.iterrows():
            sentiment_dot = "🟢" if response["Sentiment Score"] > 0.2 else "🔴" if response["Sentiment Score"] < -0.2 else "🟠"
            st.write(f"{sentiment_dot} \"{response['Corrected Response']}\"")

    # Section 3: Sentiment Treemap
    st.subheader("Section 3: Sentiment Treemap and Building Details")
    fig = px.treemap(
        building_data,
        path=["Buildings Name"],
        values="Count",
        color="Avg_Sentiment",
        color_continuous_scale="RdYlGn",
        title="Building Sentiment Treemap"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Dropdown for Building Details
    selected_building = st.selectbox("Select a Building for Details:", building_data["Buildings Name"])

    if selected_building:
        building_details = building_data[building_data["Buildings Name"] == selected_building]
        st.markdown(f"### Details for {selected_building}")
        st.write(f"**Average Sentiment Score:** {building_details['Avg_Sentiment'].values[0]:.2f}")
        st.write(f"**Total Responses:** {building_details['Count'].values[0]}")

        responses = df[df["Buildings Name"] == selected_building]["Corrected Response"].tolist()
        for response in responses[:5]:
            sentiment_dot = "🟢" if sia.polarity_scores(response)["compound"] > 0.2 else "🔴" if sia.polarity_scores(response)["compound"] < -0.2 else "🟠"
            st.write(f"{sentiment_dot} \"{response}\"")
