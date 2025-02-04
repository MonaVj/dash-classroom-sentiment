import streamlit as st
import pandas as pd
import plotly.express as px
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from streamlit_folium import folium_static
import folium

# Download NLTK resources
nltk.download("vader_lexicon")

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Set Streamlit page configuration
st.set_page_config(page_title="University of Alabama, Huntsville Engagement Analysis", layout="wide")
st.title("University of Alabama, Huntsville Engagement Analysis Using Machine Learning")

# Predefined themes and keywords
themes_dict = {
    "Spacious": ["spacious", "large", "roomy", "open", "airy"],
    "Lighting": ["bright", "natural light", "well-lit", "sunny"],
    "Comfort": ["comfortable", "cozy", "relaxed", "ergonomic"],
    "Accessibility": ["accessible", "inclusive", "easy access", "barrier-free"],
    "Collaborative": ["collaborative", "interactive", "teamwork", "group"]
}

# Upload CSV File
uploaded_file = st.file_uploader("Upload Classroom Data CSV", type=["csv"])

if uploaded_file:
    # Load and preprocess data
    @st.cache_data
    def load_data(file):
        df = pd.read_csv(file, encoding="ISO-8859-1")
        df["Corrected Response"] = df["Tell us about your classroom"].str.strip().str.lower()
        df["Sentiment Score"] = df["Corrected Response"].apply(
            lambda x: sia.polarity_scores(x)["compound"] if isinstance(x, str) else 0
        )
        return df

    df = load_data(uploaded_file)

    # Assign themes to responses
    def assign_themes(response):
        matched_themes = []
        for theme, keywords in themes_dict.items():
            if any(keyword in response for keyword in keywords):
                matched_themes.append(theme)
        return ", ".join(matched_themes) if matched_themes else None

    df["Themes"] = df["Corrected Response"].apply(lambda x: assign_themes(x))

    # Aggregate data by buildings
    building_data = df.groupby("Buildings Name").agg(
        Avg_Sentiment=("Sentiment Score", "mean"),
        Count=("Corrected Response", "count"),
        Themes=("Themes", lambda x: ", ".join(set(", ".join(x.dropna()).split(", "))))
    ).reset_index()

    # Map Visualization
    st.subheader("Overall Sentiment Analysis by Building")
    total_responses = df["Corrected Response"].notnull().sum()

    map_center = [df["Latitude"].mean(), df["Longitude"].mean()]
    folium_map = folium.Map(location=map_center, zoom_start=15)

    # Add a custom legend
    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 50px;
        left: 50px;
        width: 300px;
        background-color: white;
        z-index:9999;
        font-size:14px;
        padding:10px;
        border: 2px solid grey;
    ">
    <strong>Legend:</strong><br>
    Positive Sentiment: > 0.2<br>
    Neutral Sentiment: -0.2 to 0.2<br>
    Negative Sentiment: < -0.2<br><br>
    <strong>Total Responses:</strong> {total_responses}
    </div>
    """
    folium_map.get_root().html.add_child(folium.Element(legend_html))

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

    folium_static(folium_map)

    # Treemap Visualization
    st.subheader("Sentiment Treemap (Click a Building to Learn More)")
    treemap_fig = px.treemap(
        building_data,
        path=["Buildings Name"],
        values="Count",
        color="Avg_Sentiment",
        color_continuous_scale="RdYlGn",
        title=""
    )
    treemap_fig.update_layout(margin=dict(t=10, l=10, r=10, b=10), height=500)
    clicked_building = st.plotly_chart(treemap_fig, use_container_width=True)

    # Theme Selection and Analysis
    st.sidebar.title("Select a Theme")
    selected_theme = st.sidebar.radio("Choose a theme:", list(themes_dict.keys()))

    if selected_theme:
        st.subheader(f"Buildings Mentioning '{selected_theme}'")
        theme_data = df[df["Themes"].str.contains(selected_theme, na=False)]
        if not theme_data.empty:
            st.markdown(f"**Buildings Mentioning '{selected_theme}':**")
            theme_buildings = building_data[building_data["Buildings Name"].isin(theme_data["Buildings Name"])]
            for _, row in theme_buildings.iterrows():
                sentiment_color = "🟢" if row["Avg_Sentiment"] > 0.2 else "🔴" if row["Avg_Sentiment"] < -0.2 else "🟠"
                st.markdown(f"{sentiment_color} **{row['Buildings Name']}**")
        else:
            st.warning(f"No responses matched the theme '{selected_theme}'.")

        st.markdown(f"### Key Responses for '{selected_theme}'")
        positive_responses = theme_data[theme_data["Sentiment Score"] > 0.2]["Corrected Response"].tolist()
        neutral_responses = theme_data[(theme_data["Sentiment Score"] <= 0.2) & (theme_data["Sentiment Score"] >= -0.2)]["Corrected Response"].tolist()
        negative_responses = theme_data[theme_data["Sentiment Score"] < -0.2]["Corrected Response"].tolist()

        st.markdown("#### Positive Responses:")
        for response in positive_responses[:3]:
            st.markdown(f"🟢 \"{response}\"")

        st.markdown("#### Neutral Responses:")
        for response in neutral_responses[:3]:
            st.markdown(f"🟠 \"{response}\"")

        st.markdown("#### Negative Responses:")
        for response in negative_responses[:3]:
            st.markdown(f"🔴 \"{response}\"")
else:
    st.info("Upload a CSV file to get started!")
