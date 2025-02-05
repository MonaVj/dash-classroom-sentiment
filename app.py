import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import plotly.express as px
from nltk.sentiment import SentimentIntensityAnalyzer

# Page Configuration
st.set_page_config(
    page_title="University of Alabama, Huntsville: Engagement Analysis",
    layout="wide",
)

# Title
st.markdown(
    "<h1 style='text-align: center;'>University of Alabama, Huntsville: Engagement Analysis</h1>",
    unsafe_allow_html=True,
)

# File Upload
if "uploaded_file" not in st.session_state:
    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])
    if uploaded_file:
        st.session_state["uploaded_file"] = uploaded_file
else:
    uploaded_file = st.session_state["uploaded_file"]

if uploaded_file:
    # Remove the upload file option from the interface
    st.markdown("<style>.uploadedFile {display: none;}</style>", unsafe_allow_html=True)

    try:
        # Load Data
        df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")

        # Check for required columns
        required_columns = ["Latitude", "Longitude", "Buildings Name", "Tell us about your classroom"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            st.error(f"The following required columns are missing: {missing_columns}")
        else:
            # Drop rows with missing Latitude or Longitude
            df = df.dropna(subset=["Latitude", "Longitude"])

            # Sentiment Analysis
            sia = SentimentIntensityAnalyzer()
            if "Avg_Sentiment" not in df.columns:
                df["Avg_Sentiment"] = df["Tell us about your classroom"].apply(
                    lambda x: sia.polarity_scores(x)["compound"] if pd.notnull(x) else 0
                )

            if "Count" not in df.columns:
                df["Count"] = 1
            else:
                df["Count"] = df["Count"].fillna(0).astype(int)

            # Aggregate data by building
            building_summary = df.groupby("Buildings Name").agg(
                Avg_Sentiment=("Avg_Sentiment", "mean"),
                Latitude=("Latitude", "mean"),
                Longitude=("Longitude", "mean"),
                Count=("Count", "sum"),
            ).reset_index()

            # Section: Overall Sentiment Analysis Map
            st.markdown(
                "<h2 style='margin-top: 30px;'>Sentiment Analysis of Classroom Spaces by Buildings</h2>",
                unsafe_allow_html=True,
            )
            map_center = [df["Latitude"].mean(), df["Longitude"].mean()]
            folium_map = folium.Map(location=map_center, zoom_start=15, scrollWheelZoom=False)  # Disable scroll zoom

            for _, row in building_summary.iterrows():
                sentiment_color = (
                    "green" if row["Avg_Sentiment"] > 0.2 else "red" if row["Avg_Sentiment"] < -0.2 else "orange"
                )
                popup_content = f"""
                <strong>{row['Buildings Name']}</strong><br>
                Average Sentiment: {row['Avg_Sentiment']:.2f}<br>
                Responses: {row['Count']}
                """
                folium.CircleMarker(
                    location=[row["Latitude"], row["Longitude"]],
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

            # Section 2: Explore Emerging Themes and Responses
st.markdown("<h2 style='margin-top: 30px;'>Explore Emerging Themes and Responses</h2>", unsafe_allow_html=True)

# Predefined keywords for each theme
theme_keywords = {
    "Spacious": ["spacious", "large", "roomy", "open space", "ample room", "big", "not cramped"],
    "Lighting": ["bright", "lighting", "natural light", "well-lit", "light", "dark", "dim"],
    "Comfort": ["comfortable", "seating", "chairs", "desk", "cozy", "comfort"],
    "Accessibility": ["accessible", "ramp", "wheelchair", "disability", "parking", "entry"],
    "Collaborative": ["collaborative", "group", "teamwork", "discussion", "shared space"],
}

# Theme selection
themes = list(theme_keywords.keys())
selected_theme = st.radio("Select a Theme to Explore:", themes, index=0)

# Filter and analyze responses based on selected theme
if selected_theme:
    st.markdown(f"<h3>Buildings Mentioning '{selected_theme}'</h3>", unsafe_allow_html=True)

    # Fetch keywords for the selected theme
    keywords = theme_keywords[selected_theme]

    # Filter responses containing the selected theme's keywords
    theme_data = df[df["Tell us about your classroom"].str.contains('|'.join(keywords), case=False, na=False)]

    # Aggregate sentiment and response data
    grouped_theme_data = theme_data.groupby("Buildings Name").agg({
        "Tell us about your classroom": list,
        "Avg_Sentiment": "mean",
        "Count": "sum"
    }).reset_index()

    # Display buildings mentioning the theme
    grouped_theme_data["Sentiment"] = grouped_theme_data["Avg_Sentiment"].apply(
        lambda x: "🟢" if x > 0.2 else "🟠" if -0.2 <= x <= 0.2 else "🔴"
    )
    grouped_theme_data_display = grouped_theme_data[["Buildings Name", "Sentiment"]]
    st.dataframe(grouped_theme_data_display, use_container_width=True)

    # Display key responses for the theme
    st.markdown(f"<h3>Key Responses for '{selected_theme}'</h3>", unsafe_allow_html=True)
    responses = []
    for _, row in theme_data.iterrows():
        sentiment_icon = (
            "🟢" if row["Avg_Sentiment"] > 0.2 else "🟠" if -0.2 <= row["Avg_Sentiment"] <= 0.2 else "🔴"
        )
        responses.append({
            "response": f"*{row['Tell us about your classroom']} (Building: {row['Buildings Name']})*",
            "sentiment": sentiment_icon,
            "score": row["Avg_Sentiment"]
        })

    # Sort responses by sentiment (Good → Neutral → Bad)
    responses_sorted = sorted(responses, key=lambda x: x["score"], reverse=True)

    # Display up to 5 responses
    for res in responses_sorted[:5]:
        st.markdown(f"{res['sentiment']} {res['response']}")


            # Section: Sentiment Classification by Buildings
            st.markdown(
                "<h2 style='margin-top: 30px;'>Sentiment Classification by Buildings</h2>",
                unsafe_allow_html=True,
            )
            fig = px.treemap(
                building_summary,
                path=["Buildings Name"],
                values="Count",
                color="Avg_Sentiment",
                color_continuous_scale="RdYlGn",
                title="Building Sentiment Treemap",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Building Selection for Details
            selected_building = st.selectbox("Select a Building for Details:", building_summary["Buildings Name"])
            if selected_building:
                st.markdown(f"<h3>Details for {selected_building}</h3>", unsafe_allow_html=True)
                avg_sentiment = building_summary.loc[
                    building_summary["Buildings Name"] == selected_building, "Avg_Sentiment"
                ].values[0]
                count = building_summary.loc[
                    building_summary["Buildings Name"] == selected_building, "Count"
                ].values[0]
                st.write(f"**Average Sentiment Score:** {avg_sentiment:.2f}")
                st.write(f"**Total Responses:** {count}")
                st.markdown("<h4>Key Responses:</h4>", unsafe_allow_html=True)
                building_responses = df[df["Buildings Name"] == selected_building]
                for _, row in building_responses.iterrows():
                    sentiment = (
                        "🟢" if row["Avg_Sentiment"] > 0.2 else "🟠" if row["Avg_Sentiment"] > -0.2 else "🔴"
                    )
                    st.markdown(f"{sentiment} {row['Tell us about your classroom']}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.info("Please upload a CSV file to begin the analysis.")

