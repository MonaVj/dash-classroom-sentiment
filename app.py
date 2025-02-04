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
    st.markdown("<style>.uploadedFile {display: none;}</style>", unsafe_allow_html=True)

    try:
        # Load data
        df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")

        # Check for required columns
        required_columns = ["Latitude", "Longitude", "Buildings Name", "Tell us about your classroom"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"The following required columns are missing: {missing_columns}")
        else:
            # Preprocess
            df.dropna(subset=["Latitude", "Longitude"], inplace=True)
            sia = SentimentIntensityAnalyzer()
            if "Avg_Sentiment" not in df.columns:
                df["Avg_Sentiment"] = df["Tell us about your classroom"].apply(
                    lambda x: sia.polarity_scores(x)["compound"] if pd.notnull(x) else 0
                )
            df["Count"] = 1

            # Aggregate by building
            building_summary = df.groupby("Buildings Name").agg(
                Avg_Sentiment=("Avg_Sentiment", "mean"),
                Latitude=("Latitude", "mean"),
                Longitude=("Longitude", "mean"),
                Count=("Count", "sum"),
            ).reset_index()

            # Section 1: Map
            st.markdown(
                "<h2 style='margin-top: 30px;'>Overall Sentiment Analysis of Classroom Spaces by Buildings</h2>",
                unsafe_allow_html=True,
            )
            map_center = [df["Latitude"].mean(), df["Longitude"].mean()]
            folium_map = folium.Map(location=map_center, zoom_start=15, scrollWheelZoom=False)
            for _, row in building_summary.iterrows():
                sentiment_color = (
                    "green" if row["Avg_Sentiment"] > 0.2 else "red" if row["Avg_Sentiment"] < -0.2 else "orange"
                )
                popup_content = f"""
                <strong>{row['Buildings Name']}</strong><br>
                Average Sentiment: {row['Avg_Sentiment']:.2f}<br>
                Total Responses: {row['Count']}
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

            # Section 2: Themes and Responses
            st.markdown("<h2 style='margin-top: 30px;'>Explore Emerging Themes and Responses</h2>", unsafe_allow_html=True)

            theme_keywords = {
                "Spacious": ["spacious", "roomy", "open space", "ample", "not cramped"],
                "Lighting": ["bright", "natural light", "well-lit", "dark", "dim"],
                "Comfort": ["comfortable", "seating", "chairs", "desk", "cozy"],
                "Accessibility": ["accessible", "ramp", "wheelchair", "disability", "parking"],
                "Collaborative": ["collaborative", "group", "discussion", "teamwork"],
            }

            themes = list(theme_keywords.keys())
            selected_theme = st.radio("Select a Theme to Explore:", themes, index=0)

            if selected_theme:
                st.markdown(f"<h3>Buildings Mentioning '{selected_theme}'</h3>", unsafe_allow_html=True)
                keywords = theme_keywords[selected_theme]
                theme_data = df[df["Tell us about your classroom"].str.contains('|'.join(keywords), case=False, na=False)]

                grouped_theme_data = theme_data.groupby("Buildings Name").agg(
                    Avg_Sentiment=("Avg_Sentiment", "mean"),
                    Count=("Count", "sum"),
                ).reset_index()
                grouped_theme_data["Sentiment"] = grouped_theme_data["Avg_Sentiment"].apply(
                    lambda x: "🟢" if x > 0.2 else "🟠" if -0.2 <= x <= 0.2 else "🔴"
                )
                st.dataframe(grouped_theme_data[["Buildings Name", "Sentiment"]], use_container_width=True)

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

                # Balance responses by sentiment type
                positive = [r for r in responses if r["score"] > 0.2]
                neutral = [r for r in responses if -0.2 <= r["score"] <= 0.2]
                negative = [r for r in responses if r["score"] < -0.2]
                balanced_responses = positive[:2] + neutral[:2] + negative[:1]  # Adjust balance here
                balanced_responses_sorted = sorted(balanced_responses, key=lambda x: x["score"], reverse=True)

                for res in balanced_responses_sorted:
                    st.markdown(f"{res['sentiment']} {res['response']}")

            # Section 3: Treemap
            st.markdown("<h2>Sentiment Classification by Buildings</h2>", unsafe_allow_html=True)
            fig = px.treemap(
                building_summary,
                path=["Buildings Name"],
                values="Count",
                color="Avg_Sentiment",
                color_continuous_scale="RdYlGn",
                title="Building Sentiment Treemap",
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.info("Please upload a CSV file to begin the analysis.")
