import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Page Configuration
st.set_page_config(page_title="University of Alabama, Huntsville: Engagement Analysis", layout="wide")

# Title
st.title("University of Alabama, Huntsville: Engagement Analysis")
st.markdown("Upload your dataset (CSV format)")

# File Upload
uploaded_file = st.file_uploader("Drag and drop file here", type="csv")

@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    analyzer = SentimentIntensityAnalyzer()
    data["Sentiment"] = data["Tell us about your classroom"].apply(
        lambda x: analyzer.polarity_scores(x)["compound"] if pd.notnull(x) else 0
    )
    data["Sentiment_Category"] = data["Sentiment"].apply(
        lambda x: "Positive" if x > 0.2 else "Neutral" if -0.2 <= x <= 0.2 else "Negative"
    )
    return data

if uploaded_file:
    # Load data
    df = load_data(uploaded_file)

    # Section 1: Overall Sentiment Analysis
    st.header("Overall Sentiment Analysis of Classroom Spaces by Buildings")
    map_center = [df["Latitude"].mean(), df["Longitude"].mean()]

    sentiment_map = folium.Map(location=map_center, zoom_start=16)

    for _, row in df.iterrows():
        color = "green" if row["Sentiment"] > 0.2 else "orange" if row["Sentiment"] >= -0.2 else "red"
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=8,
            color=color,
            fill=True,
            fill_color=color,
            popup=f"<b>Building:</b> {row['Buildings Name']}<br><b>Sentiment:</b> {row['Sentiment']:.2f}<br><b>Response:</b> {row['Tell us about your classroom']}",
        ).add_to(sentiment_map)

    st_folium(sentiment_map, width=800)

    # Legend
    st.markdown(
        """
        <div style="margin-top: -20px; margin-left: 20px; font-size: 14px;">
            <b>Legend:</b><br>
            <span style="color: green;">● Positive (&gt; 0.2)</span><br>
            <span style="color: orange;">● Neutral (-0.2 to 0.2)</span><br>
            <span style="color: red;">● Negative (&lt; -0.2)</span><br>
            Total Responses: <b>{}</b>
        </div>
        """.format(len(df)),
        unsafe_allow_html=True,
    )

    # Section 2: Explore Themes and Responses
    st.header("Explore Emerging Themes and Responses")

    themes = ["Spacious", "Lighting", "Comfort", "Accessibility", "Collaborative"]
    selected_theme = st.radio("Select a Theme to Explore:", themes)

    if selected_theme:
        # Filter data for the selected theme
        filtered_data = df[df["Tell us about your classroom"].str.contains(selected_theme, na=False, case=False)]

        st.subheader(f"Buildings Mentioning '{selected_theme}'")
        building_summary = filtered_data.groupby("Buildings Name")["Sentiment_Category"].value_counts().unstack(fill_value=0)
        st.dataframe(building_summary)

        st.subheader(f"Key Responses for '{selected_theme}'")
        responses = filtered_data[["Buildings Name", "Sentiment", "Tell us about your classroom"]].head(5)
        for _, row in responses.iterrows():
            sentiment_color = "green" if row["Sentiment"] > 0.2 else "orange" if row["Sentiment"] >= -0.2 else "red"
            st.markdown(
                f"""
                <div style="color: {sentiment_color};">
                    <b>{row['Buildings Name']}:</b> "{row['Tell us about your classroom']}"
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Section 3: Sentiment Classification by Buildings
    st.header("Sentiment Classification by Buildings")

    buildings = df["Buildings Name"].unique()
    selected_building = st.selectbox("Select a Building for Details:", buildings)

    if selected_building:
        building_data = df[df["Buildings Name"] == selected_building]
        avg_sentiment = building_data["Sentiment"].mean()
        total_responses = len(building_data)

        st.subheader(f"Details for {selected_building}")
        st.write(f"**Average Sentiment Score:** {avg_sentiment:.2f}")
        st.write(f"**Total Responses:** {total_responses}")

        st.subheader("Key Responses")
        responses = building_data[["Sentiment", "Tell us about your classroom"]].head(5)
        for _, row in responses.iterrows():
            sentiment_color = "green" if row["Sentiment"] > 0.2 else "orange" if row["Sentiment"] >= -0.2 else "red"
            st.markdown(
                f"""
                <div style="color: {sentiment_color};">
                    "{row['Tell us about your classroom']}"
                </div>
                """,
                unsafe_allow_html=True,
            )
else:
    st.warning("Please upload a CSV file to continue.")
