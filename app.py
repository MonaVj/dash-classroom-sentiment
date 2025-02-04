import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import plotly.express as px

# Page Configuration
st.set_page_config(
    page_title="University of Alabama, Huntsville: Engagement Analysis",
    layout="wide",
)

# Title
st.markdown("<h1 style='text-align: center;'>University of Alabama, Huntsville: Engagement Analysis</h1>", unsafe_allow_html=True)

# File Upload
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")
if uploaded_file:
    # Load Data
    df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
    
    # Preview Data
    st.write("### Data Preview")
    st.dataframe(df)
    
    # Preprocessing
    # Drop rows with missing values in essential columns
    df.dropna(subset=["Latitude", "Longitude", "Buildings Name", "Avg_Sentiment"], inplace=True)
    df["Avg_Sentiment"] = df["Avg_Sentiment"].fillna(0)
    df["Count"] = df["Count"].fillna(0).astype(int)

    # Section 1: Overall Sentiment Map
    st.markdown("### Overall Sentiment Analysis of Classroom Spaces by Buildings")
    map_center = [df["Latitude"].mean(), df["Longitude"].mean()]
    folium_map = folium.Map(location=map_center, zoom_start=15)

    for _, row in df.iterrows():
        sentiment_color = (
            "green" if row["Avg_Sentiment"] > 0.2 else "red" if row["Avg_Sentiment"] < -0.2 else "orange"
        )
        popup_content = f"""
            <strong>{row['Buildings Name']}</strong><br>
            Average Sentiment: {row['Avg_Sentiment']:.2f}<br>
            Responses: {row['Count']}<br>
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

    # Section 2: Explore Themes and Responses
    st.markdown("### Explore Emerging Themes and Responses")
    themes = ["Spacious", "Lighting", "Comfort", "Accessibility", "Collaborative"]
    selected_theme = st.radio("Select a Theme to Explore:", themes, index=0)

    if selected_theme:
        st.markdown(f"#### Buildings Mentioning '{selected_theme}'")
        theme_filtered = df[df["Themes"].str.contains(selected_theme, case=False, na=False)]
        if not theme_filtered.empty:
            building_summary = theme_filtered.groupby("Buildings Name").agg(
                Avg_Sentiment=("Avg_Sentiment", "mean"),
                Count=("Count", "sum")
            ).reset_index()
            building_summary["Sentiment"] = building_summary["Avg_Sentiment"].apply(
                lambda x: "🟢" if x > 0.2 else "🔴" if x < -0.2 else "🟠"
            )
            st.dataframe(building_summary)
            
            st.markdown(f"#### Key Responses for '{selected_theme}'")
            for _, row in theme_filtered.iterrows():
                st.markdown(f"{row['Sentiment']} {row['Response']}")
        else:
            st.write("No buildings mention this theme.")

    # Section 3: Sentiment Classification by Buildings
    st.markdown("### Sentiment Classification by Buildings")
    building_summary = df.groupby("Buildings Name").agg(
        Avg_Sentiment=("Avg_Sentiment", "mean"),
        Count=("Count", "sum")
    ).reset_index()
    fig = px.treemap(
        building_summary,
        path=["Buildings Name"],
        values="Count",
        color="Avg_Sentiment",
        color_continuous_scale="RdYlGn",
        title="Building Sentiment Treemap",
    )
    st.plotly_chart(fig, use_container_width=True)

    selected_building = st.selectbox("Select a Building for Details:", building_summary["Buildings Name"])
    if selected_building:
        st.markdown(f"#### Details for {selected_building}")
        building_data = building_summary[building_summary["Buildings Name"] == selected_building]
        avg_sentiment = building_data["Avg_Sentiment"].values[0]
        count = building_data["Count"].values[0]
        st.write(f"**Average Sentiment Score:** {avg_sentiment:.2f}")
        st.write(f"**Total Responses:** {count}")
else:
    st.warning("Please upload a CSV file to begin the analysis.")
