import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download necessary NLTK data
nltk.download('vader_lexicon')

# Set up the page configuration
st.set_page_config(page_title="University of Alabama, Huntsville: Engagement Analysis", layout="wide")

# App Title
st.title("University of Alabama, Huntsville: Engagement Analysis")

# File Upload Section
st.subheader("Upload your dataset (CSV format)")
uploaded_file = st.file_uploader("Drag and drop file here", type=["csv"])

# Load and Process Data
@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
    # Drop rows with missing latitude/longitude
    df = df.dropna(subset=["Latitude", "Longitude"])
    # Convert latitude and longitude to numeric
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df = df.dropna(subset=["Latitude", "Longitude"])
    return df

@st.cache_data
def analyze_sentiments(data):
    sia = SentimentIntensityAnalyzer()
    data["Sentiment"] = data["Tell us about your classroom"].apply(lambda x: sia.polarity_scores(str(x))["compound"])
    return data

if uploaded_file:
    df = load_data(uploaded_file)
    df = analyze_sentiments(df)

    # Section 1: Overall Sentiment Analysis Map
    st.header("Overall Sentiment Analysis of Classroom Spaces by Buildings")
    
    # Generate the map
    sentiment_map = folium.Map(location=[df["Latitude"].mean(), df["Longitude"].mean()], zoom_start=15)
    for _, row in df.iterrows():
        color = 'green' if row['Sentiment'] > 0.2 else 'orange' if -0.2 <= row['Sentiment'] <= 0.2 else 'red'
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=8,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=f"<b>{row['Buildings Name']}</b><br>Sentiment: {row['Sentiment']:.2f}"
        ).add_to(sentiment_map)

    # Display map and legend
    col1, col2 = st.columns([4, 1])
    with col1:
        folium_static(sentiment_map, width=800, height=600)
    with col2:
        st.markdown("### Legend")
        st.markdown("""
        - 🟢 Positive (Sentiment > 0.2)  
        - 🟠 Neutral (-0.2 ≤ Sentiment ≤ 0.2)  
        - 🔴 Negative (Sentiment < -0.2)  
        **Total Responses:** {0}
        """.format(len(df)))

    # Section 2: Explore Themes
    st.header("Explore Emerging Themes and Responses")
    themes = ["Spacious", "Lighting", "Comfort", "Accessibility", "Collaborative"]
    selected_theme = st.radio("Select a Theme to Explore:", themes)

    # Filter data by selected theme
    theme_responses = df[df["Tell us about your classroom"].str.contains(selected_theme, case=False, na=False)]
    
    # Display buildings and responses
    col3, col4 = st.columns([1, 3])
    with col3:
        st.subheader(f"Buildings Mentioning '{selected_theme}'")
        theme_summary = theme_responses.groupby("Buildings Name").agg({"Sentiment": "mean"}).reset_index()
        theme_summary["Sentiment_Color"] = theme_summary["Sentiment"].apply(
            lambda x: "🟢" if x > 0.2 else "🟠" if -0.2 <= x <= 0.2 else "🔴"
        )
        st.table(theme_summary[["Buildings Name", "Sentiment_Color"]])

    with col4:
        st.subheader(f"Key Responses for '{selected_theme}'")
        responses = theme_responses["Tell us about your classroom"].head(5).tolist()
        for i, response in enumerate(responses):
            sentiment = theme_responses.iloc[i]["Sentiment"]
            color = "🟢" if sentiment > 0.2 else "🟠" if -0.2 <= sentiment <= 0.2 else "🔴"
            st.markdown(f"{color} {response}")

    # Section 3: Sentiment Classification by Buildings
    st.header("Sentiment Classification by Buildings")
    st.subheader("Building Sentiment Treemap")
    selected_building = st.selectbox("Select a Building for Details:", df["Buildings Name"].unique())
    building_data = df[df["Buildings Name"] == selected_building]

    # Display building details
    st.subheader(f"Details for {selected_building}")
    st.write(f"**Average Sentiment Score:** {building_data['Sentiment'].mean():.2f}")
    st.write(f"**Total Responses:** {len(building_data)}")

    # Show responses for the selected building
    st.write("**Key Responses:**")
    for i, response in building_data["Tell us about your classroom"].head(5).items():
        sentiment = building_data.iloc[i]["Sentiment"]
        color = "🟢" if sentiment > 0.2 else "🟠" if -0.2 <= sentiment <= 0.2 else "🔴"
        st.markdown(f"{color} {response}")

else:
    st.warning("Please upload a valid CSV file to continue.")

---

### **requirements.txt**
