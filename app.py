import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_folium import folium_static
import folium
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

# Set up the app
st.set_page_config(page_title="University of Alabama, Huntsville: Engagement Analysis", layout="wide")

# App Title
st.title("University of Alabama, Huntsville: Engagement Analysis")

# Section 1: Overall Sentiment Analysis of Classroom Spaces by Buildings
st.header("Overall Sentiment Analysis of Classroom Spaces by Buildings")

@st.cache_data
def load_data(uploaded_file):
    try:
        if uploaded_file is not None:
            return pd.read_csv(uploaded_file, encoding="ISO-8859-1")
        else:
            return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

uploaded_file = st.file_uploader("Upload your dataset (CSV format):", type="csv")
df = load_data(uploaded_file)

if df is not None:
    # Preprocess data
    sia = SentimentIntensityAnalyzer()
    df['Sentiment'] = df['Response'].apply(lambda x: sia.polarity_scores(x)['compound'])
    df['Sentiment_Category'] = df['Sentiment'].apply(
        lambda x: "Positive" if x > 0.2 else "Neutral" if -0.2 <= x <= 0.2 else "Negative"
    )

    # Create map
    st.subheader("")
    m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=15)
    for _, row in df.iterrows():
        color = "green" if row['Sentiment_Category'] == "Positive" else "orange" if row['Sentiment_Category'] == "Neutral" else "red"
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=7,
            popup=f"{row['Building']}<br>Sentiment: {row['Sentiment']:.2f}",
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
        ).add_to(m)

    # Render map and legend
    col1, col2 = st.columns([3, 1])
    with col1:
        folium_static(m)
    with col2:
        st.markdown(
            """
            **Legend**  
            🟢 Positive (> 0.2)  
            🟠 Neutral (-0.2 to 0.2)  
            🔴 Negative (< -0.2)  
            **Total Responses:** {}
            """.format(len(df))
        )

    # Section 2: Explore Themes
    st.header("Explore Emerging Themes and Responses")
    themes = ["Spacious", "Lighting", "Comfort", "Accessibility", "Collaborative"]

    selected_theme = st.radio("Select a Theme to Explore:", themes)
    if selected_theme:
        theme_data = df[df['Response'].str.contains(selected_theme, case=False, na=False)]
        st.subheader(f"Buildings Mentioning '{selected_theme}'")
        theme_summary = theme_data.groupby('Building').agg(
            Sentiment_Score=('Sentiment', 'mean'),
            Total_Responses=('Response', 'count')
        ).reset_index()

        if not theme_summary.empty:
            col1, col2 = st.columns([1, 2])
            with col1:
                theme_summary['Sentiment_Category'] = theme_summary['Sentiment_Score'].apply(
                    lambda x: "Positive" if x > 0.2 else "Neutral" if -0.2 <= x <= 0.2 else "Negative"
                )
                st.table(theme_summary[['Building', 'Sentiment_Category', 'Total_Responses']])
            with col2:
                st.write("Key Responses:")
                for _, row in theme_data.iterrows():
                    color = "🟢" if row['Sentiment'] > 0.2 else "🟠" if -0.2 <= row['Sentiment'] <= 0.2 else "🔴"
                    st.markdown(f"{color} {row['Response']} ({row['Building']})")

    # Section 3: Sentiment Classification by Buildings
    st.header("Sentiment Classification by Buildings")
    st.subheader("Building Sentiment Treemap")
    building_summary = df.groupby('Building').agg(
        Avg_Sentiment=('Sentiment', 'mean'),
        Total_Responses=('Response', 'count')
    ).reset_index()

    fig = px.treemap(
        building_summary,
        path=['Building'],
        values='Total_Responses',
        color='Avg_Sentiment',
        color_continuous_scale='RdYlGn',
        title="Building Sentiment Treemap"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Detailed Analysis by Building
    selected_building = st.selectbox("Select a Building for Details:", building_summary['Building'])
    if selected_building:
        st.subheader(f"Details for {selected_building}")
        building_data = df[df['Building'] == selected_building]
        st.write(f"**Average Sentiment Score:** {building_data['Sentiment'].mean():.2f}")
        st.write(f"**Total Responses:** {len(building_data)}")
        st.write("**Key Responses:**")
        for _, row in building_data.iterrows():
            color = "🟢" if row['Sentiment'] > 0.2 else "🟠" if -0.2 <= row['Sentiment'] <= 0.2 else "🔴"
            st.markdown(f"{color} {row['Response']}")

else:
    st.warning("Please upload a dataset to proceed.")
