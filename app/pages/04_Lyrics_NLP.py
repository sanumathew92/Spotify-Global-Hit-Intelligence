import streamlit as st 
import pandas as pd 
import numpy as np 
import plotly.express as px 

from utils import (
    load_clean_data,
    load_nlp_data,
    format_number
)

# PAGE CONFIGURATION

st.set_page_config(
    page_title="Lyrics & NLP Intelligence",
    page_icon="🎤",
    layout="wide"
)

# LOAD DATA 

df = load_clean_data()
nlp_df = load_nlp_data()

# PREPARE NLP DATA 

# Create row-level ID for safe merging 
df["record_id"] = np.arange(len(df))
nlp_df["record_id"] = np.arange(len(nlp_df))

# Ensure lyrics availability comed from the cleaned dataset

if "lyrics_available" not in df.columns:

    df["lyrics_available"] = (
        df["lyrics"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.len()
        .gt(0)
    )


# NLP columns available in the verified NLP dataset 

nlp_columns = [
    "record_id",
    "lyrics_unique_word_count",
    "meaningful_word_count",
    "lexical_diversity",
    "avg_word_length",
    "sentiment_polarity",
    "sentiment_subjectivity",
    "dominant_topic",
    "topic_probability",
    "lyric_complexity_group"
]

available_nlp_columns = [
    col for col in nlp_columns
    if col in nlp_df.columns
]

# Merge cleaned chart data + NLP features

lyrics_data = df.merge(
    nlp_df[available_nlp_columns],
    on="record_id",
    how="left",
    validate="one_to_one"
)

# Ensure Core Lyric Features exist

if "lyrics_char_count" not in lyrics_data.columns:

    lyrics_data["lyrics_char_count"] = (
        lyrics_data["lyrics"]
        .fillna("")
        .astype(str)
        .str.len()
    )

if "lyrics_word_count" not in lyrics_data.columns:

    lyrics_data["lyrics_word_count"] = (
        lyrics_data["lyrics"]
        .fillna("")
        .astype(str)
        .str.split()
        .str.len()
    )

# DERIVED NLP FEATURES

lyrics_data["sentiment_intensity"] = (
    lyrics_data["sentiment_polarity"].abs()
)

lyrics_data["lyric_repetition_ratio"] = (
    1 - lyrics_data["lexical_diversity"]
)

# Recreate sentiment category from polarity

lyrics_data["sentiment_category"] = "Neutral"

lyrics_data.loc[
    lyrics_data["sentiment_polarity"] > 0.05,
    "sentiment_category"
] = "Positive"

lyrics_data.loc[
    lyrics_data["sentiment_polarity"] < -0.05,
    "sentiment_category"
] = "Negative"


# PAGE HEADER 

st.title("🎤 Lyrics & NLP Intelligence")

st.markdown(
    """
    Explore the linguistic characteristics of songs in the Spotify Global Weekly Chart dataset.

    This page analyzes:

    - 📝 Lyric length and vocabulary
    - 🔤 Lexical diversity
    - ❤️ Sentiment
    - 🧠 Topic patterns
    - 📊 Lyric characteristics vs chart performance
    """
)


st.warning(
    """
    **Interpretation note:** NLP features describe associations between lyrics and historical chart outcomes. They should not 
    be interpreted as evidence that a particular lyric characteristic causes chart success.
    """
)

# SIDEBAR FILTERS

st.sidebar.header("🔎 NLP Filters")


# Lyrics availability

availability_filter = st.sidebar.selectbox(
    "Lyrics Availability",
    [
        "All Songs",
        "Lyrics Available",
        "Lyrics Unavailable"
    ]
)



# Sentiment Filter

sentiment_options = ["All"] + sorted(
    lyrics_data["sentiment_category"]
    .dropna()
    .unique()
    .tolist()
)


selected_sentiment = st.sidebar.selectbox(
    "Sentiment",
    sentiment_options
)


# Apply filters

filtered_df = lyrics_data.copy()

if availability_filter == "Lyrics Available":

    filtered_df = filtered_df[
        filtered_df["lyrics_available"] == True
    ]


elif availability_filter == "Lyrics Unavailable":

    filtered_df = filtered_df[
        filtered_df["lyrics_available"] == False
    ]


if selected_sentiment != "All":

    filtered_df = filtered_df[
        filtered_df["sentiment_category"] == selected_sentiment
    ]


# KPI SECTION

st.divider()

st.subheader("📊 NLP Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Songs",
        f"{len(filtered_df):,}"
    )


with col2:

    lyrics_count = filtered_df["lyrics_available"].sum()

    st.metric(
        "Lyrics Available",
        f"{lyrics_count:,}"
    )

with col3:

    median_words = filtered_df[
        filtered_df["lyrics_available"]
    ]["lyrics_word_count"].median()

    st.metric(
        "Median Lyric Words",
        f"{median_words:,.0f}"
    )

with col4:

    median_diversity = filtered_df[
        filtered_df["lyrics_available"]
    ]["lexical_diversity"].median()

    st.metric(
        "Median Lexical Diversity",
        f"{median_diversity:.3f}"
    )


# LYRIC LENGTH DISTRIBUTION

st.divider()

st.subheader("📝 Lyric Length Distribution")


lyrics_available_df = filtered_df[
    filtered_df["lyrics_available"]
].copy()

if len(lyrics_available_df) > 0:

    fig = px.histogram(
        lyrics_available_df,
        x="lyrics_word_count",
        nbins=50,
        title="Distribution of Lyric Word Counts",
        labels={
            "lyrics_word_count": "Number of Words"
        }
    )


    fig.update_layout(
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True 
    )

else:

    st.info("No songs with avaialble lyrics match the selected filters.")


# LYRIC LENGTH VS STREAMS

st.subheader("📈 Lyric Length vs Total Streams")

if len(lyrics_available_df) > 0:

    fig = px.scatter(
        lyrics_available_df,
        x="lyrics_word_count",
        y="total",
        size="wks",
        hover_name="song name",
        hover_data=[
            "artist name",
            "wks",
            "pk",
            "t10"
        ],
        log_x=True,
        log_y=True,
        title="Lyric Length vs Total Streams",
        labels={
            "lyrics_word_count": "Lyric Word Count",
            "total": "Total Streams",
            "wks": "Chart Weeks"
        }
    )

    fig.update_layout(
        height=550
    )

    st.plotly_chart(
        fig,
        use_container_width=True 
    )


# LEXICAL DIVERSITY 

st.divider()

st.subheader("🔤 Lexical Diversity")


col1, col2 = st.columns(2)

with col1:

    fig = px.histogram(
        lyrics_available_df,
        x="lexical_diversity",
        nbins=40,
        title="Distribution of Lexical Diversity",
        labels={
            "lexical_diversity": "Unique Words / Total Words"
        }
    )

    fig.update_layout(
        height=450
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    fig = px.scatter(
        lyrics_available_df,
        x="lexical_diversity",
        y="wks",
        size="total",
        hover_name="song name",
        hover_data=[
            "artist name",
            "lyrics_word_count",
            "total"
        ],
        log_y=True,
        title="Lexical Diversity vs Chart Longevity",
        labels={
            "lexical_diversity": "Lexical Diversity",
            "wks": "Chart Weeks"
        }

    )

    fig.update_layout(
        height=450
    )

    st.plotly_chart(
        fig,
        use_container_width=True 
    )

# SENTIMENT ANALYSIS

st.divider()

st.subheader("❤️ Sentiment Analysis")

col1, col2, col3 = st.columns(3)

with col1:

    sentiment_counts = (
        filtered_df["sentiment_category"]
        .value_counts()
        .reset_index()
    )


    sentiment_counts.columns = [
        "Sentiment",
        "Songs"
    ]


    fig = px.pie(
        sentiment_counts,
        names="Sentiment",
        values="Songs",
        title="Sentiment Distribution"
    )


    fig.update_layout(
        height = 400
    )


    st.plotly_chart(
        fig,
        use_container_width=True 
    )


with col2:

    fig = px.histogram(
        lyrics_available_df,
        x="sentiment_polarity",
        nbins=40,
        title="Sentiment Polarity",
        labels={
            "sentiment_polarity": "Polarity"
        }
    )

    fig.add_vline(
        x=0,
        line_dash="dash"
    )

    fig.update_layout(
        height=400
    )

    st.plotly_chart(
        fig,
        use_container_width=True 
    )

with col3:

    fig = px.scatter(
        lyrics_available_df,
        x="sentiment_polarity",
        y="total",
        size="wks",
        hover_name="song name",
        hover_data=[
            "artist name",
            "sentiment_subjectivity"
        ],
        log_y=True,
        title="Sentiment vs Total Streams",
        labels={
            "sentiment_polarity": "Sentiment Polarity",
            "total": "Total Streams"
        }
    )

    fig.update_layout(
        height=400
    )


    st.plotly_chart(
        fig,
        use_container_width=True 
    )


# SENTIMENT SUMMARY

st.subheader("📋 Sentiment Performance Summary")

sentiment_summary = (
    filtered_df
    .groupby("sentiment_category")
    .agg(
        songs=("song name", "count"),
        median_streams =("total", "median"),
        median_weeks=("wks","median"),
        mean_streams=("total","mean"),
        mean_weeks=("wks","mean")
    )
    .reset_index()
)

sentiment_summary_display = sentiment_summary.copy()

sentiment_summary_display["median_streams"] = (
    sentiment_summary_display["median_streams"]
    .apply(format_number)
)


sentiment_summary_display["mean_streams"] = (
    sentiment_summary_display["mean_streams"]
    .apply(format_number)
)

sentiment_summary_display["median_weeks"] = (
    sentiment_summary_display["median_weeks"]
    .round(1)
)

sentiment_summary_display["mean_weeks"] = (
    sentiment_summary_display["mean_weeks"]
    .round(1)
)

sentiment_summary_display.columns = [
    "Sentiment",
    "Songs",
    "Median Streams",
    "Median Weeks",
    "Mean Streams",
    "Mean Weeks"
]

st.dataframe(
    sentiment_summary_display,
    use_container_width=True,
    hide_index=True
)


# TOPIC ANALYSIS

st.divider()

st.subheader("🧠 Lyric Topic Intelligence")

topic_summary = (
    lyrics_available_df
    .groupby("dominant_topic")
    .agg(
        songs=("song name", "count"),
        median_streams=("total", "median"),
        median_weeks=("wks", "mean"),
        mean_topic_confidence=("topic_probability", "mean")
    )
    .reset_index()
)

col1, col2 = st.columns(2)

with col1:

    fig = px.bar(
        topic_summary,
        x="dominant_topic",
        y="songs",
        title="Songs by Dominant Lyric Topic",
        labels={
            "dominant topic": "Topic",
            "songs": "Number of Songs"
        }
    )

    fig.update_layout(
        height=450
    )

    st.plotly_chart(
        fig,
        use_container_width=True 
    )


with col2:


    fig = px.bar(
        topic_summary,
        x="dominant_topic",
        y="median_streams",
        title="Median Streams by Lyric Topic",
        labels={
            "dominant_topic": "Topic",
            "median_streams": "Median Total Streams"
        }
    )

    fig.update_layout(
        height=450
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# TOPIC PERFORMANCE TABLE

st.subheader("📊 Topic Performance")

topic_display = topic_summary.copy()

topic_display["median_streams"] = (
    topic_display["median_streams"]
    .apply(format_number)
)


topic_display["median_weeks"] = (
    topic_display["median_weeks"]
    .round(1)
)

topic_display["mean_topic_confidence"] = (
    topic_display["mean_topic_confidence"]
    .round(3)
)


topic_display.columns = [
    "Topic",
    "Songs",
    "Median Streams",
    "Median Weeks",
    "Mean Topic Confidence"
]

st.dataframe(
    topic_display,
    use_container_width=True,
    hide_index=True 
)

# LYRIC COMPLEXITY

if "lyric_complexity_group" in lyrics_available_df.columns:

    st.divider()

    st.subheader("🧩 Lyric Complexity")

    complexity_summary = (
        lyrics_available_df
        .groupby("lyric_complexity_group", observed=False)
        .agg(
            songs=("song name", "count"),
            median_streams=("total", "median"),
            median_weeks= ("wks", "median"),
            median_words=("lyrics_word_count", "median")
        )
        .reset_index()
    )

    fig = px.bar(
        complexity_summary,
        x="lyric_complexity_group",
        y="median_streams",
        title="Median Streams by Lyric Complexity Group",
        labels={
            "lyric_complexity_group": "Complexity_Group",
            "median_streams": "Median Total Streams"
        }
    )

    fig.update_layout(
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True 
    )

    complexity_display = complexity_summary.copy()

    complexity_display["median_streams"] = (
        complexity_display["median_streams"]
        .apply(format_number)
    )


    complexity_display["median_weeks"] = (
        complexity_display["median_weeks"]
        .round(1)
    )

    complexity_display["median_words"] = (
        complexity_display["median_words"]
        .round(0)
    )

    complexity_display.columns = [
        "Complexity Group",
        "Songs",
        "Median Streams",
        "Median Weeks",
        "Median Words"
    ]

    st.dataframe(
        complexity_display,
        use_container_width=True,
        hide_index=True 
    )

# LYRIC FEATURES CORRELATION

st.divider()

st.subheader("🔗 Lyric Features vs Chart Performance")

correlation_features = [
    "lyrics_word_count",
    "lyrics_unique_word_count",
    "meaningful_word_count",
    "lexical_diversity",
    "avg_word_length",
    "sentiment_polarity",
    "sentiment_subjectivity",
    "topic_probability",
    "wks",
    "total",
    "PkStreams"
]


correlation_features = [
    col for col in correlation_features
    if col in lyrics_available_df.columns 
]

correlation_matrix = (
    lyrics_available_df[correlation_features]
    .corr(method="spearman")
)

fig = px.imshow(
    correlation_matrix,
    text_auto = ".2f",
    aspect = "auto",
    title = "Spearman Correlation Matrix"
)

fig.update_layout(
    height=700
)

st.plotly_chart(
    fig,
    use_container_width=True 
)


# TOP SONGS WITH LYRICS

st.divider()

st.subheader("🔥 Top Songs by Total Streams - Lyrics Intelligence")

top_lyrics_songs = (
    lyrics_available_df
    .sort_values("total", ascending=False)
    .head(15)
    .copy()
)

top_display = top_lyrics_songs[
    [
        "artist name",
        "song name",
        "wks",
        "t10",
        "pk",
        "lyrics_word_count",
        "lexical_diversity",
        "sentiment_category",
        "total"
    ]
].copy()

top_display["total"] = (
    top_display["total"]
    .apply(format_number)
)


top_display["lexical_diversity"] = (
    top_display["lexical_diversity"]
    .round(3)
)

top_display.columns = [
    "Artist",
    "Song",
    "Weeks",
    "Top-10 Weeks",
    "Peak",
    "Lyric Words",
    "Lexical Diversity",
    "Sentiment",
    "Total Streams"
]


st.dataframe(
    top_display,
    use_container_width=True,
    hide_index=True 
)


# FINAL INSIGHTS 

st.divider()

st.subheader("💡 NLP Intelligence - Interpretation")

st.markdown(
    """

    ### What this page helps investigate 

    **1. Lyric Length**

    Longer lyrics can be compared with streaming performance and chart longevity, but length alone should not be 
    interpreted as a predictor of success.

    **2. Lexical diversity**

    Lexical diversity measures the proportion of unique words in the lyrics. Higher values indicate a broader vocabulary 
    relative to lyric length.

    **3. Sentiment**

    Sentiment polarity provides an approximate measure of positive or negative language. Music can succeed across a wide 
    range of emotional styles.

    **4. Topics**

    Topic modeling identifies recurring groups of words and themes across the lyric corpus. Topics represent statistical
    patterns, not manually assigned meanings.

    **5. Chart performance**

    Relationships between lyric characteristics and streams, longevity or peak performance should be treated as 
    associations, not causal effects.
    """
)

# FOOTER

st.divider()

st.caption(
    "Spotify Global Hit Intelligence | Lyrics & NLP Intelligence"
)
 
