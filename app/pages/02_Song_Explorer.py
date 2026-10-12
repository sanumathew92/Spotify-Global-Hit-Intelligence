import streamlit as st
import plotly.express as px
import pandas as pd

from utils import (
    load_song_explorer_data,
    format_number,
    format_weeks,
    create_longevity_segments
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Song Explorer",
    page_icon="🎵",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

df = load_song_explorer_data()

df = create_longevity_segments(df)


# ============================================================
# PAGE HEADER
# ============================================================

st.title("🎵 Song Explorer")

st.markdown(
    """
    Explore individual songs and understand their chart
    performance, longevity, streaming success and lyric profile.
    """
)


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("🔎 Song Filters")


# ============================================================
# ARTIST FILTER
# ============================================================

artists = sorted(
    df["artist name"]
    .dropna()
    .astype(str)
    .unique()
)

selected_artist = st.sidebar.selectbox(
    "Select Artist",
    ["All Artists"] + artists
)


# ============================================================
# APPLY ARTIST FILTER
# ============================================================

filtered_df = df.copy()

if selected_artist != "All Artists":

    filtered_df = filtered_df[
        filtered_df["artist name"] == selected_artist
    ]


# ============================================================
# TOP-10 FILTER
# ============================================================

top10_filter = st.sidebar.selectbox(
    "Top-10 Status",
    [
        "All Songs",
        "Reached Top 10",
        "Did Not Reach Top 10"
    ]
)

if top10_filter == "Reached Top 10":

    filtered_df = filtered_df[
        filtered_df["t10"] > 0
    ]

elif top10_filter == "Did Not Reach Top 10":

    filtered_df = filtered_df[
        filtered_df["t10"] == 0
    ]


# ============================================================
# LONGEVITY FILTER
# ============================================================

longevity_options = [
    "All",
    "Short",
    "Moderate",
    "Long",
    "Very Long",
    "Exceptional"
]

selected_longevity = st.sidebar.selectbox(
    "Chart Longevity",
    longevity_options
)

if selected_longevity != "All":

    filtered_df = filtered_df[
        filtered_df["longevity_group"] == selected_longevity
    ]


# ============================================================
# MINIMUM CHART WEEKS
# ============================================================

min_weeks = st.sidebar.slider(
    "Minimum Chart Weeks",
    min_value=1,
    max_value=int(df["wks"].max()),
    value=1
)

filtered_df = filtered_df[
    filtered_df["wks"] >= min_weeks
]


# ============================================================
# SONG SELECTION
# ============================================================

st.subheader("🎧 Select a Song")


if filtered_df.empty:

    st.warning(
        "No songs match the selected filters. "
        "Try relaxing one or more filters."
    )

    st.stop()


filtered_df = filtered_df.copy()

filtered_df["song_label"] = (
    filtered_df["song name"].astype(str)
    + " — "
    + filtered_df["artist name"].astype(str)
)


song_options = (
    filtered_df["song_label"]
    .drop_duplicates()
    .sort_values()
    .tolist()
)


selected_song = st.selectbox(
    "Choose a song",
    song_options
)


# ============================================================
# GET SELECTED SONG
# ============================================================

selected_row = filtered_df[
    filtered_df["song_label"] == selected_song
].iloc[0]


artist = selected_row["artist name"]
song = selected_row["song name"]


# ============================================================
# SONG HEADER
# ============================================================

st.divider()

st.header(f"🎵 {song}")

st.markdown(
    f"### by **{artist}**"
)


# ============================================================
# CHART PERFORMANCE
# ============================================================

st.subheader("📊 Chart Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Chart Weeks",
        format_weeks(selected_row["wks"])
    )

with col2:
    st.metric(
        "Top-10 Weeks",
        int(selected_row["t10"])
    )

with col3:
    st.metric(
        "Peak Position",
        f"#{int(selected_row['pk'])}"
    )

with col4:
    st.metric(
        "Total Streams",
        format_number(selected_row["total"])
    )


# ============================================================
# SECONDARY KPIs
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Peak Streams",
        format_number(selected_row["PkStreams"])
    )

with col2:

    streams_per_week = (
        selected_row["total"] /
        selected_row["wks"]
        if selected_row["wks"] > 0
        else 0
    )

    st.metric(
        "Streams / Week",
        format_number(streams_per_week)
    )

with col3:

    top10_ratio = (
        selected_row["t10"] /
        selected_row["wks"]
        if selected_row["wks"] > 0
        else 0
    )

    st.metric(
        "Top-10 Week Ratio",
        f"{top10_ratio * 100:.1f}%"
    )

with col4:

    st.metric(
        "Longevity Segment",
        str(selected_row["longevity_group"])
    )


# ============================================================
# PERFORMANCE PROFILE
# ============================================================

st.divider()

st.subheader("📈 Performance Profile")

profile_data = pd.DataFrame(
    {
        "Metric": [
            "Chart Weeks",
            "Top-10 Weeks",
            "Peak Position Score",
            "Peak Stream Share"
        ],
        "Value": [
            selected_row["wks"],
            selected_row["t10"],
            (247 - selected_row["pk"]) / 246,
            (
                selected_row["PkStreams"] /
                selected_row["total"]
                if selected_row["total"] > 0
                else 0
            )
        ]
    }
)

fig = px.bar(
    profile_data,
    x="Metric",
    y="Value",
    title=f"Performance Profile — {song}",
    text="Value"
)

fig.update_traces(
    textposition="outside"
)

fig.update_layout(
    height=450,
    showlegend=False
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# LYRICS PROFILE
# ============================================================

st.divider()

st.subheader("🎤 Lyrics Profile")


lyrics_available = bool(
    selected_row["lyrics_available"]
)


if lyrics_available:

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Lyric Words",
            format_number(
                selected_row["lyrics_word_count"]
            )
        )

    with col2:
        st.metric(
            "Unique Words",
            format_number(
                selected_row["lyrics_unique_word_count"]
            )
        )

    with col3:

        lexical_diversity = (
            selected_row["lexical_diversity"]
        )

        if pd.notna(lexical_diversity):

            st.metric(
                "Lexical Diversity",
                f"{lexical_diversity:.3f}"
            )

        else:

            st.metric(
                "Lexical Diversity",
                "N/A"
            )

    with col4:

        sentiment = (
            selected_row["sentiment_polarity"]
        )

        if pd.notna(sentiment):

            st.metric(
                "Sentiment",
                f"{sentiment:.3f}"
            )

        else:

            st.metric(
                "Sentiment",
                "N/A"
            )

else:

    st.info(
        "Lyrics are not available for this song."
    )


# ============================================================
# LYRIC DETAILS
# ============================================================

if lyrics_available:

    col1, col2, col3 = st.columns(3)

    with col1:

        sentiment_category = (
            selected_row["sentiment_category"]
        )

        if pd.notna(sentiment_category):

            st.info(
                f"**Sentiment Category**\n\n"
                f"{sentiment_category}"
            )

    with col2:

        topic = selected_row["dominant_topic"]

        if pd.notna(topic):

            st.info(
                f"**Dominant Topic**\n\n"
                f"Topic {int(topic)}"
            )

    with col3:

        complexity = (
            selected_row["lyric_complexity_group"]
        )

        if pd.notna(complexity):

            st.info(
                f"**Lyric Complexity**\n\n"
                f"{complexity}"
            )


# ============================================================
# VIEW LYRICS
# ============================================================

if lyrics_available:

    with st.expander("🎤 View Lyrics"):

        if (
            "lyrics_clean" in selected_row.index
            and pd.notna(selected_row["lyrics_clean"])
            and str(selected_row["lyrics_clean"]).strip()
        ):

            st.text(
                selected_row["lyrics_clean"]
            )

        elif (
            "lyrics" in selected_row.index
            and pd.notna(selected_row["lyrics"])
        ):

            st.text(
                selected_row["lyrics"]
            )

        else:

            st.info(
                "Lyrics text is unavailable."
            )


# ============================================================
# SONG VS DATASET
# ============================================================

st.divider()

st.subheader("📊 Song vs Dataset")

comparison_data = pd.DataFrame(
    {
        "Metric": [
            "Chart Weeks",
            "Total Streams",
            "Peak Streams"
        ],
        "Selected Song": [
            selected_row["wks"],
            selected_row["total"],
            selected_row["PkStreams"]
        ],
        "Dataset Median": [
            df["wks"].median(),
            df["total"].median(),
            df["PkStreams"].median()
        ]
    }
)

st.dataframe(
    comparison_data,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# SONG DETAILS
# ============================================================

st.divider()

st.subheader("🔎 Song Details")

details = pd.DataFrame(
    {
        "Attribute": [
            "Artist",
            "Song",
            "Chart Weeks",
            "Top-10 Weeks",
            "Peak Position",
            "Peak Streams",
            "Total Streams",
            "Longevity Segment"
        ],
        "Value": [
            artist,
            song,
            int(selected_row["wks"]),
            int(selected_row["t10"]),
            f"#{int(selected_row['pk'])}",
            format_number(
                selected_row["PkStreams"]
            ),
            format_number(
                selected_row["total"]
            ),
            str(selected_row["longevity_group"])
        ]
    }
)

st.dataframe(
    details,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Spotify Global Hit Intelligence | Song Explorer"
)