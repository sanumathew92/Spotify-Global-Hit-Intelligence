import streamlit as st 
import plotly.express as px 

from utils import (
    load_clean_data,
    calculate_kpis,
    format_number,
    format_weeks,
    create_longevity_segments,
    get_top_songs
)

# PAGE CONFIGURATION

st.set_page_config(
    page_title="Executive Dashboard",
    page_icon= "📊",
    layout="wide"
)

# LOAD DATA 

df = load_clean_data()

kpis = calculate_kpis(df)

df = create_longevity_segments(df)


# HEADER 

st.title ("📊 Executive Dashboard")

st.markdown(
    """
    ### Spotify Global Hit Intelligence

    A high-level view of global chart performance, longevity and streaming success.
    """
)


# KPI SECTION 

st.subheader("📌 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Songs Analyzed",
        f"{kpis['total_songs']:,}"
    )

with col2:
    st.metric(
        "Unique Artists",
        f"{kpis['unique_artists']:,}"
    )

with col3:
    st.metric(
        "Lyrics Available",
        f"{kpis['lyrics_available']:,}"
    )

with col4:
    st.metric(
        "Top-10 Songs",
        f"{kpis['top10_songs']:,}"
    )


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Longest Chart Run",
        f"{format_weeks(kpis['max_weeks'])} weeks"
    )

with col2:
    st.metric(
        "Highest Total Streams",
        format_number(kpis["max_total_streams"])
    )

with col3:
    st.metric(
        "Median Chart Weeks",
        format_number(kpis["median_total_streams"])
    )


# CHART LONGEVITY 

st.divider()

st.subheader("⏱️ Chart Longevity")

longevity_counts = (
    df["longevity_group"]
    .value_counts()
    .reindex(
        [
            "Short",
            "Moderate",
            "Long",
            "Very Long",
            "Exceptional"
        ]
    )
    .reset_index()
)

longevity_counts.columns = [
    "Longevity Group",
    "Songs"
]

fig = px.bar(
    longevity_counts,
    x="Longevity Group",
    y="Songs",
    title="Songs by Chart Longevity",
    text="Songs"
)

fig.update_traces(
    textposition = "outside" 
)


fig.update_layout(
    height=450,
    showlegend=False 
)

st.plotly_chart(
    fig,
    use_container_width=True 
)

# TOP SONGS 

st.divider()

st.subheader("🔥 Top Songs by Total Streams")

top_songs = get_top_songs(
    df,
    metric="total",
    n=10
)

top_songs_display = top_songs.copy()

top_songs_display["Total Streams"] = (
    top_songs_display["total"]
    .apply(format_number)
)

top_songs_display["Chart Weeks"] = (
    top_songs_display["wks"]
    .astype(int)
)

top_songs_display["Peak Position"] = (
    top_songs_display["pk"]
    .astype(int)
)

top_songs_display["Top-10 Weeks"] = (
    top_songs_display["t10"]
    .astype(int)
)

top_songs_display = top_songs_display[
    [
        "artist name",
        "song name",
        "Chart Weeks",
        "Top-10 Weeks",
        "Peak Position",
        "Total Streams"
    ]
]

top_songs_display.columns = [
    "Artist",
    "Song",
    "Chart Weeks",
    "Top-10 Weeks",
    "Peak Position",
    "Total Streams"
]

st.dataframe(
    top_songs_display,
    use_container_width=True,
    hide_index=True 
)

# TOP SONGS VISUALIZATION

fig = px.bar(
    top_songs.sort_values("total"),
    x="total",
    y="song name",
    color="artist name",
    orientation = "h",
    title = "Top 10 Songs by Cumulative Streams",
    labels = {
        "total": "Total Streams",
        "song name": "Song",
        "artist name": "Artist"
    }
)

fig.update_layout(
    height=550,
    showlegend=False
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# TOP-10 PERFORMANCE

st.divider()

st.subheader("🏆 Top-10 Performance")

top10_status = (
    df["t10"]
    .gt(0)
    .map({
        True:"Reached Top 10",
        False: "Did Not Reach Top 10"
    })
    .value_counts()
    .reset_index()
)

top10_status.columns = [
    "Status",
    "Songs"
]

fig = px.pie(
    top10_status,
    names="Status",
    values="Songs",
    title="Top-10 Reach"
)

fig.update_layout(
    height=450
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# EXECUTIVE INSIGHTS

st.divider()

st.subheader("🏆 Executive Insights")

exceptional_count = (
    df["longevity_group"]
    .eq("Exceptional")
    .sum()
)

exceptional_pct = (
    exceptional_count / len(df) * 100
)

st.markdown(
    f"""
    **Key observations**

    - The dataset contains **{kpis['total_songs']:,} unique songs**
      represented across **{kpis['unique_artists']:,} artists**.

    - Only approximately **{exceptional_pct:.1f}%** of songs fall into
     the **Exceptional** longevity category (53+ weeks).

    - The strongest songs combine chart longevity with substantial cumulative 
      streaming performance.

    - Reaching the Top 10 and maintaining a long chart run represent related but 
     distinct dimensions of chart success.

     -The dashboard separates descriptive lifetime performance from predictive 
      modeling to reduce the risk of target leakage.
    """
)


# FOOTER 

st.divider()

st.caption(
    "Spotify Global Hit Intelligence | Executive Dashboard"
)