import streamlit as st 

# PAGE CONFIGURATION

st.set_page_config(
    page_title = "Spotify Global Hit Intelligence",
    page_icon = "🎵",
    layout = "wide",
    initial_sidebar_state = "expanded"
)


# APPLICATION TITLE

st.title("🎵 Spotify Global Hit Intelligence")

st.markdown(
    """

    ### Data-Driven Analysis of Chart Success, Longevity & Lyrics.

    Explore global Spotify chart performance through:

    - 📊 Chart Performance Analytics
    - 👤 Artist Intelligence
    - 🎤 Lyrics & NLP
    - 🎯 Hit Segmentation
    - 🤖 Machine Learning
    - 🔎 SHAP Explainability
    """ 
)

# SIDEBAR 

st.sidebar.title("🎵 Spotify Hit Intelligence")

st.sidebar.markdown(
    """

    **Navigate using the pages in the sidebar.**

    This application transforms the analytical work from the project notebooks into an
    interactive decision-support system.
    """ 
)


# PROJECT OVERVIEW 

st.divider()

st.subheader("📌 Project Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.info(
        """

        **Dataset**

        8,000+ songs from the Global Weekly Chart,
        combining chart-performance statistics with lyrics.
        """
    )

with col2:
    st.info(
        """
        **Analytics**

        Exploratory analysis, artist intelligence, NLP, clustering and machine learning.
        """
    )


with col3:
    st.info(
        """
        **Decision Support**

        Identify performance patterns and explain the factors associated with different
        success outcomes.
        """
    )



# IMPORTANT MODELING NOTE

st.divider()

st.warning(
    """
    **Modeling note:** Chart metrics such as total streams, chart weeks and Top-10 weeks represent lifetime 
    outcomes. They are therefore treated carefully in predictive analysis to avoid target leakage.
    """
)


# FOOTER 

st.divider()

st.caption(
    "Spotify Global Hit Intelligence | Data Science Portfolio Project"
)