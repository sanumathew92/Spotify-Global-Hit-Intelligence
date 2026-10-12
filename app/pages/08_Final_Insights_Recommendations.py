from pathlib import Path 

import pandas as pd 
import plotly.express as px 
import streamlit as st 

# PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FINAL_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "final_insights"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_PATH = OUTPUT_DIR / "reports" / "executive_summary.txt"


# PAGE CONFIGURATION

st.set_page_config(
    page_title="Final Insights & Recommendations",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Final Insights & Recommendations")

st.markdown(
    """
    ## From chart data to actionable intelligence

    This executive dashboard consolidates findings from chart performance analytics, artist intelligence, lyrics NLP,
    hit segmentation, machine learning and SHAP explainability.
    """
)

# DATA LOADING 

@st.cache_data
def load_csv(path_string):
    path = Path(path_string)

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path, low_memory=False)
    except (
        OSError,
        pd.errors,ParserError,
        UnicodeDecodeError
    ):
        return pd.DataFrame()

#st.cache_data
def load_text(path_string):
    path = Path(path_string)

    if not path.exists():
        return ""

    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""

def find_column(df, candidates):
    """Return a matching column without assuming exact capitalization."""
    lookup = {
        str(col).strip().lower(): col 
        for col in df.columns 
    }

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def display_table(df, title, key):
    st.subheader(title)

    if df.empty:
        st.info(f"No data available for {title}.")
        return 

    st.caption(
        f"{len(df):,} rows . {len(df.columns)} columns"
    )

    search = st.text_input(
        "Filter rows",
        key=f"filter_{key}",
        placeholder="Type a keyword to filter this table..."
    )

    display_df = df.copy()

    if search.strip():
        mask = display_df.astype(str).apply(
            lambda column: column.str.contains(
                search.strip(),
                case=False,
                regex=False,
                na=False
            )
        ).any(axis=1)

        display_df = display_df.loc[mask]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


def show_numeric_chart(df, title, key):
    """Show a chart only when suitable columns are available."""
    if df.empty:
        st.info(f"No data available for {title}")
        return 

    numeric_cols = df.select_dtypes(
        include="number"
    ).columns.tolist()


    category_col = find_column(
        df,
        [
            "segment",
            "cluster",
            "artist",
            "metric",
            "feature",
            "target",
            "task",
            "category",
            "insight",
            "recommendation",
            "group",
            "name"
        ]
    )

    if not numeric_cols:
        st.info(
            "This table has no numeric columns suitable for a chart."
        )
        return 

    if category_col is None:
        non_numeric_cols = df.select_dtypes(
            exclude="number"
        ).columns.tolist()


        if non_numeric_cols:
            category_col = non_numeric_cols[0]


    if category_col is None:
        st.info(
            "No category column was detected for a comparison chart. "
        )
        return 

    metric_col = st.selectbox(
        "Select metric",
        numeric_cols,
        key=f"chart_metric_{key}"
    ) 

    chart_df = df[[category_col, metric_col]].copy()
    chart_df[metric_col] = pd.to_numeric(
        chart_df[metric_col],
        errors="coerce"
    )

    chart_df= chart_df.dropna(subset=[metric_col])

    if chart_df.empty:
        st.info("No valid numeric observations are available.")
        return

    chart_df[category_col] = (
        chart_df[category_col].astype(str)
    )

    chart_df = chart_df.head(30)

    fig = px.bar(
        chart_df,
        x=category_col,
        y=metric_col,
        title=f"{title}: {metric_col}",
        text_auto=".3s"
    )

    fig.update_layout(
        showlegend=False,
        xaxis_title=category_col,
        yaxis_title=metric_col,
        xaxis_tickangle=-30
    )

    st.plotly_chart(
        fig,
        use_container_width=True 
    )

# LOAD FINAL INSIGHTS TABLES

project_kpis = load_csv(
    str(FINAL_DIR / "final_project_kpis.csv")
)

project_insights = load_csv(
    str(FINAL_DIR / "final_project_insights.csv")
)

recommendations = load_csv(
    str(FINAL_DIR / "final_recommendations.csv")
)

segment_summary = load_csv(
    str(FINAL_DIR / "final_segment_summary.csv")
)

shap_summary = load_csv(
    str(FINAL_DIR / "final_shap_summary.csv")
)
top10_comparison = load_csv(
    str(FINAL_DIR / "final_top10_comparison.csv")
)

longevity_summary = load_csv(
    str(FINAL_DIR / "final_longevity_summary.csv")
)

success_drivers = load_csv(
    str(FINAL_DIR / "success_drivers_summary.csv")
)

artist_summary = load_csv(
    str(FINAL_DIR / "final_artist_summary.csv")
)

executive_summary= load_text(
    str(REPORT_PATH)
)

nlp_correlations = load_csv(
    str(FINAL_DIR / "final_nlp_correlations.csv")
)

# OVERVIEW KPIS 

st.header("1. Project at a Glance")

if not project_kpis.empty:
    st.caption(
        "Key performance indicators from your saved analysis."
    )

    # Support common metric/value table structures.
    metric_col = find_column(
        project_kpis,
        ["metric", "kpi", "name", "indicator", "measure"]
    )

    value_col = find_column(
        project_kpis,
        ["value", "metric_value", "result", "count", "score"]
    )

    if(
        metric_col is not None 
        and value_col is not None 
        and metric_col != value_col
    ):

        kpi_data = project_kpis[
            [metric_col, value_col]
        ].dropna(subset=[metric_col])


        kpi_data = kpi_data.head(8)

        for start in range(0, len(kpi_data), 4):
            row_data = kpi_data.iloc[start:start + 4]
            columns = st.columns(len(row_data))

            for column, (_, record) in zip(
                columns,
                row_data.iterrows()
            ):

                label= str(record[metric_col])
                value = record[value_col]

                if pd.isna(value):
                  value_text = "N/A"

                elif isinstance(value, (int,float)):
                    label_lower = label.lower()

                    if "rate" in label_lower and '%' in label:
                        value_text = f"{value:.1f}%"

                    elif abs(value) >= 1_000_000_000:
                        value_text = f"{value / 1_000_000_000:.2f}B"

                    elif abs(value) >= 1_000_000:
                        value_text = f"{value / 1_000_000:.2f}M"

                    elif float(value).is_integer():
                        value_text = f"{int(value):,}"

                    else:
                        value_text = f"{value:,.2f}"
                
                else:
                    valu_text = str(value)

                column.metric(
                    label=label[:50],
                    value=value_text
                )

        with st.expander("View complete KPI table"):
            st.dataframe(
                project_kpis,
                use_container_width=True,
                hide_index=True
            )


    else:
        st.dataframe(
            project_kpis,
            use_container_width=True,
            hide_index=True 
        )

else:

    st.warning(
        "The final KPI table was not found or is empty. "
        "Check outputs/tables/final_insights/final_project_kpis.csv."
    )


# EXECUTIVE SUMMARY 

st.divider()
st.header("2. Executive Summary")

if executive_summary.strip():
    st.markdown(executive_summary)
else:
    st.info(
        "The executive summary text file is unavailable. "
        "Explore the saved insight and recommendation table below."
    )


# NAVIGABLE SECTIONS

overview_tab, insights_tab, recommendations_tab, segments_tab, shap_tab = (
    st.tabs(
        [
            "Key Insights",
            "Success Drivers",
            "Recommendations",
            "Hit Segments",
            "SHAP Findings"
        ]
    )
)


# KEY INSIGHTS 

with overview_tab:
    st.subheader("Final Project Insights")

    if not project_insights.empty:
        display_table(
            project_insights,
            "Consolidated Findings",
            "project_insights"
        )

    else:
        st.info("No consolidated project insights were found.")

    st.subheader("Top-10 Performance")

    if not top10_comparison.empty:
        display_table(
            top10_comparison,
            "Top-10 Comparison",
            "top10_comparison"
        )
    else:
        st.info("No Top-10 comparison table was found.")

    st.subheader("Chart Longevity")

    if not longevity_summary.empty:
        display_table(
            longevity_summary,
            "Longevity Summary",
            "longevity_summary"
        )
    else:
        st.info("No longevity summary table was found.")


# SUCCESS DRIVERS

with insights_tab:
    st.subheader("What Is Associated With Chart Success?")

    if not success_drivers.empty:
        display_table(
            success_drivers,
            "Success Drivers Summary",
            "success_drivers"
        )

        with st.expander("Explore numeric patterns"):
            show_numeric_chart(
                success_drivers,
                "Success Drivers",
                "success_drivers"
            )

    else:
        st.info("No success-driver summary was found.")

    
    st.subheader("Artist Intelligence")

    if not artist_summary.empty:
        display_table(
            artist_summary,
            "Artist Summary",
            "artist_summary"
        )
    else:
        st.info("No final artist summary was found.")

    st.subheader("Lyrics and Chart Performance")

    if not nlp_correlations.empty:
        display_table(
            nlp_correlations,
            "NLP Correlation Results",
            "nlp_correlations"
        )
    else:
        st.info("No final NLP correlation table was found.")


# RECOMMENDATIONS

with recommendations_tab:
    st.subheader("Data-Driven Recommendations")

    if not recommendations.empty:
        display_table(
            recommendations,
            "Project Recommendations",
            "recommendations"
        )
    else:
        st.info("No final recommendations table was found.")

    st.markdown(
        """
        **How to use these recommendations**

        - Prioritize actions supported by multiple analyses.
        - Distinguish predictive signals from causal evidence.
        - Validate model results on appropriate held-out data.
        - Revisit recommendations when new data becomes available.
        """
    )

# HIT SEGMENTS 

with segments_tab:
    st.subheader("Song Segmentation")

    if not segment_summary.empty:
        display_table(
            segment_summary,
            "Final Segment Summary",
            "segment_summary"
        )

        with st.expander("Explore segment metrics"):
            show_numeric_chart(
                segment_summary,
                "Segment Comparison",
                "segment_summary"
            )

    else:
        st.info("No final segment summary was found.")


# SHAP FINDINGS

with shap_tab:
    st.subheader("Model Explainability Findings")

    if not shap_summary.empty:
        display_table(
            shap_summary,
            "Final SHAP Summary",
            "shap_summary"
        )

        with st.expander("Explore SHAP metrics"):
            show_numeric_chart(
                shap_summary,
                "SHAP Summary",
                "shap_summary"
            )

    else:
        st.info("No final SHAP summary was found.")


# METHODOLOGY AND LIMITATIONS

st.divider()
st.header("3. Methodolody and Limitations")

st.markdown(
    """
    **Historical data:** Findings describe patterns in the available Spotify Global Weekly Chart dataset.
    **Association is not causation:** Lyrics, artist history, streaming metrics and chart longevity can be associated 
    without one necessarily causing another.
    **Prediction timing:** Performance metrics can be misleading if features unavailable at prediction time leak 
    information from the test outcome.
    **Generalization:** Results from charting songs may not generalize to every released song or future music market.
    **Business use:** Treat recommendations as hypotheses to validate, not guarantees of commercial success.
    """
)

st.caption(
    "Spotify Global Hit Intelligence | Final Insights & Recommendations"
)
