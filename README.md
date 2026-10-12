Spotify Global Hit Intelligence

Data-Driven Analysis of Chart Success, Longevity & Lyrics

Overview

Spotify Global Hit Intelligence is a data science portfolio projecct that analyzes historical Spotify Global Weekly Chart performance to explore song popularity, chart longevity, artist performance, lyrics characteristics and patterns associated with successful songs.

The project combines exploratory data analysis, natural language processing, clustering, machine learning and SHAP explainability in an interactive Streamlit dashboard.

Key Features
1) Executive Dashboard: Key performance indicators and project highlights.
2) Song Explorer: Explore song-level chart performance.
3) Artist Intelligence: Analyze artist performance and chart presence.
4) Lyrics NLP: Explore lyrics characteristics and text-based patterns.
5) Hit Segmentation: Identify groups of songs with similar performance characteristics.
6) Machine Learning Intelligence: Evaluate models for high-stream performance, chart longevity and Top-10 classification.
7) SHAP Explainability: Examine model feature contributions.
8) Final Insights & Recommendations: Consolidate analytical findings and recommendations.

Technology Stack
1) Python
2) pandas and Numpy
3) Matplotlib, Seaborn and Plotly
4) Natural language processing tools
5) SHAP
6) Streamlit
7) Jupyter Notebook

Project Structure

Spotify-Global-Hit-Intelligence
|_ app/
|   |_ app.py
|   |_ utils.py
|   |_ pages/
|_ notebook/
|_ outputs/
|   |_ reports/
|   |_ tables/
|_ requirements.txt
|_ .gitignore
|_ README.md 

Run Locally

1) Clone the repository.
2) Create and activate a Python virtual environment.
3) Install the dependencies
4) Start the Streamlit application

python -m venv .venv

Windows PowerShell:

.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app/app.py 

Deployment

The application can be deployed using Streamlit Community Cloud by connecting this GitHub repository and selecting 
app/app.py as the application entry point.

Ensure all required analysis tables and other runtime assets are committed to the repository.

Important Limitations

1) The analysis describes historical charting songs and may not generalize to every song released.
2) Associations between lyrics and chart performance do not establish causation.
3) Model performance depends on the available features, data quality, evaluation strategy and class imbalance.
4) Classification results should be interpreted as analytical estimates, not guarantees of future commercial success.

Author 
Sanu Mathew
Data Science Portfolio Project - Spotify Global Hit Intelligence