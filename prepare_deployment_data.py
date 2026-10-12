from pathlib import Path
import pandas as pd

root = Path.cwd()
source = root / "data" / "processed"
destination = root / "data" / "deployment"

clean_path = source / "spotify_global_chart_totals_lyrics_cleaned.csv"
nlp_path = source / "spotify_lyrics_nlp_features.csv"
segments_path = source / "spotify_hit_segments.csv"

clean = pd.read_csv(clean_path, low_memory=False)

# Remove full lyrics text from the deployment copy.
text_columns = ["lyrics", "lyrics_original", "lyrics_clean"]
clean = clean.drop(
    columns=[c for c in text_columns if c in clean.columns],
    errors="ignore"
)

clean.to_csv(
    destination / "spotify_global_chart_totals_lyrics_cleaned.csv",
    index=False
)

nlp = pd.read_csv(nlp_path, low_memory=False)
nlp.to_csv(
    destination / "spotify_lyrics_nlp_features.csv",
    index=False
)

segments = pd.read_csv(segments_path, low_memory=False)
segments.to_csv(
    destination / "spotify_hit_segments.csv",
    index=False
)

print("Deployment files created:")
for path in sorted(destination.glob("*.csv")):
    print(f"{path.name}: {path.stat().st_size / (1024 * 1024):.2f} MB")

print("\nCleaned data rows:", len(clean))
print("NLP data rows:", len(nlp))
print("Segmentation rows:", len(segments))
print("Lyrics text columns remaining:",
      [c for c in text_columns if c in clean.columns])
