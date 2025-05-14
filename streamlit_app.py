import streamlit as st
import pandas as pd

# Load the dataset (replace with your actual file)
@st.cache_data
def load_data():
    return pd.read_csv("lake_health_data.csv")

# Function to calculate health score
def calculate_health_score(lake, weights):
    score = (
        weights['vegetation'] * lake['Vegetation Area Normalized'] +
        weights['barren'] * lake['Barren Area Normalized'] +
        weights['urban'] * lake['Urban Area Normalized'] +
        weights['precipitation'] * lake['Precipitation Normalized'] +
        weights['evaporation'] * lake['Evaporation Normalized'] +
        weights['temperature'] * lake['Air Temperature Normalized'] +
        weights['vegetation'] * lake['Vegetation Area Trend Normalized'] +
        weights['barren'] * lake['Barren Area Trend Normalized'] +
        weights['urban'] * lake['Urban Area Trend Normalized'] +
        weights['precipitation'] * lake['Precipitation Trend Normalized'] +
        weights['evaporation'] * lake['Evaporation Trend Normalized'] +
        weights['temperature'] * lake['Air Temperature Trend Normalized']
    )
    return score

# Streamlit UI
st.title("Lake Health Score Comparator")

# Ask how many lakes to compare
num_lakes = st.number_input("How many lakes do you want to compare?", min_value=1, step=1)

# Collect Lake IDs
lake_ids = []
for i in range(num_lakes):
    lake_id = st.text_input(f"Enter Lake ID #{i + 1} (from 'Lake' column):", key=f"lake_{i}")
    if lake_id:
        lake_ids.append(lake_id.strip())

# When all IDs are entered
if len(lake_ids) == num_lakes:
    df = load_data()

    # Convert Lake IDs to correct type (if needed)
    try:
        lake_ids = [int(lid) for lid in lake_ids]
    except ValueError:
        st.error("Please enter valid numeric Lake IDs.")
        st.stop()

    # Filter data
    selected_lakes = df[df['Lake'].isin(lake_ids)]

    if selected_lakes.empty:
        st.warning("No matching lakes found in the dataset.")
    else:
        # Define equal weights
        weights = {
            'vegetation': 1/6,
            'barren': 1/6,
            'urban': 1/6,
            'precipitation': 1/6,
            'evaporation': 1/6,
            'temperature': 1/6
        }

        # Calculate scores
        selected_lakes = selected_lakes.copy()
        selected_lakes["Health Score"] = selected_lakes.apply(lambda row: calculate_health_score(row, weights), axis=1)

        # Sort and rank
        selected_lakes["Rank"] = selected_lakes["Health Score"].rank(ascending=False).astype(int)
        selected_lakes = selected_lakes.sort_values("Rank")

        st.subheader("Health Score Results")
        st.dataframe(selected_lakes[["Lake", "Health Score", "Rank"] + [col for col in selected_lakes.columns if "Normalized" in col]])

        # Download option
        csv_data = selected_lakes.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV of Selected Lakes",
            data=csv_data,
            file_name="selected_lakes_health_scores.csv",
            mime="text/csv"
        )
