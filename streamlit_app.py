import streamlit as st
import pandas as pd
import numpy as np

# --- Lake Health Score Calculation Function ---
def calculate_lake_health_score(df,
                                vegetation_weight=1/6, barren_weight=1/6, urban_weight=1/6,
                                precipitation_weight=1/6, evaporation_weight=1/6, air_temperature_weight=1/6):

    # Convert columns to numeric
    for col in ['Vegetation Area', 'Barren Area', 'Urban Area', 'Precipitation', 'Evaporation', 'Air Temperature']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    latest_year_data = df[df['Year'] == df['Year'].max()].copy()
    if latest_year_data.empty:
        return pd.DataFrame()

    # Normalize current levels
    latest_year_data['Vegetation Area Normalized'] = (latest_year_data['Vegetation Area'] - latest_year_data['Vegetation Area'].min()) / (latest_year_data['Vegetation Area'].max() - latest_year_data['Vegetation Area'].min())
    latest_year_data['Barren Area Normalized'] = 1 - (latest_year_data['Barren Area'] - latest_year_data['Barren Area'].min()) / (latest_year_data['Barren Area'].max() - latest_year_data['Barren Area'].min())
    latest_year_data['Urban Area Normalized'] = 1 - (latest_year_data['Urban Area'] - latest_year_data['Urban Area'].min()) / (latest_year_data['Urban Area'].max() - latest_year_data['Urban Area'].min())
    latest_year_data['Precipitation Normalized'] = (latest_year_data['Precipitation'] - latest_year_data['Precipitation'].min()) / (latest_year_data['Precipitation'].max() - latest_year_data['Precipitation'].min())
    latest_year_data['Evaporation Normalized'] = 1 - (latest_year_data['Evaporation'] - latest_year_data['Evaporation'].min()) / (latest_year_data['Evaporation'].max() - latest_year_data['Evaporation'].min())
    latest_year_data['Air Temperature Normalized'] = 1 - (latest_year_data['Air Temperature'] - latest_year_data['Air Temperature'].min()) / (latest_year_data['Air Temperature'].max() - latest_year_data['Air Temperature'].min())

    for col in latest_year_data.columns:
        if 'Normalized' in col:
            latest_year_data[col] = latest_year_data[col].replace([np.inf, -np.inf, np.nan], 0)

    # Trends over time
    trends = df.groupby('Lake').apply(lambda x: pd.Series({
        'Vegetation Area Trend': np.polyfit(x['Year'], x['Vegetation Area'], 1)[0],
        'Barren Area Trend': np.polyfit(x['Year'], x['Barren Area'], 1)[0],
        'Urban Area Trend': np.polyfit(x['Year'], x['Urban Area'], 1)[0],
        'Precipitation Trend': np.polyfit(x['Year'], x['Precipitation'], 1)[0],
        'Evaporation Trend': np.polyfit(x['Year'], x['Evaporation'], 1)[0],
        'Air Temperature Trend': np.polyfit(x['Year'], x['Air Temperature'], 1)[0],
    }))

    # Normalize trends
    trends['Vegetation Area Trend Normalized'] = (trends['Vegetation Area Trend'] - trends['Vegetation Area Trend'].min()) / (trends['Vegetation Area Trend'].max() - trends['Vegetation Area Trend'].min())
    trends['Barren Area Trend Normalized'] = 1 - (trends['Barren Area Trend'] - trends['Barren Area Trend'].min()) / (trends['Barren Area Trend'].max() - trends['Barren Area Trend'].min())
    trends['Urban Area Trend Normalized'] = 1 - (trends['Urban Area Trend'] - trends['Urban Area Trend'].min()) / (trends['Urban Area Trend'].max() - trends['Urban Area Trend'].min())
    trends['Precipitation Trend Normalized'] = (trends['Precipitation Trend'] - trends['Precipitation Trend'].min()) / (trends['Precipitation Trend'].max() - trends['Precipitation Trend'].min())
    trends['Evaporation Trend Normalized'] = 1 - (trends['Evaporation Trend'] - trends['Evaporation Trend'].min()) / (trends['Evaporation Trend'].max() - trends['Evaporation Trend'].min())
    trends['Air Temperature Trend Normalized'] = 1 - (trends['Air Temperature Trend'] - trends['Air Temperature Trend'].min()) / (trends['Air Temperature Trend'].max() - trends['Air Temperature Trend'].min())

    for col in trends.columns:
        if 'Normalized' in col:
            trends[col] = trends[col].replace([np.inf, -np.inf, np.nan], 0)

    # Join and compute score
    latest_year_data = latest_year_data.set_index('Lake')
    combined_data = latest_year_data.join(trends, how='inner')

    combined_data['Health Score'] = (
        vegetation_weight * combined_data['Vegetation Area Normalized'] +
        barren_weight * combined_data['Barren Area Normalized'] +
        urban_weight * combined_data['Urban Area Normalized'] +
        precipitation_weight * combined_data['Precipitation Normalized'] +
        evaporation_weight * combined_data['Evaporation Normalized'] +
        air_temperature_weight * combined_data['Air Temperature Normalized'] +
        vegetation_weight * combined_data['Vegetation Area Trend Normalized'] +
        barren_weight * combined_data['Barren Area Trend Normalized'] +
        urban_weight * combined_data['Urban Area Trend Normalized'] +
        precipitation_weight * combined_data['Precipitation Trend Normalized'] +
        evaporation_weight * combined_data['Evaporation Trend Normalized'] +
        air_temperature_weight * combined_data['Air Temperature Trend Normalized']
    )

    combined_data['Rank'] = combined_data['Health Score'].rank(ascending=False)
    return combined_data.reset_index()


# -------------------- STREAMLIT APP --------------------

st.title("Lake Health Score Dashboard")

# Load CSV from fixed path
df = pd.read_csv("lake_health_data.csv")

# Ask number of lakes
num_lakes = st.number_input("How many lakes do you want to compare?", min_value=1, max_value=10, step=1)

# Gather lake IDs
lake_ids = []
for i in range(num_lakes):
    lake_id = st.text_input(f"Enter Lake ID #{i + 1}", key=f"lake_{i}")
    if lake_id:
        lake_ids.append(lake_id)

# Process only if valid IDs provided
if lake_ids:
    selected_df = df[df["Lake"].astype(str).isin(lake_ids)]

    if selected_df.empty:
        st.error("No data found for the entered Lake IDs.")
    else:
        # Compute full health scores
        health_scores = calculate_lake_health_score(selected_df)

        # ✅ Display only selected summary columns
        st.subheader("Lake Health Scores")
        st.dataframe(health_scores[["Lake", "Health Score", "Rank"]])  # Show only 3 columns

        # ✅ Download only original data (not the computed scores)
        csv = selected_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Original CSV for Selected Lakes", csv, "selected_lake_data.csv", "text/csv")

        # ✅ You still have access to full health_scores DataFrame internally
        # e.g., use: health_scores.to_csv("full_scores.csv")
