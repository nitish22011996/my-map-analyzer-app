import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
import requests

# --- Load CSV in background ---
CSV_PATH = "lake_health_data.csv"  # <-- change this to your actual CSV path

@st.cache_data
def load_data():
    return pd.read_csv(CSV_PATH)

# --- Lake Health Score Calculation Function ---
def calculate_lake_health_score(df,
                                vegetation_weight=1/6, barren_weight=1/6, urban_weight=1/6,
                                precipitation_weight=1/6, evaporation_weight=1/6, air_temperature_weight=1/6):

    for col in ['Vegetation Area', 'Barren Area', 'Urban Area', 'Precipitation', 'Evaporation', 'Air Temperature']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    latest_year_data = df[df['Year'] == df['Year'].max()].copy()
    if latest_year_data.empty:
        return pd.DataFrame()

    # Normalize latest year values
    latest_year_data['Vegetation Area Normalized'] = (latest_year_data['Vegetation Area'] - latest_year_data['Vegetation Area'].min()) / (latest_year_data['Vegetation Area'].max() - latest_year_data['Vegetation Area'].min() + 1e-9)
    latest_year_data['Barren Area Normalized'] = 1 - (latest_year_data['Barren Area'] - latest_year_data['Barren Area'].min()) / (latest_year_data['Barren Area'].max() - latest_year_data['Barren Area'].min() + 1e-9)
    latest_year_data['Urban Area Normalized'] = 1 - (latest_year_data['Urban Area'] - latest_year_data['Urban Area'].min()) / (latest_year_data['Urban Area'].max() - latest_year_data['Urban Area'].min() + 1e-9)
    latest_year_data['Precipitation Normalized'] = (latest_year_data['Precipitation'] - latest_year_data['Precipitation'].min()) / (latest_year_data['Precipitation'].max() - latest_year_data['Precipitation'].min() + 1e-9)
    latest_year_data['Evaporation Normalized'] = 1 - (latest_year_data['Evaporation'] - latest_year_data['Evaporation'].min()) / (latest_year_data['Evaporation'].max() - latest_year_data['Evaporation'].min() + 1e-9)
    latest_year_data['Air Temperature Normalized'] = 1 - (latest_year_data['Air Temperature'] - latest_year_data['Air Temperature'].min()) / (latest_year_data['Air Temperature'].max() - latest_year_data['Air Temperature'].min() + 1e-9)

    for col in latest_year_data.columns:
        if 'Normalized' in col:
            latest_year_data[col] = latest_year_data[col].replace([np.inf, -np.inf, np.nan], 0)

    # Calculate trends per lake
    trends = df.groupby('Lake').apply(lambda x: pd.Series({
        'Vegetation Area Trend': np.polyfit(x['Year'], x['Vegetation Area'], 1)[0],
        'Barren Area Trend': np.polyfit(x['Year'], x['Barren Area'], 1)[0],
        'Urban Area Trend': np.polyfit(x['Year'], x['Urban Area'], 1)[0],
        'Precipitation Trend': np.polyfit(x['Year'], x['Precipitation'], 1)[0],
        'Evaporation Trend': np.polyfit(x['Year'], x['Evaporation'], 1)[0],
        'Air Temperature Trend': np.polyfit(x['Year'], x['Air Temperature'], 1)[0],
    }))

    # Normalize trends
    for col in trends.columns:
        if 'Trend' in col:
            if any(x in col for x in ['Barren', 'Urban', 'Evaporation', 'Air Temperature']):
                trends[col + ' Normalized'] = 1 - (trends[col] - trends[col].min()) / (trends[col].max() - trends[col].min() + 1e-9)
            else:
                trends[col + ' Normalized'] = (trends[col] - trends[col].min()) / (trends[col].max() - trends[col].min() + 1e-9)
            trends[col + ' Normalized'] = trends[col + ' Normalized'].replace([np.inf, -np.inf, np.nan], 0)

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

# --- Generate time series plots for each metric ---
def generate_metric_time_series_plots_per_lake(df, lake_ids, metrics):
    images = []
    for lake in lake_ids:
        lake_df = df[df['Lake'].astype(str) == str(lake)].copy()
        if lake_df.empty:
            continue
        plt.figure(figsize=(10, 5), dpi=150)
        for metric in metrics:
            # Convert column to numeric, coerce errors to NaN
            lake_df[metric] = pd.to_numeric(lake_df[metric], errors='coerce')

            # If the metric is all NaN or empty, skip plotting it
            if lake_df[metric].dropna().empty:
                continue
            
            plt.plot(lake_df['Year'], lake_df[metric], marker='o', label=metric)
        
        plt.title(f"Time Series for Lake {lake}")
        plt.xlabel("Year")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        
        buf = BytesIO()
        plt.savefig(buf, format='PNG')
        plt.close()
        buf.seek(0)
        images.append((lake, buf))
    return images

# --- AI insight generation for all lakes combined ---
def generate_ai_insight_combined(prompt):
    API_KEY = st.secrets["OPENROUTER_API_KEY"]
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek/deepseek-chat:free",
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(API_URL, json=data, headers=headers)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return "Failed to generate insight."

# --- PDF Report Generation ---
def generate_comparative_pdf_report(df, results, lake_ids):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 50

    def writeln(text, step=20):
        nonlocal y
        for line in text.split('\n'):
            if y < 100:
                c.showPage()
                y = height - 50
            while len(line) > 110:
                c.drawString(40, y, line[:110])
                line = line[110:]
                y -= step
                if y < 100:
                    c.showPage()
                    y = height - 50
            c.drawString(40, y, line)
            y -= step

    # Title page
    writeln("Lake Health Comparative Report")
    writeln("="*60)
    writeln(f"Comparing Lakes: {', '.join(map(str,lake_ids))}")
    writeln("")

    # Summary table of health scores & ranks
    writeln("Summary of Health Scores and Ranks:")
    for _, row in results.iterrows():
        writeln(f"{row['Lake']}: Health Score = {row['Health Score']:.2f}, Rank = {int(row['Rank'])}")
    writeln("")

    # Build AI prompt for combined comparative analysis
    combined_prompt = "Provide a detailed comparative analysis for lakes: " + ", ".join(map(str,lake_ids)) + ".\n"
    for _, row in results.iterrows():
        combined_prompt += (f"Lake {row['Lake']} has a health score of {row['Health Score']:.2f} and rank {int(row['Rank'])}.\n")
    combined_prompt += "Discuss the values and trends of Vegetation Area, Barren Area, Urban Area, Precipitation, Evaporation, and Air Temperature for these lakes."

    ai_text = generate_ai_insight_combined(combined_prompt)
    writeln("AI-Generated Comparative Analysis:")
    writeln("-" * 40)
    writeln(ai_text)
    writeln("")

    # Generate and add plots per metric
    metrics = ['Vegetation Area', 'Barren Area', 'Urban Area', 'Precipitation', 'Evaporation', 'Air Temperature']
    plots = generate_metric_time_series_plots(df, lake_ids, metrics)

    for metric, img_buffer in plots:
        c.showPage()
        img = ImageReader(img_buffer)
        max_width = width - 100
        max_height = height - 200
        c.drawString(40, height - 50, f"Time Series Plot for {metric}")
        c.drawImage(img, 50, 100, width=max_width, height=max_height, preserveAspectRatio=True, mask='auto')

    c.save()
    buffer.seek(0)
    return buffer

# --- Streamlit App ---
st.title("Lake Health Score Dashboard")

# Load data automatically
df = load_data()
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Select lakes to analyze
lake_ids = st.text_input("Enter Lake IDs separated by commas (e.g. 630,2168266,737797)").replace(" ", "").split(",")
lake_ids = [x for x in lake_ids if x]  # Remove empty strings

if lake_ids:
    # Filter dataframe for selected lakes
    selected_df = df[df["Lake"].astype(str).isin(lake_ids)]
    if selected_df.empty:
        st.error("No data found for the entered Lake IDs.")
    else:
        results = calculate_lake_health_score(selected_df)
        if results.empty:
            st.error("No data available for latest year to calculate scores.")
        else:
            st.subheader("Lake Health Scores")
            st.dataframe(results[["Lake", "Health Score", "Rank"]])

            # Download CSV of selected lakes data
            csv = selected_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Original CSV for Selected Lakes", csv, "selected_lake_data.csv", "text/csv")

            # Download combined PDF report
            st.subheader("Download Combined PDF Report")
            pdf_buffer = generate_comparative_pdf_report(selected_df, results, lake_ids)
            st.download_button(
                label="📄 Download Combined Lake Health Report",
                data=pdf_buffer,
                file_name="combined_lake_health_report.pdf",
                mime="application/pdf"
            )
else:
    st.info("Enter at least one Lake ID above to get started.")
