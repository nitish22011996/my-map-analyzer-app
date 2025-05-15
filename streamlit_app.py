import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
import openai

# --- Lake Health Score Calculation Function ---
def calculate_lake_health_score(df,
                                vegetation_weight=1/6, barren_weight=1/6, urban_weight=1/6,
                                precipitation_weight=1/6, evaporation_weight=1/6, air_temperature_weight=1/6):

    for col in ['Vegetation Area', 'Barren Area', 'Urban Area', 'Precipitation', 'Evaporation', 'Air Temperature']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    latest_year_data = df[df['Year'] == df['Year'].max()].copy()
    if latest_year_data.empty:
        return pd.DataFrame()

    latest_year_data['Vegetation Area Normalized'] = (latest_year_data['Vegetation Area'] - latest_year_data['Vegetation Area'].min()) / (latest_year_data['Vegetation Area'].max() - latest_year_data['Vegetation Area'].min())
    latest_year_data['Barren Area Normalized'] = 1 - (latest_year_data['Barren Area'] - latest_year_data['Barren Area'].min()) / (latest_year_data['Barren Area'].max() - latest_year_data['Barren Area'].min())
    latest_year_data['Urban Area Normalized'] = 1 - (latest_year_data['Urban Area'] - latest_year_data['Urban Area'].min()) / (latest_year_data['Urban Area'].max() - latest_year_data['Urban Area'].min())
    latest_year_data['Precipitation Normalized'] = (latest_year_data['Precipitation'] - latest_year_data['Precipitation'].min()) / (latest_year_data['Precipitation'].max() - latest_year_data['Precipitation'].min())
    latest_year_data['Evaporation Normalized'] = 1 - (latest_year_data['Evaporation'] - latest_year_data['Evaporation'].min()) / (latest_year_data['Evaporation'].max() - latest_year_data['Evaporation'].min())
    latest_year_data['Air Temperature Normalized'] = 1 - (latest_year_data['Air Temperature'] - latest_year_data['Air Temperature'].min()) / (latest_year_data['Air Temperature'].max() - latest_year_data['Air Temperature'].min())

    for col in latest_year_data.columns:
        if 'Normalized' in col:
            latest_year_data[col] = latest_year_data[col].replace([np.inf, -np.inf, np.nan], 0)

    trends = df.groupby('Lake').apply(lambda x: pd.Series({
        'Vegetation Area Trend': np.polyfit(x['Year'], x['Vegetation Area'], 1)[0],
        'Barren Area Trend': np.polyfit(x['Year'], x['Barren Area'], 1)[0],
        'Urban Area Trend': np.polyfit(x['Year'], x['Urban Area'], 1)[0],
        'Precipitation Trend': np.polyfit(x['Year'], x['Precipitation'], 1)[0],
        'Evaporation Trend': np.polyfit(x['Year'], x['Evaporation'], 1)[0],
        'Air Temperature Trend': np.polyfit(x['Year'], x['Air Temperature'], 1)[0],
    }))

    for col in trends.columns:
        if 'Trend' in col:
            if 'Barren' in col or 'Urban' in col or 'Evaporation' in col or 'Air Temperature' in col:
                trends[col + ' Normalized'] = 1 - (trends[col] - trends[col].min()) / (trends[col].max() - trends[col].min())
            else:
                trends[col + ' Normalized'] = (trends[col] - trends[col].min()) / (trends[col].max() - trends[col].min())
            trends[col + ' Normalized'] = trends[col + ' Normalized'].replace([np.inf, -np.inf, np.nan], 0)

    latest_year_data = latest_year_data.set_index('Lake')
    combined_data = latest_year_data.join(trends, how='inner')

    combined_data['Health Score'] = sum([
        vegetation_weight * combined_data['Vegetation Area Normalized'],
        barren_weight * combined_data['Barren Area Normalized'],
        urban_weight * combined_data['Urban Area Normalized'],
        precipitation_weight * combined_data['Precipitation Normalized'],
        evaporation_weight * combined_data['Evaporation Normalized'],
        air_temperature_weight * combined_data['Air Temperature Normalized'],
        vegetation_weight * combined_data['Vegetation Area Trend Normalized'],
        barren_weight * combined_data['Barren Area Trend Normalized'],
        urban_weight * combined_data['Urban Area Trend Normalized'],
        precipitation_weight * combined_data['Precipitation Trend Normalized'],
        evaporation_weight * combined_data['Evaporation Trend Normalized'],
        air_temperature_weight * combined_data['Air Temperature Trend Normalized']
    ])

    combined_data['Rank'] = combined_data['Health Score'].rank(ascending=False)
    return combined_data.reset_index()

# --- Generate Plot ---
def generate_lake_plot(lake_name, lake_data, comparison_data):
    metrics = ['Vegetation Area', 'Barren Area', 'Urban Area', 'Precipitation', 'Evaporation', 'Air Temperature']
    trends = [m + ' Trend' for m in metrics]

    fig, axs = plt.subplots(2, 1, figsize=(6, 6), dpi=150)
    lake_vals = [lake_data[m] for m in metrics]
    comp_avg_vals = [comparison_data[m].mean() for m in metrics]

    axs[0].bar(metrics, lake_vals, label=f"{lake_name}", color="skyblue")
    axs[0].plot(metrics, comp_avg_vals, label="Average", color="red", linestyle="--", marker="o")
    axs[0].set_title("Current Metric Comparison")
    axs[0].tick_params(axis='x', rotation=45)
    axs[0].legend()

    lake_trends = [lake_data[t] for t in trends]
    comp_trends = [comparison_data[t].mean() for t in trends]

    axs[1].bar(metrics, lake_trends, label=f"{lake_name} Trend", color="green")
    axs[1].plot(metrics, comp_trends, label="Average Trend", color="orange", linestyle="--", marker="o")
    axs[1].set_title("Trend Comparison")
    axs[1].tick_params(axis='x', rotation=45)
    axs[1].legend()

    plt.tight_layout()

    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='PNG')
    plt.close()
    img_buffer.seek(0)
    return img_buffer

from openai import OpenAI


def generate_ai_insight(lake_name, lake_data):
    API_KEY = st.secrets["OPENROUTER_API_KEY"]
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    prompt = f"""
    Provide a comparative analysis of why Lake {lake_name} has a health score of {lake_data['Health Score']:.2f} 
    and is ranked {int(lake_data['Rank'])} among the lakes. Include values and trends for:

    Vegetation Area: {lake_data['Vegetation Area']} (Trend: {lake_data.get('Vegetation Area Trend', 0)}),
    Barren Area: {lake_data['Barren Area']} (Trend: {lake_data.get('Barren Area Trend', 0)}),
    Urban Area: {lake_data['Urban Area']} (Trend: {lake_data.get('Urban Area Trend', 0)}),
    Precipitation: {lake_data['Precipitation']} (Trend: {lake_data.get('Precipitation Trend', 0)}),
    Evaporation: {lake_data['Evaporation']} (Trend: {lake_data.get('Evaporation Trend', 0)}),
    Air Temperature: {lake_data['Air Temperature']} (Trend: {lake_data.get('Air Temperature Trend', 0)}).

    Write a detailed report in paragraph form comparing this lake to others.
    """

    response = client.chat.completions.create(
        model="gpt-4",  # or "gpt-3.5-turbo"
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=600
    )
    return response.choices[0].message.content.strip()
# --- PDF Report Generation ---
def generate_pdf_report(lake_name, lake_data, comparison_data):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 50

    def writeln(text, step=20):
        nonlocal y
        # Split long texts into lines and paginate if needed
        for line in text.split('\n'):
            if y < 100:
                c.showPage()
                y = height - 50
            # Wrap text if longer than ~110 chars (basic handling)
            while len(line) > 110:
                c.drawString(40, y, line[:110])
                line = line[110:]
                y -= step
                if y < 100:
                    c.showPage()
                    y = height - 50
            c.drawString(40, y, line)
            y -= step

    # Title & Header
    writeln(f"Lake Health Report - {lake_name}")
    writeln("=" * 60)
    writeln(f"Health Score: {lake_data['Health Score']:.2f} | Rank: {int(lake_data['Rank'])}")
    writeln("")

    # AI Generated Comparative Analysis
    ai_text = generate_ai_insight(lake_name, lake_data)
    writeln("AI-Generated Comparative Analysis:")
    writeln("-" * 40)
    writeln(ai_text)
    writeln("")

    # Add a new page for the plot
    c.showPage()
    img_buffer = generate_lake_plot(lake_name, lake_data, comparison_data)
    img = ImageReader(img_buffer)

    # Calculate image dimensions to fit width with aspect ratio
    max_width = width - 100
    max_height = height - 200
    c.drawImage(img, 50, 100, width=max_width, height=max_height, preserveAspectRatio=True, mask='auto')

    c.save()
    buffer.seek(0)
    return buffer
# --- Streamlit App ---
st.title("Lake Health Score Dashboard")
df = pd.read_csv("lake_health_data.csv")

num_lakes = st.number_input("How many lakes do you want to compare?", min_value=1, max_value=10, step=1)
lake_ids = []
for i in range(num_lakes):
    lake_id = st.text_input(f"Enter Lake ID #{i + 1}", key=f"lake_{i}")
    if lake_id:
        lake_ids.append(lake_id)

if lake_ids:
    selected_df = df[df["Lake"].astype(str).isin(lake_ids)]

    if selected_df.empty:
        st.error("No data found for the entered Lake IDs.")
    else:
        results = calculate_lake_health_score(selected_df)

        st.subheader("Lake Health Scores")
        st.dataframe(results[["Lake", "Health Score", "Rank"]])

        csv = selected_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Original CSV for Selected Lakes", csv, "selected_lake_data.csv", "text/csv")

        st.subheader("Download PDF Reports")
        for i, row in results.iterrows():
            lake_data = row
            pdf_buffer = generate_pdf_report(row['Lake'], lake_data, results)
            st.download_button(
                label=f"📄 Download Report for {row['Lake']}",
                data=pdf_buffer,
                file_name=f"{row['Lake']}_health_report.pdf",
                mime="application/pdf"
            )
