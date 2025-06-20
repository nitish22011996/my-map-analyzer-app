# Full Streamlit App: Lake Health with Hierarchical Selection, Health Scores, Plots, PDF, and AI Analysis

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import requests
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from scipy.stats import linregress

# --- Load Data ---
@st.cache_data
def load_lake_data():
    return pd.read_csv("lake_health_data.csv")

@st.cache_data
def load_mapping_data():
    df = pd.read_csv("HDI_lake_district.csv")
    df.columns = df.columns.str.strip()  # Clean column names
    return df

lake_df = load_lake_data()
mapping_df = load_mapping_data()

# --- UI: Hierarchical Selection ---
st.title("Lake Health Score Dashboard")
st.subheader("Step 1: Select Lakes by Region")

states = sorted(mapping_df['State'].dropna().unique())
selected_state = st.selectbox("Select State", states)

filtered_districts = mapping_df[mapping_df['State'] == selected_state]
districts = sorted(filtered_districts['District'].dropna().unique())
selected_district = st.selectbox("Select District", districts)

filtered_lakes = mapping_df[(mapping_df['State'] == selected_state) &
                            (mapping_df['District'] == selected_district)]
lake_options = filtered_lakes[['Lake_ID']].drop_duplicates()
lake_names = lake_options['Lake_ID'].astype(str).tolist()
lake_ids_dict = dict(zip(lake_names, lake_names))

selected_lakes_by_name = st.multiselect("Select Lake(s)", lake_names)
selected_lake_ids = [str(lake_ids_dict[name]) for name in selected_lakes_by_name]

# --- Show Preview ---
st.subheader("Dataset Preview")
st.dataframe(lake_df.head())

# --- AI Insight Generation Function ---
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
    try:
        response = requests.post(API_URL, json=data, headers=headers, timeout=15)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Failed to generate insight: {response.status_code}"
    except Exception as e:
        return f"AI generation error: {str(e)}"

# --- Grouped Plot Generation ---
def generate_grouped_plots_by_metric(df, lake_ids, metrics):
    grouped_images = []
    for metric in metrics:
        plt.figure(figsize=(10, 6), dpi=150)
        for lake in lake_ids:
            lake_df = df[df['Lake'].astype(str) == str(lake)].copy()
            if lake_df.empty or metric not in lake_df:
                continue
            lake_df = lake_df.sort_values("Year")
            lake_df[metric] = pd.to_numeric(lake_df[metric], errors='coerce')
            plt.plot(lake_df["Year"], lake_df[metric], marker='o', label=f"Lake {lake}")
            if lake_df[metric].notna().sum() > 1:
                x = lake_df["Year"]
                y = lake_df[metric]
                slope, intercept, *_ = linregress(x, y)
                plt.plot(x, intercept + slope * x, linestyle='--', alpha=0.6)
        plt.title(f"{metric} Over Time")
        plt.xlabel("Year")
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)

        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        grouped_images.append((metric, buf))
    return grouped_images

# --- PDF Generation ---
def generate_comparative_pdf_report(df, results, lake_ids):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 60

    def writeln(text, step=18):
        nonlocal y
        for line in text.split('\n'):
            while len(line) > 100:
                c.drawString(40, y, line[:100])
                y -= step
                line = line[100:]
                if y < 80:
                    c.showPage()
                    y = height - 60
            c.drawString(40, y, line)
            y -= step
            if y < 80:
                c.showPage()
                y = height - 60

    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width / 2, y, "Lake Health Comparative Report")
    y -= 40
    c.setFont("Helvetica", 12)
    writeln(f"Lakes compared: {', '.join(lake_ids)}")
    y -= 10
    c.showPage()
    y = height - 50

    writeln("Health Score Rankings (Color-coded):")
    bar_start_x = 80
    bar_height = 16
    for i, row in results.iterrows():
        score = row['Health Score']
        rank = int(row['Rank'])
        color = colors.green if rank == 1 else colors.orange if rank == 2 else colors.red
        c.setFillColor(color)
        bar_width = score * 200
        c.rect(bar_start_x, y, bar_width, bar_height, fill=1)
        c.setFillColor(colors.black)
        c.drawString(bar_start_x + bar_width + 10, y + 2, f"Lake {row['Lake']} (Score: {score:.2f}, Rank: {rank})")
        y -= (bar_height + 10)
        if y < 100:
            c.showPage()
            y = height - 50

    prompt = "Compare lakes: " + ", ".join(lake_ids) + " based on their health scores and metric trends."
    for _, row in results.iterrows():
        prompt += f"\nLake {row['Lake']}: Score={row['Health Score']:.2f}, Rank={int(row['Rank'])}"
    ai_text = generate_ai_insight_combined(prompt)

    writeln("AI-Generated Comparative Analysis:\n" + "-"*40)
    writeln(ai_text)
    c.showPage()

    metrics = ['Vegetation Area', 'Barren Area', 'Urban Area', 'Precipitation', 'Evaporation', 'Air Temperature']
    plots = generate_grouped_plots_by_metric(df, lake_ids, metrics)

    for i in range(0, len(plots), 2):
        y_positions = [height / 2 + 20, 50]
        for j in range(2):
            if i + j >= len(plots): break
            metric, img_buf = plots[i + j]
            c.drawString(40, y_positions[j] + 270, f"{metric}")
            img = ImageReader(img_buf)
            c.drawImage(img, 50, y_positions[j], width=500, height=250, preserveAspectRatio=True, anchor='sw')
        c.showPage()

    c.save()
    buffer.seek(0)
    return buffer

# --- Health Score Calculation Function ---
def calculate_lake_health_score(df,
                                vegetation_weight=1/6, barren_weight=1/6, urban_weight=1/6,
                                precipitation_weight=1/6, evaporation_weight=1/6, air_temperature_weight=1/6):
    def norm(x): return (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0
    def rev_norm(x): return 1 - norm(x)

    for col in ['Vegetation Area', 'Barren Area', 'Urban Area', 'Precipitation', 'Evaporation', 'Air Temperature']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    latest_year_data = df[df['Year'] == df['Year'].max()].copy()
    if latest_year_data.empty:
        return pd.DataFrame()

    latest_year_data['Vegetation Area Normalized'] = norm(latest_year_data['Vegetation Area'])
    latest_year_data['Barren Area Normalized'] = rev_norm(latest_year_data['Barren Area'])
    latest_year_data['Urban Area Normalized'] = rev_norm(latest_year_data['Urban Area'])
    latest_year_data['Precipitation Normalized'] = norm(latest_year_data['Precipitation'])
    latest_year_data['Evaporation Normalized'] = rev_norm(latest_year_data['Evaporation'])
    latest_year_data['Air Temperature Normalized'] = rev_norm(latest_year_data['Air Temperature'])

    def get_slope_and_p(x, y):
        slope, _, _, p, _ = linregress(x, y)
        return slope, p

    def extract_trends(x):
        metrics = ['Vegetation Area', 'Barren Area', 'Urban Area', 'Precipitation', 'Evaporation', 'Air Temperature']
        data = {}
        for col in metrics:
            slope, p = get_slope_and_p(x['Year'], x[col])
            data[f"{col} Trend"] = slope
            data[f"{col} PValue"] = p
        return pd.Series(data)

    trends = df.groupby('Lake').apply(extract_trends).reset_index()

    for factor, desirable in [
        ('Vegetation Area', 'positive'),
        ('Barren Area', 'negative'),
        ('Urban Area', 'negative'),
        ('Precipitation', 'positive'),
        ('Evaporation', 'negative'),
        ('Air Temperature', 'negative')
    ]:
        slope_col = f"{factor} Trend"
        pval_col = f"{factor} PValue"
        trends[f"{slope_col} Normalized"] = norm(trends[slope_col]) if desirable == 'positive' else rev_norm(trends[slope_col])
        trends[f"{pval_col} Normalized"] = 1 - norm(trends[pval_col])

    latest_year_data = latest_year_data.set_index('Lake')
    trends = trends.set_index('Lake')
    combined = latest_year_data.join(trends, how='inner')

    def factor_score(factor, weight):
        return weight * (
            combined[f'{factor} Normalized'] +
            combined[f'{factor} Trend Normalized'] +
            combined[f'{factor} PValue Normalized']
        ) / 3

    combined['Health Score'] = (
        factor_score('Vegetation Area', vegetation_weight) +
        factor_score('Barren Area', barren_weight) +
        factor_score('Urban Area', urban_weight) +
        factor_score('Precipitation', precipitation_weight) +
        factor_score('Evaporation', evaporation_weight) +
        factor_score('Air Temperature', air_temperature_weight)
    )
    combined['Rank'] = combined['Health Score'].rank(ascending=False)
    return combined.reset_index()

# --- Main Execution ---
if selected_lake_ids:
    selected_df = lake_df[lake_df['Lake'].astype(str).isin(selected_lake_ids)]
    if selected_df.empty:
        st.error("No matching data found for selected lakes.")
    else:
        results = calculate_lake_health_score(selected_df)
        if not results.empty:
            st.subheader("Lake Health Scores")
            st.dataframe(results[['Lake', 'Health Score', 'Rank']])

            csv = selected_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Selected Lake Data", csv, "selected_lake_data.csv", "text/csv")

            pdf_buffer = generate_comparative_pdf_report(selected_df, results, selected_lake_ids)
            st.download_button(
                label="📄 Download Combined Lake Health Report",
                data=pdf_buffer,
                file_name="combined_lake_health_report.pdf",
                mime="application/pdf"
            )
        else:
            st.warning("No data available for the latest year.")
else:
    st.info("Please select at least one lake to continue.")
