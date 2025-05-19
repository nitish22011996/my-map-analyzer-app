import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import requests
import textwrap
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
import seaborn as sns
from scipy.stats import linregress

# --- Load CSV in background ---
CSV_PATH = "lake_health_data.csv"  # <-- change this to your actual CSV path

@st.cache_data
def load_data():
    return pd.read_csv(CSV_PATH)

# --- Helper: Parse fraction input strings ---
def parse_fraction(s):
    s = s.strip()
    if '/' in s:
        parts = s.split('/')
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return float(parts[0]) / float(parts[1])
    try:
        return float(s)
    except:
        return None

# --- Lake Health Score Calculation Function ---
def calculate_lake_health_score(df,
                                vegetation_weight=1/6, barren_weight=1/6, urban_weight=1/6,
                                precipitation_weight=1/6, evaporation_weight=1/6, air_temperature_weight=1/6):
    def norm(x): return (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0
    def rev_norm(x): return 1 - norm(x)

    required_columns = ['Lake', 'Year', 'Vegetation Area', 'Barren Area', 'Urban Area',
                        'Precipitation', 'Evaporation', 'Air Temperature']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    for col in ['Vegetation Area', 'Barren Area', 'Urban Area', 'Precipitation', 'Evaporation', 'Air Temperature']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    latest_year_data = df[df['Year'] == df['Year'].max()].copy()
    if latest_year_data.empty:
        return pd.DataFrame()

    # Normalize latest values
    latest_year_data['Vegetation Area Normalized'] = norm(latest_year_data['Vegetation Area'])
    latest_year_data['Barren Area Normalized'] = rev_norm(latest_year_data['Barren Area'])
    latest_year_data['Urban Area Normalized'] = rev_norm(latest_year_data['Urban Area'])
    latest_year_data['Precipitation Normalized'] = norm(latest_year_data['Precipitation'])
    latest_year_data['Evaporation Normalized'] = rev_norm(latest_year_data['Evaporation'])
    latest_year_data['Air Temperature Normalized'] = rev_norm(latest_year_data['Air Temperature'])

    for col in latest_year_data.columns:
        if 'Normalized' in col:
            latest_year_data[col] = latest_year_data[col].replace([np.inf, -np.inf, np.nan], 0)

    def get_slope_and_p(x, y):
        slope, _, _, p_value, _ = linregress(x, y)
        return slope, p_value

    trends = df.groupby('Lake').apply(lambda x: pd.Series({
        'Vegetation Area Trend': get_slope_and_p(x['Year'], x['Vegetation Area'])[0],
        'Vegetation Area PValue': get_slope_and_p(x['Year'], x['Vegetation Area'])[1],
        'Barren Area Trend': get_slope_and_p(x['Year'], x['Barren Area'])[0],
        'Barren Area PValue': get_slope_and_p(x['Year'], x['Barren Area'])[1],
        'Urban Area Trend': get_slope_and_p(x['Year'], x['Urban Area'])[0],
        'Urban Area PValue': get_slope_and_p(x['Year'], x['Urban Area'])[1],
        'Precipitation Trend': get_slope_and_p(x['Year'], x['Precipitation'])[0],
        'Precipitation PValue': get_slope_and_p(x['Year'], x['Precipitation'])[1],
        'Evaporation Trend': get_slope_and_p(x['Year'], x['Evaporation'])[0],
        'Evaporation PValue': get_slope_and_p(x['Year'], x['Evaporation'])[1],
        'Air Temperature Trend': get_slope_and_p(x['Year'], x['Air Temperature'])[0],
        'Air Temperature PValue': get_slope_and_p(x['Year'], x['Air Temperature'])[1],
    })).reset_index()

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
        trends[f"{pval_col} Normalized"] = 1 - norm(trends[pval_col])  # lower p-value is better

    for col in trends.columns:
        if 'Normalized' in col:
            trends[col] = trends[col].replace([np.inf, -np.inf, np.nan], 0)

    latest_year_data = latest_year_data.set_index('Lake')
    trends = trends.set_index('Lake')
    combined_data = latest_year_data.join(trends, how='inner')

    def factor_score(factor, weight):
        return weight * (
            (combined_data[f'{factor} Normalized'] +
             combined_data[f'{factor} Trend Normalized'] +
             combined_data[f'{factor} PValue Normalized']) / 3
        )

    combined_data['Health Score'] = (
        factor_score('Vegetation Area', vegetation_weight) +
        factor_score('Barren Area', barren_weight) +
        factor_score('Urban Area', urban_weight) +
        factor_score('Precipitation', precipitation_weight) +
        factor_score('Evaporation', evaporation_weight) +
        factor_score('Air Temperature', air_temperature_weight)
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
        for metric in metrics:
            plt.figure(figsize=(10, 5), dpi=150)
            lake_df[metric] = pd.to_numeric(lake_df[metric], errors='coerce')
            if lake_df[metric].dropna().empty:
                continue
            plt.plot(lake_df['Year'], lake_df[metric], marker='o', label=metric)
            plt.title(f"Time Series of {metric} for Lake {lake}")
            plt.xlabel("Year")
            plt.ylabel(metric)
            plt.legend()
            plt.grid(True)
            buf = BytesIO()
            plt.savefig(buf, format='PNG')
            plt.close()
            buf.seek(0)
            images.append((f"Lake {lake} - {metric}", buf))
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

# Function to generate individual plots per lake for each metric with trendlines
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
            # Trendline
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

# Final PDF generation function with weights shown as fractions
def generate_comparative_pdf_report(df, results, lake_ids, weights_dict):
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

    # Helper: Format float weight to fraction string
    def float_to_fraction(f, max_denominator=100):
        from fractions import Fraction
        frac = Fraction(f).limit_denominator(max_denominator)
        return f"{frac.numerator}/{frac.denominator}"

    # Title Page
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width / 2, y, "Lake Health Comparative Report")
    y -= 40
    c.setFont("Helvetica", 12)
    writeln(f"Lakes compared: {', '.join(lake_ids)}")
    y -= 10

    # Show weights used
    writeln("Weights used for Health Score Calculation:")
    for factor, val in weights_dict.items():
        frac_str = float_to_fraction(val)
        writeln(f" - {factor}: {frac_str} ({val:.3f})")
    y -= 20
    c.showPage()
    y = height - 50

    # Colored Health Ranking Bars
    writeln("Health Score Rankings (Color-coded):")
    bar_start_x = 80
    bar_width = 300
    max_score = results['Health Score'].max()

    for idx, row in results.sort_values('Rank').iterrows():
        lake = row['Lake']
        score = row['Health Score']
        rank = int(row['Rank'])
        bar_len = bar_width * score / max_score if max_score > 0 else 0

        # Color gradient: green to red by rank (1=green best)
        green_value = int(255 * (len(results) - rank) / max(1, len(results) - 1))
        red_value = 255 - green_value
        c.setFillColorRGB(red_value / 255, green_value / 255, 0)
        c.rect(bar_start_x, y - 12, bar_len, 12, fill=1)
        c.setFillColor(colors.black)
        c.drawString(40, y - 12, f"{lake} (Rank {rank}) Score: {score:.3f}")
        y -= 25
        if y < 100:
            c.showPage()
            y = height - 50

    c.showPage()
    y = height - 60

    # Insert AI insight placeholder - You can replace with real AI text
    ai_prompt = f"Generate comparative insights for lakes: {', '.join(lake_ids)} using weights: {weights_dict}"
    ai_text = generate_ai_insight_combined(ai_prompt)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "🤖 AI-Generated Comparative Insight")
    y -= 30
    c.setFont("Helvetica", 11)
    writeln(textwrap.fill(ai_text, width=90))

    # Historical Extremes Overview Section (sample)
    y -= 40
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "📉 Historical Extremes Overview")
    y -= 25
    c.setFont("Helvetica", 10)
    for lake in lake_ids:
        lake_df = df[df['Lake'].astype(str) == str(lake)]
        if lake_df.empty:
            continue
        writeln(f"Lake {lake}:")

        # For each factor: lowest and highest value with year and % difference from latest
        factors = ['Vegetation Area', 'Barren Area', 'Urban Area', 'Precipitation', 'Evaporation', 'Air Temperature']
        latest_year = lake_df['Year'].max()
        latest_row = lake_df[lake_df['Year'] == latest_year]
        if latest_row.empty:
            continue

        for factor in factors:
            if factor not in lake_df:
                continue
            min_val = lake_df[factor].min()
            min_year = lake_df.loc[lake_df[factor].idxmin(), 'Year']
            max_val = lake_df[factor].max()
            max_year = lake_df.loc[lake_df[factor].idxmax(), 'Year']
            latest_val = latest_row.iloc[0][factor]

            # Calculate percentage difference relative to latest value, avoid division by zero
            def pct_diff(old, new):
                try:
                    return ((new - old) / old) * 100 if old != 0 else 0
                except:
                    return 0

            low_diff = pct_diff(min_val, latest_val)
            high_diff = pct_diff(max_val, latest_val)

            writeln(f"  - {factor}:")
            writeln(f"    * Lowest: {min_val:.3f} in {min_year} ({low_diff:+.1f}% vs latest)")
            writeln(f"    * Highest: {max_val:.3f} in {max_year} ({high_diff:+.1f}% vs latest)")
            y -= 5
            if y < 80:
                c.showPage()
                y = height - 60
        y -= 15
        if y < 100:
            c.showPage()
            y = height - 60

    c.save()
    buffer.seek(0)
    return buffer

# --- Streamlit UI ---

st.title("Lake Health Analysis with Fractional Weights")

# Load data
df = load_data()

# User inputs for lake IDs
lake_ids_input = st.text_input("Enter lake IDs separated by commas (e.g. 630,2168266,737797):", "630,2168266")
lake_ids = [x.strip() for x in lake_ids_input.split(",") if x.strip() != ""]

# Default weights (equal weights)
default_weights = {
    'Vegetation Area': 1/6,
    'Barren Area': 1/6,
    'Urban Area': 1/6,
    'Precipitation': 1/6,
    'Evaporation': 1/6,
    'Air Temperature': 1/6
}

st.markdown("### Enter weights for each factor as fractions (e.g. 1/6) or decimals (e.g. 0.1667).")
st.markdown("Leave blank to use equal weights (1/6).")

# Input fields for weights
weights = {}
cols = st.columns(3)
factor_list = list(default_weights.keys())
for i, factor in enumerate(factor_list):
    with cols[i % 3]:
        val = st.text_input(f"Weight for {factor}:", value=str(float_to_fraction(default_weights[factor])))
        weights[factor] = val.strip()

def float_to_fraction(f, max_denominator=100):
    from fractions import Fraction
    frac = Fraction(f).limit_denominator(max_denominator)
    return f"{frac.numerator}/{frac.denominator}"

# Button to submit weights and calculate
if st.button("Submit Weights and Calculate Scores"):

    # Parse weights
    parsed_weights = {}
    error_in_weights = False
    total_weight = 0
    for factor, val in weights.items():
        if val == "":
            parsed_val = default_weights[factor]
        else:
            parsed_val = parse_fraction(val)
            if parsed_val is None or parsed_val < 0 or parsed_val > 1:
                st.error(f"Invalid weight for {factor}: {val}. Please enter a fraction or decimal between 0 and 1.")
                error_in_weights = True
                break
        parsed_weights[factor] = parsed_val
        total_weight += parsed_val

    if not error_in_weights:
        # Normalize weights so they sum to 1 if sum != 1
        if abs(total_weight - 1) > 1e-6:
            parsed_weights = {k: v/total_weight for k,v in parsed_weights.items()}
            st.info(f"Weights normalized to sum to 1.")

        # Calculate lake health score
        results = calculate_lake_health_score(
            df,
            vegetation_weight=parsed_weights['Vegetation Area'],
            barren_weight=parsed_weights['Barren Area'],
            urban_weight=parsed_weights['Urban Area'],
            precipitation_weight=parsed_weights['Precipitation'],
            evaporation_weight=parsed_weights['Evaporation'],
            air_temperature_weight=parsed_weights['Air Temperature']
        )

        # Filter results by lake IDs user entered
        results = results[results['Lake'].astype(str).isin(lake_ids)]

        if results.empty:
            st.warning("No data available for the selected lakes or latest year.")
        else:
            st.success("Lake health scores calculated!")

            st.dataframe(results[['Lake', 'Health Score', 'Rank']].sort_values('Rank'))

            # Show time series plots for selected lakes and factors
            metrics = ['Vegetation Area', 'Barren Area', 'Urban Area', 'Precipitation', 'Evaporation', 'Air Temperature']
            grouped_plots = generate_grouped_plots_by_metric(df, lake_ids, metrics)
            for metric, img_buf in grouped_plots:
                st.image(img_buf, caption=f"{metric} Over Time", use_column_width=True)

            # Generate and offer PDF report
            pdf_buffer = generate_comparative_pdf_report(df, results, lake_ids, parsed_weights)
            st.download_button("Download PDF Report", data=pdf_buffer, file_name="lake_health_report.pdf", mime="application/pdf")


