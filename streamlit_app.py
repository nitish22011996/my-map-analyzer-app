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
from fractions import Fraction

# Utility: Convert float to fraction string (approximate)
def float_to_fraction_str(x, max_denominator=20):
    try:
        frac = Fraction(x).limit_denominator(max_denominator)
        return f"{frac.numerator}/{frac.denominator}"
    except Exception:
        return str(round(x, 3))

# --- Lake Health Score Calculation Function (use passed weights) ---
def calculate_lake_health_score(df,
                                vegetation_weight, barren_weight, urban_weight,
                                precipitation_weight, evaporation_weight, air_temperature_weight):
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

    def get_slope_and_p(x, y):
        if len(x.unique()) <= 1:
            return 0, 1  # No trend if single year or no variation
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
        trends[f"{pval_col} Normalized"] = 1 - norm(trends[pval_col])  # lower p-value better

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

# --- AI Insight function ---
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

# --- PDF generation with weights and AI text ---
def generate_comparative_pdf_report(df, results, lake_ids, weights, ai_text):
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

    # Title Page
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width / 2, y, "Lake Health Comparative Report")
    y -= 40
    c.setFont("Helvetica", 12)
    writeln(f"Lakes compared: {', '.join(lake_ids)}")
    y -= 10

    # Weights used
    writeln("Weights used to calculate Lake Health Score (approximate fractions):")
    for factor, val in weights.items():
        frac_str = float_to_fraction_str(val)
        writeln(f"- {factor}: {frac_str} ({val:.3f})")
    y -= 10
    c.showPage()
    y = height - 50

    # Colored Health Ranking Bars
    writeln("Health Score Rankings (Color-coded):")
    bar_start_x = 80
    bar_height = 16
    for i, row in results.iterrows():
        score = row['Health Score']
        rank = int(row['Rank'])
        color = colors.green if rank == 1 else colors.orange if rank == 2 else colors.red
        c.setFillColor(color)
        bar_width = score * 200  # Scale bar width
        c.rect(bar_start_x, y, bar_width, bar_height, fill=1)
        c.setFillColor(colors.black)
        c.drawString(bar_start_x + bar_width + 10, y + 2, f"Lake {row['Lake']} (Score: {score:.2f}, Rank: {rank})")
        y -= (bar_height + 10)
        if y < 100:
            c.showPage()
            y = height - 50

    # AI Generated Insight Section
    writeln("AI-Generated Comparative Analysis:\n" + "-"*40)
    writeln(ai_text)
    c.showPage()

    # You can add plots here similarly if needed...

    c.save()
    buffer.seek(0)
    return buffer

# --- Streamlit App ---

st.title("Lake Health Score Dashboard")

# Load your CSV
@st.cache_data
def load_data():
    return pd.read_csv("lake_health_data.csv")

df = load_data()
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Sidebar weights with fraction input
st.sidebar.header("Adjust Factor Weights (Sum must be 1.0)")

def set_equal_weights():
    for factor in weight_keys:
        st.session_state[factor] = 1/6

weight_keys = ['vegetation_weight', 'barren_weight', 'urban_weight', 'precipitation_weight', 'evaporation_weight', 'air_temperature_weight']

for key in weight_keys:
    if key not in st.session_state:
        st.session_state[key] = 1/6  # default equal weight

if st.sidebar.button("Set All Weights Equal"):
    set_equal_weights()

def fraction_input(label, key):
    val = st.sidebar.text_input(label + " (fraction, e.g. 1/6)", value=float_to_fraction_str(st.session_state[key]), key=key+"_input")
    try:
        # Parse fraction string like "1/6" to float
        st.session_state[key] = float(Fraction(val))
    except Exception:
        st.sidebar.error(f"Invalid fraction input for {label}. Using previous value.")
    return st.session_state[key]

veg_w = fraction_input("Vegetation Area Weight", "vegetation_weight")
barr_w = fraction_input("Barren Area Weight", "barren_weight")
urb_w = fraction_input("Urban Area Weight", "urban_weight")
precip_w = fraction_input("Precipitation Weight", "precipitation_weight")
evap_w = fraction_input("Evaporation Weight", "evaporation_weight")
temp_w = fraction_input("Air Temperature Weight", "air_temperature_weight")

total_weight = veg_w + barr_w + urb_w + precip_w + evap_w + temp_w
if abs(total_weight - 1.0) > 1e-3:
    st.sidebar.warning(f"Total weight is {total_weight:.3f}, should be exactly 1.0.")

lake_ids = st.text_input("Enter comma-separated Lake IDs for analysis (e.g. Lake1,Lake2)")

if st.button("Calculate and Generate Report") and lake_ids:
    selected_ids = [x.strip() for x in lake_ids.split(",") if x.strip()]

    filtered_df = df[df['Lake'].isin(selected_ids)]
    if filtered_df.empty:
        st.error("No data for given Lake IDs")
    else:
        results = calculate_lake_health_score(filtered_df,
                                              veg_w, barr_w, urb_w,
                                              precip_w, evap_w,
                                              temp_w)
        st.subheader("Lake Health Scores")
        st.dataframe(results)

        # Compose prompt for AI insight
        prompt = f"Compare these lakes based on health scores and factor weights:\n{results.to_string(index=False)}\nWeights used:\nVegetation Area: {float_to_fraction_str(veg_w)}\nBarren Area: {float_to_fraction_str(barr_w)}\nUrban Area: {float_to_fraction_str(urb_w)}\nPrecipitation: {float_to_fraction_str(precip_w)}\nEvaporation: {float_to_fraction_str(evap_w)}\nAir Temperature: {float_to_fraction_str(temp_w)}"

        with st.spinner("Generating AI insights..."):
            ai_text = generate_ai_insight_combined(prompt)
            st.markdown("### AI-Generated Insight")
            st.write(ai_text)

        pdf_buffer = generate_comparative_pdf_report(filtered_df, results, selected_ids,
                                                     {
                                                        "Vegetation Area": veg_w,
                                                        "Barren Area": barr_w,
                                                        "Urban Area": urb_w,
                                                        "Precipitation": precip_w,
                                                        "Evaporation": evap_w,
                                                        "Air Temperature": temp_w,
                                                     },
                                                     ai_text)

        st.download_button(
            label="Download PDF Report",
            data=pdf_buffer,
            file_name="lake_health_report.pdf",
            mime="application/pdf"
        )
else:
    st.info("Enter Lake IDs and click 'Calculate and Generate Report'")

