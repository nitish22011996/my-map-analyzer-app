import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from scipy.stats import linregress

# --- PDF & Map Specific Imports ---
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors

# --- CONFIGURATION ---
LOCATION_DATA_PATH = 'HDI_lake_district.csv'
HEALTH_DATA_PATH = "lake_health_data.csv"

# --- PARAMETER DICTIONARY ---
# **FIX**: Added 'Area' and ensured all parameters are recognized.
PARAMETER_PROPERTIES = {
    'Air Temperature': {'impact': 'negative', 'type': 'climate'},
    'Evaporation': {'impact': 'negative', 'type': 'climate'},
    'Precipitation': {'impact': 'positive', 'type': 'climate'},
    'Lake Water Surface Temperature': {'impact': 'negative', 'type': 'water_quality'},
    'Water Clarity': {'impact': 'positive', 'type': 'water_quality'},
    'Barren Area': {'impact': 'negative', 'type': 'land_cover'},
    'Urban and Vegetation Area': {'impact': 'negative', 'type': 'land_cover'},
    'HDI': {'impact': 'positive', 'type': 'socioeconomic'},
    'Area': {'impact': 'positive', 'type': 'physical'}
}
LAND_COVER_INTERNAL_COLS = ['Barren Area', 'Urban and Vegetation Area']


# --- FUNCTION DEFINITIONS ---

@st.cache_data
def prepare_all_data(health_path, location_path):
    """
    Loads, cleans, merges data, and dynamically discovers which parameters are
    actually available in the loaded files for the UI.
    """
    try:
        df_health = pd.read_csv(health_path)
        df_location = pd.read_csv(location_path)
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}.")
        return None, None, None

    # **FIX**: Expanded map to include 'Area' and both 'Water_Clarity' variations.
    health_col_map = {
        'Air_Temperature': 'Air Temperature', 'Evaporation': 'Evaporation', 'Precipitation': 'Precipitation',
        'Barren': 'Barren Area', 'Urban and Vegetation': 'Urban and Vegetation Area',
        'Lake_Water_Surface_Temperature': 'Lake Water Surface Temperature', 
        'Water_Clarity': 'Water Clarity', 'Water_Clarity(FUI)': 'Water Clarity', # Handles both possible names
        'Area': 'Area'
    }
    df_health = df_health.rename(columns=health_col_map)

    potential_id_cols = ['Lake_ID', 'Lake_id', 'lake_id']
    health_id_col = next((col for col in potential_id_cols if col in df_health.columns), None)
    loc_id_col = next((col for col in potential_id_cols if col in df_location.columns), None)
    if not health_id_col or not loc_id_col:
        st.error(f"Critical Error: Could not find a lake identifier column in one or both CSV files.")
        return None, None, None
    df_health = df_health.rename(columns={health_id_col: 'Lake_ID'})
    df_location = df_location.rename(columns={loc_id_col: 'Lake_ID'})

    df_health['Lake_ID'] = pd.to_numeric(df_health['Lake_ID'], errors='coerce').dropna().astype(int)
    df_location['Lake_ID'] = pd.to_numeric(df_location['Lake_ID'], errors='coerce').dropna().astype(int)
    location_subset = df_location[['Lake_ID', 'HDI']].copy()
    df_merged = pd.merge(df_health, location_subset, on='Lake_ID', how='left')

    available_data_cols = [col for col in df_merged.columns if col in PARAMETER_PROPERTIES]
    ui_options = [p for p in available_data_cols if p not in LAND_COVER_INTERNAL_COLS]
    if any(p in available_data_cols for p in LAND_COVER_INTERNAL_COLS):
        ui_options.append("Land Cover")
    if not ui_options:
        st.error("No valid analysis parameters found in the data files.")
        return None, None, None

    return df_merged, df_location, sorted(ui_options)


def get_effective_weights(selected_ui_options, available_data_cols):
    """
    **NEW FUNCTION**: Centralizes the hierarchical weight calculation logic.
    This is the core fix for the weighting bug.
    """
    effective_weights = {}
    num_main_groups = len(selected_ui_options)
    w_main = 1.0 / num_main_groups if num_main_groups > 0 else 0.0

    if "Land Cover" in selected_ui_options:
        # Find which land cover columns are actually available in the data
        available_lc_cols_in_data = [p for p in LAND_COVER_INTERNAL_COLS if p in available_data_cols]
        num_land_cover_items = len(available_lc_cols_in_data)
        w_sub_landcover = 1.0 / num_land_cover_items if num_land_cover_items > 0 else 0.0
        
        for lc_param in available_lc_cols_in_data:
            effective_weights[lc_param] = w_main * w_sub_landcover
    
    for param in selected_ui_options:
        if param != "Land Cover":
            effective_weights[param] = w_main
            
    return effective_weights


def calculate_lake_health_score(df, selected_ui_options):
    if not selected_ui_options: return pd.DataFrame()
    def norm(x): return (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
    def rev_norm(x): return 1.0 - norm(x)

    # **FIX**: Use the new, correct weighting function.
    final_weights = get_effective_weights(selected_ui_options, df.columns)
    params_to_process = list(final_weights.keys())

    latest_year_data = df.loc[df.groupby('Lake_ID')['Year'].idxmax()].copy().set_index('Lake_ID')
    total_score = pd.Series(0.0, index=latest_year_data.index)

    for param in params_to_process:
        props = PARAMETER_PROPERTIES[param]
        df[param] = pd.to_numeric(df[param], errors='coerce').fillna(0)
        
        latest_values = latest_year_data[param]
        present_value_score = norm(latest_values) if props['impact'] == 'positive' else rev_norm(latest_values)
        
        if param == 'HDI':
            factor_score = present_value_score
        else:
            trends = df.groupby('Lake_ID').apply(lambda x: linregress(x['Year'], x[param]) if len(x['Year'].unique()) > 1 else (0,0,0,1,0))
            slopes = trends.apply(lambda x: x.slope); p_values = trends.apply(lambda x: x.pvalue)
            slope_norm = norm(slopes) if props['impact'] == 'positive' else rev_norm(slopes)
            p_value_norm = 1.0 - norm(p_values)
            factor_score = (present_value_score + slope_norm + p_value_norm) / 3.0
        
        total_score += final_weights[param] * factor_score

    latest_year_data['Health Score'] = total_score
    latest_year_data['Rank'] = latest_year_data['Health Score'].rank(ascending=False, method='min').astype(int)
    
    return latest_year_data.reset_index().sort_values('Rank')


def generate_ai_insight_combined(prompt):
    API_KEY = st.secrets.get("OPENROUTER_API_KEY")
    if not API_KEY: return "API Key not found."
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    data = {"model": "deepseek/deepseek-chat:free", "messages": [{"role": "user", "content": prompt}]}
    try:
        response = requests.post(API_URL, json=data, headers=headers, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e: return f"Network error: {e}"
    except (KeyError, IndexError): return "Failed to parse AI response."


def generate_grouped_plots_by_metric(df, lake_ids, metrics):
    grouped_images = []
    for metric in metrics:
        plt.figure(figsize=(10, 6), dpi=150)
        has_data = False
        for lake_id in lake_ids:
            lake_df = df[df['Lake_ID'] == lake_id].copy()
            if not lake_df.empty and metric in lake_df:
                lake_df = lake_df.sort_values("Year")
                lake_df[metric] = pd.to_numeric(lake_df[metric], errors='coerce')
                if lake_df[metric].notna().sum() > 0:
                    plt.plot(lake_df["Year"], lake_df[metric], marker='o', linestyle='-', label=f"Lake {lake_id}")
                    has_data = True
                if lake_df[metric].notna().sum() > 1:
                    x = lake_df["Year"][lake_df[metric].notna()]; y = lake_df[metric][lake_df[metric].notna()]
                    slope, intercept, *_ = linregress(x, y); plt.plot(x, intercept + slope * x, linestyle='--', alpha=0.7)
        if not has_data:
            plt.close(); continue
        plt.title(f"{metric} Over Time", fontsize=14); plt.xlabel("Year", fontsize=12); plt.ylabel(metric, fontsize=12)
        plt.legend(); plt.grid(True, which='both', linestyle='--', linewidth=0.5); plt.tight_layout()
        buf = BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight'); plt.close(); buf.seek(0)
        grouped_images.append((metric, buf))
    return grouped_images


def generate_comparative_pdf_report(df, results, lake_ids, selected_ui_options):
    buffer = BytesIO(); c = canvas.Canvas(buffer, pagesize=A4); width, height = A4; y = height - 50
    def writeln(text, step=16):
        nonlocal y; y_buffer = 20
        text_lines = []
        for line in text.split('\n'): text_lines.extend(line[i:i+95] for i in range(0, len(line), 95))
        if y - (len(text_lines) * step) < y_buffer: c.showPage(); y = height - 50
        for line in text_lines: c.drawString(40, y, line); y -= step
    
    c.setFont("Helvetica-Bold", 20); c.drawCentredString(width / 2, y, "Dynamic Lake Health Report"); y -= 50
    c.setFont("Helvetica", 12)
    writeln(f"Lakes Analyzed: {', '.join(map(str, lake_ids))}")
    writeln(f"Parameters Considered: {', '.join(selected_ui_options)}")
    
    # **FIX**: Use the centralized weight function for accurate display.
    y -= 10; c.setFont("Helvetica-Bold", 14); writeln("Effective Weights Used:"); c.setFont("Helvetica", 10)
    final_weights = get_effective_weights(selected_ui_options, df.columns)
    for param, weight in final_weights.items():
        writeln(f"- {param}: {weight:.3f}")

    c.showPage(); y = height - 50
    c.setFont("Helvetica-Bold", 14); writeln("Health Score Ranking")
    bar_start_x = 60; bar_height = 18; max_bar_width = width - bar_start_x - 200 
    for _, row in results.iterrows():
        if y < 100: c.showPage(); y = height - 50
        score = row['Health Score']; rank = int(row['Rank'])
        if score > 0.75: color = colors.darkgreen
        elif score > 0.5: color = colors.orange
        else: color = colors.firebrick
        c.setFillColor(color); c.rect(bar_start_x, y, score * max_bar_width, bar_height, fill=1, stroke=0)
        c.setFillColor(colors.black); c.drawString(bar_start_x + 5, y + 5, f"Lake {row['Lake_ID']} (Rank {rank}) - Score: {score:.3f}")
        y -= (bar_height + 10)
    
    if y < 250: c.showPage(); y = height - 50
    c.setFont("Helvetica-Bold", 14); writeln("AI-Generated Analysis"); c.setFont("Helvetica", 10)
    prompt = f"Based on ({', '.join(selected_ui_options)}), analyze lakes: {', '.join(map(str, lake_ids))}.\n"
    for _, row in results.iterrows(): prompt += f"- Lake {row['Lake_ID']}: Score {row['Health Score']:.3f}, Rank {int(row['Rank'])}.\n"
    prompt += "\nDiscuss factors and compare their health. Be concise."
    writeln(generate_ai_insight_combined(prompt))
    
    params_to_plot = list(final_weights.keys())
    if 'HDI' in params_to_plot: params_to_plot.remove('HDI')
    plots = generate_grouped_plots_by_metric(df, lake_ids, params_to_plot)
    for i in range(0, len(plots), 2):
        c.showPage()
        for j in range(2):
            if i + j < len(plots):
                y_pos = height / 2 + 20 if j == 0 else 50; metric, img_buf = plots[i + j]; img = ImageReader(img_buf)
                c.setFont("Helvetica-Bold", 12); c.drawCentredString(width / 2, y_pos + 280, f"Comparison of: {metric}")
                c.drawImage(img, 40, y_pos, width=width-80, height=270, preserveAspectRatio=True, anchor='n')
    
    c.save(); buffer.seek(0); return buffer

# --- STREAMLIT APP LAYOUT ---

st.set_page_config(layout="wide")
st.title("🌊 Dynamic Lake Health Dashboard")

df_health_full, df_location, ui_options = prepare_all_data(HEALTH_DATA_PATH, LOCATION_DATA_PATH)
if df_health_full is None: st.stop()

if 'confirmed_parameters' not in st.session_state: st.session_state.confirmed_parameters = []
if "selected_lake_ids" not in st.session_state: st.session_state.selected_lake_ids = []
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None

with st.sidebar:
    st.header("1. Select & Set Parameters")
    temp_selected_params = st.multiselect("Choose parameters for health score:", options=ui_options, default=st.session_state.get('confirmed_parameters', []))
    if st.button("Set Parameters"):
        st.session_state.confirmed_parameters = temp_selected_params
        st.session_state.analysis_results = None
        st.success("Parameters set!")
    if st.session_state.confirmed_parameters:
        st.markdown("---"); st.markdown("**Confirmed Parameters for Analysis:**")
        for param in st.session_state.confirmed_parameters: st.markdown(f"- `{param}`")
    st.markdown("---")
    st.header("2. Select Lakes")
    sorted_states = sorted(df_location['State'].unique())
    selected_state = st.selectbox("Select State", sorted_states)
    filtered_districts = df_location[df_location['State'] == selected_state]['District'].unique()
    selected_district = st.selectbox("Select District", sorted(filtered_districts))
    filtered_lakes_by_loc = df_location[(df_location['State'] == selected_state) & (df_location['District'] == selected_district)]
    lake_ids_in_district = sorted(filtered_lakes_by_loc['Lake_ID'].unique())
    if lake_ids_in_district:
        selected_lake_id = st.selectbox("Select a Lake ID to Add", lake_ids_in_district)
        if st.button("Add Lake to Comparison"):
            if selected_lake_id not in st.session_state.selected_lake_ids: st.session_state.selected_lake_ids.append(selected_lake_id)
            st.session_state.analysis_results = None 
    else: st.warning("No lakes found in this district.")

col1, col2 = st.columns([0.6, 0.4])

with col1:
    st.subheader(f"Map of Lakes in {selected_district}, {selected_state}")
    if not filtered_lakes_by_loc.empty:
        map_center = [filtered_lakes_by_loc['Lat'].mean(), filtered_lakes_by_loc['Lon'].mean()]
        m = folium.Map(location=map_center, zoom_start=8)
        marker_cluster = MarkerCluster().add_to(m)
        for _, row in filtered_lakes_by_loc.iterrows(): folium.Marker([row['Lat'], row['Lon']], popup=f"<b>Lake ID:</b> {row['Lake_ID']}", tooltip=f"Lake ID: {row['Lake_ID']}", icon=folium.Icon(color='blue', icon='water')).add_to(marker_cluster)
        st_folium(m, height=550, use_container_width=True)

with col2:
    st.subheader("Lakes Selected for Analysis")
    ids_text = ", ".join(map(str, st.session_state.selected_lake_ids))
    edited_ids_text = st.text_area("Edit Lake IDs (comma-separated)", ids_text, height=80)
    try:
        updated_ids = [int(x.strip()) for x in edited_ids_text.split(",") if x.strip()] if edited_ids_text else []
        if updated_ids != st.session_state.selected_lake_ids: st.session_state.analysis_results = None
        st.session_state.selected_lake_ids = updated_ids
    except (ValueError, TypeError): st.warning("Invalid input.")
    lake_ids_to_analyze = st.session_state.get("selected_lake_ids", [])
    is_disabled = not lake_ids_to_analyze or not st.session_state.confirmed_parameters
    if st.button("Analyze Selected Lakes", disabled=is_disabled, use_container_width=True):
        st.session_state.analysis_results = None
        with st.spinner("Analyzing..."):
            try:
                selected_df = df_health_full[df_health_full["Lake_ID"].isin(lake_ids_to_analyze)].copy()
                if selected_df.empty: st.error(f"No health data for Lake IDs: {lake_ids_to_analyze}")
                else:
                    results = calculate_lake_health_score(selected_df, st.session_state.confirmed_parameters)
                    st.session_state.analysis_results = results
                    try: st.session_state.pdf_buffer = generate_comparative_pdf_report(selected_df, results, lake_ids_to_analyze, st.session_state.confirmed_parameters)
                    except Exception as pdf_e: st.warning(f"⚠️ PDF report failed. Error: {pdf_e}")
            except Exception as e: st.error(f"Critical analysis error: {e}")
    st.markdown("---") 
    if st.session_state.analysis_results is not None and not st.session_state.analysis_results.empty:
        st.subheader("Health Score Results")
        st.dataframe(st.session_state.analysis_results[["Lake_ID", "Health Score", "Rank"]].style.format({"Health Score": "{:.3f}"}), height=200)
        st.subheader("Download Center")
        csv_data = df_health_full[df_health_full["Lake_ID"].isin(lake_ids_to_analyze)].to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Filtered Data (CSV)", csv_data, f"data.csv", "text/csv", use_container_width=True)
        if 'pdf_buffer' in st.session_state and st.session_state.pdf_buffer: st.download_button("📄 Download PDF Report", st.session_state.pdf_buffer, f"report.pdf", "application/pdf", use_container_width=True)
    else: st.info("ℹ️ Set parameters and add lakes, then click 'Analyze'.")
