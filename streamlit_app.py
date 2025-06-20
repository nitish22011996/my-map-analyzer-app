import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import textwrap
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from scipy.stats import linregress

# Page setup
st.set_page_config(layout="wide")

# Custom CSS to fix input box width and reduce padding
st.markdown("""
<style>
    .stTextInput input {
        width: 100% !important;
    }
    .main .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# File paths
HDI_PATH = 'HDI_lake_district.csv'
HEALTH_PATH = 'lake_health_data.csv'

@st.cache_data
def load_hdi():
    df = pd.read_csv(HDI_PATH)
    df.columns = df.columns.str.strip()
    df['STATE'] = df['STATE'].astype(str)
    df['District'] = df['District'].astype(str)
    return df

@st.cache_data
def load_health():
    return pd.read_csv(HEALTH_PATH)

hdi_df = load_hdi()
health_df = load_health()

# Initialize session state
if "selected_lake_ids" not in st.session_state:
    st.session_state.selected_lake_ids = []

# --- Layout: Three Equal Columns ---
col1, col2, col3 = st.columns([1, 1, 2])

# --- Column 1: Lake Selection ---
with col1:
    st.header("Lake Selection")
    states = sorted(hdi_df['STATE'].unique())
    selected_state = st.selectbox("Select State", states)
    districts = sorted(hdi_df[hdi_df['STATE'] == selected_state]['District'].unique())
    selected_district = st.selectbox("Select District", districts)

    filtered_lakes = hdi_df[(hdi_df['STATE'] == selected_state) & (hdi_df['District'] == selected_district)]

    if filtered_lakes.empty:
        st.warning("No lakes found in selected district.")
    else:
        lake_ids = sorted(filtered_lakes['Lake_id'].unique())
        selected_lake_id = st.selectbox("Select Lake ID", lake_ids)

        if st.button("Submit Selection"):
            if selected_lake_id not in st.session_state.selected_lake_ids:
                st.session_state.selected_lake_ids.append(int(selected_lake_id))

    manual_ids = st.text_input("Enter Lake IDs (comma-separated)")
    if manual_ids:
        try:
            ids = [int(x.strip()) for x in manual_ids.split(',') if x.strip().isdigit()]
            st.session_state.selected_lake_ids.extend([x for x in ids if x not in st.session_state.selected_lake_ids])
        except Exception:
            st.error("Please enter valid numeric Lake IDs.")

    if st.button("Clear Selection"):
        st.session_state.selected_lake_ids = []
        st.experimental_rerun()

    st.subheader("Selected Lake IDs")
    if st.session_state.selected_lake_ids:
        st.write(", ".join(map(str, st.session_state.selected_lake_ids)))
        csv_data = pd.DataFrame({'Lake_id': st.session_state.selected_lake_ids})
        csv_buffer = BytesIO()
        csv_data.to_csv(csv_buffer, index=False)
        st.download_button("Download Selected Lake IDs as CSV", data=csv_buffer.getvalue(), file_name='selected_lake_ids.csv', mime='text/csv')
    else:
        st.write("No lake IDs selected yet.")

# --- Column 2: Map ---
with col2:
    st.header("Lakes in Selected District")
    if not filtered_lakes.empty:
        m = folium.Map(location=[filtered_lakes['Lat'].mean(), filtered_lakes['Lon'].mean()], zoom_start=7)
        marker_cluster = MarkerCluster().add_to(m)

        for _, row in filtered_lakes.iterrows():
            folium.Marker(
                location=[row['Lat'], row['Lon']],
                popup=f"Lake ID: {row['Lake_id']}",
                icon=folium.Icon(color='blue')
            ).add_to(marker_cluster)

        st_folium(m, width=700, height=500)

# --- Column 3: Lake Health Dashboard ---
with col3:
    st.header("Lake Health Score Dashboard")
    lake_ids = list(map(str, st.session_state.selected_lake_ids))
    if not lake_ids:
        st.info("Select lakes using the interface to get started.")
    else:
        selected_df = health_df[health_df["Lake"].astype(str).isin(lake_ids)]
        if selected_df.empty:
            st.error("No data found for selected Lake IDs.")
        else:
            def calculate_lake_health_score(df):
                def norm(x): return (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0
                def rev_norm(x): return 1 - norm(x)

                required_cols = ['Lake', 'Year', 'Vegetation Area', 'Barren Area', 'Urban Area', 'Precipitation', 'Evaporation', 'Air Temperature']
                for col in required_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

                latest = df[df['Year'] == df['Year'].max()].copy()
                if latest.empty:
                    return pd.DataFrame()

                latest['Vegetation Area Normalized'] = norm(latest['Vegetation Area'])
                latest['Barren Area Normalized'] = rev_norm(latest['Barren Area'])
                latest['Urban Area Normalized'] = rev_norm(latest['Urban Area'])
                latest['Precipitation Normalized'] = norm(latest['Precipitation'])
                latest['Evaporation Normalized'] = rev_norm(latest['Evaporation'])
                latest['Air Temperature Normalized'] = rev_norm(latest['Air Temperature'])

                for col in latest.columns:
                    if 'Normalized' in col:
                        latest[col] = latest[col].replace([np.inf, -np.inf, np.nan], 0)

                def get_slope_pval(df_group):
                    slopes = {}
                    pvals = {}
                    for col in ['Vegetation Area', 'Barren Area', 'Urban Area', 'Precipitation', 'Evaporation', 'Air Temperature']:
                        slope, _, _, pval, _ = linregress(df_group['Year'], df_group[col])
                        slopes[f"{col} Trend"] = slope
                        pvals[f"{col} PValue"] = pval
                    return pd.Series({**slopes, **pvals})

                trends = df.groupby('Lake').apply(get_slope_pval).reset_index()

                for col in trends.columns:
                    if 'Trend' in col:
                        trends[col + ' Normalized'] = norm(trends[col]) if 'Vegetation' in col or 'Precipitation' in col else rev_norm(trends[col])
                    if 'PValue' in col:
                        trends[col + ' Normalized'] = 1 - norm(trends[col])

                for col in trends.columns:
                    if 'Normalized' in col:
                        trends[col] = trends[col].replace([np.inf, -np.inf, np.nan], 0)

                latest.set_index('Lake', inplace=True)
                trends.set_index('Lake', inplace=True)
                combined = latest.join(trends, how='inner')

                def score(factor):
                    return (combined[factor + ' Normalized'] + combined[factor + ' Trend Normalized'] + combined[factor + ' PValue Normalized']) / 3

                combined['Health Score'] = (score('Vegetation Area') + score('Barren Area') + score('Urban Area') +
                                            score('Precipitation') + score('Evaporation') + score('Air Temperature')) / 6
                combined['Rank'] = combined['Health Score'].rank(ascending=False)
                return combined.reset_index()

            result_df = calculate_lake_health_score(selected_df)
            st.dataframe(result_df[['Lake', 'Health Score', 'Rank']], use_container_width=True)

            def generate_metric_plots(df, lake_ids, metrics):
                images = []
                for metric in metrics:
                    plt.figure(figsize=(10, 6), dpi=150)
                    for lake in lake_ids:
                        ldf = df[df['Lake'].astype(str) == str(lake)]
                        if ldf.empty: continue
                        plt.plot(ldf['Year'], ldf[metric], marker='o', label=f"Lake {lake}")
                        if ldf[metric].notna().sum() > 1:
                            slope, intercept, *_ = linregress(ldf['Year'], ldf[metric])
                            plt.plot(ldf['Year'], intercept + slope * ldf['Year'], linestyle='--')
                    plt.title(metric)
                    plt.xlabel("Year")
                    plt.ylabel(metric)
                    plt.legend()
                    plt.grid(True)
                    buf = BytesIO()
                    plt.tight_layout()
                    plt.savefig(buf, format='png')
                    plt.close()
                    buf.seek(0)
                    images.append((metric, buf))
                return images

            def generate_pdf_report(df, results, lake_ids):
                buf = BytesIO()
                c = canvas.Canvas(buf, pagesize=A4)
                width, height = A4
                y = height - 50

                def writeln(text):
                    nonlocal y
                    for line in textwrap.wrap(text, width=90):
                        c.drawString(40, y, line)
                        y -= 18
                        if y < 60:
                            c.showPage()
                            y = height - 50

                c.setFont("Helvetica-Bold", 18)
                c.drawCentredString(width / 2, y, "Lake Health Comparative Report")
                y -= 40
                c.setFont("Helvetica", 12)
                writeln("Compared Lakes: " + ", ".join(lake_ids))
                c.showPage()

                for _, row in results.iterrows():
                    c.setFont("Helvetica-Bold", 14)
                    c.drawString(40, y, f"Lake {row['Lake']}: Score = {row['Health Score']:.2f}, Rank = {int(row['Rank'])}")
                    y -= 20
                    if y < 60:
                        c.showPage()
                        y = height - 50

                c.showPage()
                plots = generate_metric_plots(df, lake_ids, ['Vegetation Area', 'Barren Area', 'Urban Area', 'Precipitation', 'Evaporation', 'Air Temperature'])
                for i in range(0, len(plots), 2):
                    for j in range(2):
                        if i + j >= len(plots): break
                        metric, img_buf = plots[i + j]
                        c.drawString(40, y, metric)
                        img = ImageReader(img_buf)
                        c.drawImage(img, 50, y - 270, width=500, height=250)
                        y -= 280
                        if y < 100:
                            c.showPage()
                            y = height - 50
                    c.showPage()

                c.save()
                buf.seek(0)
                return buf

            st.subheader("Download PDF Report")
            pdf = generate_pdf_report(selected_df, result_df, lake_ids)
            st.download_button("📄 Download PDF Report", data=pdf, file_name="lake_health_report.pdf", mime="application/pdf")
