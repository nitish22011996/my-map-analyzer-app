import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from matplotlib import pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import inch

# --- Constants and factor names ---
FACTORS = ["Air_Temperature", "Evaporation", "Vegetation_Area", "Urban_Area", "Barren_Area"]

# --- Load data ---
@st.cache_data
def load_data():
    # Make sure your CSV has columns: Lake, Year, and the FACTORS
    return pd.read_csv("lake_health_data.csv")

df = load_data()

# --- Normalize factor values ---
def normalize_column(col, invert=False):
    min_val = col.min()
    max_val = col.max()
    norm = (col - min_val) / (max_val - min_val) if max_val != min_val else col - min_val
    if invert:
        norm = 1 - norm
    return norm

# --- Calculate weighted health score ---
def calculate_health_scores(df, lake_ids, factors, weights):
    # Filter for latest year only
    latest_year = df["Year"].max()
    df_latest = df[df["Year"] == latest_year].copy()
    df_latest = df_latest[df_latest["Lake"].astype(str).isin(lake_ids)]
    if df_latest.empty:
        return pd.DataFrame()

    # Normalize each factor (invert for negative factors where lower is better)
    # Here: evaporation, air temperature, urban area, barren area -> lower is better
    invert_factors = ["Evaporation", "Air_Temperature", "Urban_Area", "Barren_Area"]
    for f in factors:
        invert = f in invert_factors
        df_latest[f"{f}_norm"] = normalize_column(df_latest[f], invert=invert)

    # Calculate weighted sum
    norm_cols = [f"{f}_norm" for f in factors]
    weight_list = [weights[f] for f in factors]
    df_latest["Health Score"] = df_latest[norm_cols].values @ np.array(weight_list)

    # Rank by health score descending
    df_latest["Rank"] = df_latest["Health Score"].rank(ascending=False, method="min").astype(int)

    return df_latest.sort_values("Rank")

# --- Generate plots per factor ---
def plot_factor(df, factor, lake_ids):
    plt.figure(figsize=(6, 3))
    for lake in lake_ids:
        data = df[df["Lake"].astype(str) == lake]
        plt.plot(data["Year"], data[factor], label=f"Lake {lake}")
    plt.title(f"{factor} over Years")
    plt.xlabel("Year")
    plt.ylabel(factor)
    plt.legend()
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return buf

# --- Generate PDF report ---
def generate_comparative_pdf_report(df, scores_df, lake_ids, factors, weights):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Title page
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width / 2, height - inch, "Lake Health Comparative Report")
    c.setFont("Helvetica", 12)
    c.drawCentredString(width / 2, height - inch - 20, f"Year: {df['Year'].max()}")
    c.showPage()

    # Scores Table
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - inch, "Lake Health Scores and Ranks")
    y_pos = height - inch - 30

    # Draw table header
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_pos, "Lake ID")
    c.drawString(120, y_pos, "Health Score")
    c.drawString(220, y_pos, "Rank")
    y_pos -= 20

    c.setFont("Helvetica", 12)
    for _, row in scores_df.iterrows():
        c.drawString(50, y_pos, str(row["Lake"]))
        c.drawString(120, y_pos, f"{row['Health Score']:.3f}")
        c.drawString(220, y_pos, str(row["Rank"]))
        y_pos -= 20
        if y_pos < inch:
            c.showPage()
            y_pos = height - inch

    # Factors weights info
    c.showPage()
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - inch, "Weights Used for Each Factor")
    y_pos = height - inch - 30
    c.setFont("Helvetica", 12)
    for f in factors:
        c.drawString(50, y_pos, f"{f}: {weights[f]:.2f}")
        y_pos -= 20
        if y_pos < inch:
            c.showPage()
            y_pos = height - inch

    # Plots per factor
    for f in factors:
        c.showPage()
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - inch, f"Time Series of {f}")

        img_buf = plot_factor(df, f, lake_ids)
        c.drawImage(img_buf, 50, height/2 - inch/2, width=500, height=250, preserveAspectRatio=True)

    c.save()
    buffer.seek(0)
    return buffer

# --- Streamlit UI ---
st.title("Lake Health Score Dashboard")
st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Set Weights for Each Factor (Sum will be normalized automatically)")
weights = {}
total_weight = 0.0
cols = st.columns(len(FACTORS))
for i, f in enumerate(FACTORS):
    weights[f] = cols[i].slider(f"Weight for {f}", 0.0, 1.0, 0.2, 0.01)
total_weight = sum(weights.values())
if total_weight > 0:
    for f in FACTORS:
        weights[f] /= total_weight

lake_ids_input = st.text_input("Enter Lake IDs separated by commas (e.g. 630,2168266,737797)").replace(" ", "")
lake_ids = [x for x in lake_ids_input.split(",") if x]

if lake_ids:
    filtered_df = df[df["Lake"].astype(str).isin(lake_ids)]
    if filtered_df.empty:
        st.error("No data found for the entered Lake IDs.")
    else:
        results_df = calculate_health_scores(filtered_df, lake_ids, FACTORS, weights)
        if results_df.empty:
            st.error("No data available for latest year to calculate scores.")
        else:
            st.subheader("Lake Health Scores")
            st.dataframe(results_df[["Lake", "Health Score", "Rank"]])

            csv_data = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Original CSV for Selected Lakes", csv_data, "selected_lake_data.csv", "text/csv")

            st.subheader("Download Combined PDF Report")
            pdf_buf = generate_comparative_pdf_report(filtered_df, results_df, lake_ids, FACTORS, weights)
            st.download_button(
                label="📄 Download Combined Lake Health Report",
                data=pdf_buf,
                file_name="combined_lake_health_report.pdf",
                mime="application/pdf"
            )
else:
    st.info("Enter at least one Lake ID above to get started.")
