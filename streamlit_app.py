import streamlit as st
import requests

st.set_page_config(page_title="Map Analyzer", layout="centered")
st.title("🗺️ Upload Map for Analysis")

API_URL = "https://lab123.tail7bcbe3.ts.net/"  # or public API if deployed
API_KEY = st.secrets["api_key"]

uploaded_file = st.file_uploader("Upload your map file", type=["tif", "zip"])

if uploaded_file:
    st.success(f"Uploaded {uploaded_file.name}")

    files = {
        "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }

    with st.spinner("Analyzing..."):
        response = requests.post(API_URL, headers=headers, files=files)

    if response.status_code == 200:
        st.success("✅ Analysis complete!")
        st.json(response.json())
    else:
        st.error(f"Error {response.status_code}: {response.text}")
