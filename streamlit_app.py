import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

st.set_page_config(layout="wide")

st.title("Lake Selection and Export Tool")

# Upload CSV file
uploaded_file = st.sidebar.file_uploader("HDI_lake_district.csv", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    df['STATE'] = df['STATE'].astype(str)
    df['District'] = df['District'].astype(str)

    sorted_states = sorted(df['STATE'].unique())
    selected_state = st.sidebar.selectbox("Select State", sorted_states)

    filtered_districts = df[df['STATE'] == selected_state]['District'].unique()
    selected_district = st.sidebar.selectbox("Select District", sorted(filtered_districts))

    filtered_lakes = df[(df['STATE'] == selected_state) & (df['District'] == selected_district)]

    if filtered_lakes.empty:
        st.error("No lakes found for the selected state and district.")
        st.stop()

    lake_ids = sorted(filtered_lakes['Lake_id'].unique())
    selected_lake_ids = st.sidebar.multiselect("Select Lake IDs", lake_ids)

    # Submit button
    submit = st.sidebar.button("Submit")

    if "selected_ids" not in st.session_state:
        st.session_state.selected_ids = []

    if submit:
        st.session_state.selected_ids = selected_lake_ids

    # Display selected lake IDs
    if st.session_state.selected_ids:
        st.subheader("Selected Lake IDs")
        selected_df = pd.DataFrame({'Lake_ID': st.session_state.selected_ids})
        st.dataframe(selected_df)

        # CSV download button
        csv = selected_df.to_csv(index=False)
        st.download_button(
            label="Download Selected Lake IDs as CSV",
            data=csv,
            file_name="selected_lake_ids.csv",
            mime="text/csv"
        )

    # Map of filtered lakes
    st.subheader(f"Map of Lakes in {selected_district}, {selected_state}")
    m = folium.Map(location=[filtered_lakes['Lat'].mean(), filtered_lakes['Lon'].mean()], zoom_start=7)
    marker_cluster = MarkerCluster().add_to(m)

    for _, row in filtered_lakes.iterrows():
        folium.Marker(
            location=[row['Lat'], row['Lon']],
            popup=f"Lake ID: {row['Lake_id']}",
            icon=folium.Icon(color='green')
        ).add_to(marker_cluster)

    st_folium(m, width=700, height=500)

else:
    st.info("Please upload the CSV file to begin.")
