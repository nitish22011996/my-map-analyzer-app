import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import io

# Load CSV at startup
file_path = 'HDI_lake_district.csv'  # Ensure this file is in the working directory
df = pd.read_csv(file_path)

# Clean column names
df.columns = df.columns.str.strip()

# Sidebar: State selection
df['State'] = df['State'].astype(str)
df['District'] = df['District'].astype(str)
sorted_states = sorted(df['State'].unique())
selected_state = st.sidebar.selectbox("Select State", sorted_states)

# Sidebar: District selection based on state
filtered_districts = df[df['State'] == selected_state]['District'].unique()
sorted_districts = sorted(filtered_districts)
selected_district = st.sidebar.selectbox("Select District", sorted_districts)

# Filter lakes in selected district
filtered_lakes = df[(df['State'] == selected_state) & (df['District'] == selected_district)]

if filtered_lakes.empty:
    st.warning("No lakes found in selected district.")
    st.stop()

Lake_IDs = sorted(filtered_lakes['Lake_ID'].unique())
selected_Lake_ID = st.sidebar.selectbox("Select Lake ID", Lake_IDs)

# Initialize session state
if "selected_Lake_IDs" not in st.session_state:
    st.session_state.selected_Lake_IDs = []

# Submit button to save selected lake ID
if st.sidebar.button("Submit"):
    if selected_Lake_ID not in st.session_state.selected_Lake_IDs:
        st.session_state.selected_Lake_IDs.append(selected_Lake_ID)

# Display selected lake IDs
st.subheader("Selected Lake IDs")
if st.session_state.selected_Lake_IDs:
    formatted_ids = ", ".join(str(lid) for lid in st.session_state.selected_Lake_IDs)
    st.write(formatted_ids)
else:
    st.write("No lake IDs selected yet.")
if st.button("Clear Selection"):
    st.session_state.selected_Lake_IDs = []
    st.experimental_rerun()
# Button to download selected lake IDs
if st.session_state.selected_Lake_IDs:
    csv_data = pd.DataFrame({'Lake_ID': st.session_state.selected_Lake_IDs})
    csv_buffer = io.StringIO()
    csv_data.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download Selected Lake IDs as CSV",
        data=csv_buffer.getvalue(),
        file_name='selected_Lake_IDs.csv',
        mime='text/csv'
    )

# Map of lakes in selected district
st.subheader(f"Lakes in {selected_district}, {selected_state}")
m = folium.Map(location=[filtered_lakes['Lat'].mean(), filtered_lakes['Lon'].mean()], zoom_start=7)
marker_cluster = MarkerCluster().add_to(m)

for _, row in filtered_lakes.iterrows():
    folium.Marker(
        location=[row['Lat'], row['Lon']],
        popup=f"Lake ID: {row['Lake_ID']}",
        icon=folium.Icon(color='blue')
    ).add_to(marker_cluster)

st_folium(m, width=700, height=500)
