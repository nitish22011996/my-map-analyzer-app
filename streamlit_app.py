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
df['STATE'] = df['STATE'].astype(str)
df['District'] = df['District'].astype(str)
sorted_states = sorted(df['STATE'].unique())
selected_state = st.sidebar.selectbox("Select State", sorted_states)

# Sidebar: District selection based on state
filtered_districts = df[df['STATE'] == selected_state]['District'].unique()
sorted_districts = sorted(filtered_districts)
selected_district = st.sidebar.selectbox("Select District", sorted_districts)

# Filter lakes in selected district
filtered_lakes = df[(df['STATE'] == selected_state) & (df['District'] == selected_district)]

if filtered_lakes.empty:
    st.warning("No lakes found in selected district.")
    st.stop()

lake_ids = sorted(filtered_lakes['Lake_id'].unique())
selected_lake_id = st.sidebar.selectbox("Select Lake ID", lake_ids)

# Initialize session state
if "selected_lake_ids" not in st.session_state:
    st.session_state.selected_lake_ids = []

# Submit button to save selected lake ID
if st.sidebar.button("Submit"):
    if selected_lake_id not in st.session_state.selected_lake_ids:
        st.session_state.selected_lake_ids.append(selected_lake_id)

# Display selected lake IDs
st.subheader("Selected Lake IDs")
if st.session_state.selected_lake_ids:
    st.write(st.session_state.selected_lake_ids)
else:
    st.write("No lake IDs selected yet.")

# Button to download selected lake IDs
if st.session_state.selected_lake_ids:
    csv_data = pd.DataFrame({'Lake_id': st.session_state.selected_lake_ids})
    csv_buffer = io.StringIO()
    csv_data.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download Selected Lake IDs as CSV",
        data=csv_buffer.getvalue(),
        file_name='selected_lake_ids.csv',
        mime='text/csv'
    )

# Map of lakes in selected district
st.subheader(f"Lakes in {selected_district}, {selected_state}")
m = folium.Map(location=[filtered_lakes['Lat'].mean(), filtered_lakes['Lon'].mean()], zoom_start=7)
marker_cluster = MarkerCluster().add_to(m)

for _, row in filtered_lakes.iterrows():
    folium.Marker(
        location=[row['Lat'], row['Lon']],
        popup=f"Lake ID: {row['Lake_id']}",
        icon=folium.Icon(color='blue')
    ).add_to(marker_cluster)

st_folium(m, width=700, height=500)
