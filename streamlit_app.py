import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

# Load the CSV data
file_path = 'HDI_lake_district.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# Clean column names
df.columns = df.columns.str.strip()

# Sort state and district dropdowns
df['STATE'] = df['STATE'].astype(str)
df['District'] = df['District'].astype(str)
sorted_states = sorted(df['STATE'].unique())
default_state = sorted_states[0]

# Sidebar: Select State
st.sidebar.subheader("Select State")
selected_state = st.sidebar.selectbox("Choose a State:", sorted_states, index=0)

# Filter districts for selected state
filtered_districts = df[df['STATE'] == selected_state]['District'].unique()
sorted_districts = sorted(filtered_districts)
default_district = sorted_districts[0]

# Sidebar: Select District
st.sidebar.subheader("Select District")
selected_district = st.sidebar.selectbox("Choose a District:", sorted_districts, index=0)

# Filter lakes in selected district
filtered_lakes = df[(df['STATE'] == selected_state) & (df['District'] == selected_district)]

# Ensure lakes exist
if filtered_lakes.empty:
    st.error("No lakes found for the selected State and District.")
    st.stop()

lake_ids = sorted(filtered_lakes['Lake_id'].unique())
default_lake_id = lake_ids[0]

# Sidebar: Select Lake ID
st.sidebar.subheader("Select Lake ID")
selected_lake_id = st.sidebar.selectbox("Choose a Lake ID:", lake_ids, index=0)

# Submit button
submit = st.sidebar.button("Submit Selection")

# Session state to store selected lake IDs
if "selected_lake_ids" not in st.session_state:
    st.session_state.selected_lake_ids = []

# If submit is clicked, store the selection
if submit:
    if selected_lake_id not in st.session_state.selected_lake_ids:
        st.session_state.selected_lake_ids.append(selected_lake_id)

# Display selected lake IDs
st.subheader("Selected Lake IDs")
if st.session_state.selected_lake_ids:
    st.write(", ".join(str(lid) for lid in st.session_state.selected_lake_ids))
else:
    st.write("No lake selected yet.")

# Show map with filtered lakes
st.subheader(f"Map of Lakes in {selected_district}, {selected_state}")
m = folium.Map(location=[filtered_lakes['Lat'].mean(), filtered_lakes['Lon'].mean()], zoom_start=7)
marker_cluster = MarkerCluster().add_to(m)

# Add markers for lakes
for _, row in filtered_lakes.iterrows():
    folium.Marker(
        location=[row['Lat'], row['Lon']],
        popup=f"Lake ID: {row['Lake_id']}",
        icon=folium.Icon(color='green')
    ).add_to(marker_cluster)

# Display folium map
st_folium(m, width=700, height=500)


