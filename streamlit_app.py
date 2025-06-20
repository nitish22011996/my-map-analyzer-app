import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import io

# Load only required columns
file_path = 'HDI_lake_district.csv'
df = pd.read_csv(file_path, usecols=['Lat', 'Lon', 'Lake_ID', 'District', 'State'])
df.columns = df.columns.str.strip()

# Convert relevant columns to string
df['State'] = df['State'].astype(str).str.strip()
df['District'] = df['District'].astype(str).str.strip()

# Sidebar: State and District selection
sorted_states = sorted(df['State'].unique())
selected_state = st.sidebar.selectbox("Select State", sorted_states)

filtered_districts = df[df['State'] == selected_state]['District'].unique()
sorted_districts = sorted(filtered_districts)
selected_district = st.sidebar.selectbox("Select District", sorted_districts)

# Filter lakes in selected district
filtered_lakes = df[(df['State'] == selected_state) & (df['District'] == selected_district)]

# Initialize session state list
if "selected_lake_ids" not in st.session_state:
    st.session_state.selected_lake_ids = []

# Submit button: add all lake IDs from selected district
if st.sidebar.button("Submit"):
    new_ids = filtered_lakes['Lake_ID'].tolist()
    for lake_id in new_ids:
        if lake_id not in st.session_state.selected_lake_ids:
            st.session_state.selected_lake_ids.append(lake_id)

# Display selected lake IDs
st.subheader("Selected Lake IDs")
if st.session_state.selected_lake_ids:
    st.write(", ".join(str(lid) for lid in st.session_state.selected_lake_ids))
else:
    st.write("No lake IDs selected yet.")

# Clear selected IDs
if st.button("Clear Selection"):
    st.session_state.selected_lake_ids = []
    st.experimental_rerun()

# Download selected lake IDs
if st.session_state.selected_lake_ids:
    csv_data = pd.DataFrame({'Lake_ID': st.session_state.selected_lake_ids})
    csv_buffer = io.StringIO()
    csv_data.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download Selected Lake IDs as CSV",
        data=csv_buffer.getvalue(),
        file_name='selected_Lake_IDs.csv',
        mime='text/csv'
    )

# Show map
st.subheader(f"Lakes in {selected_district}, {selected_state}")
if not filtered_lakes.empty:
    m = folium.Map(location=[filtered_lakes['Lat'].mean(), filtered_lakes['Lon'].mean()], zoom_start=7)
    marker_cluster = MarkerCluster().add_to(m)

    for _, row in filtered_lakes.iterrows():
        popup_info = f"""
        <b>Lake ID:</b> {row['Lake_ID']}<br>
        <b>District:</b> {row['District']}<br>
        <b>State:</b> {row['State']}
        """
        folium.Marker(
            location=[row['Lat'], row['Lon']],
            popup=folium.Popup(popup_info, max_width=250),
            icon=folium.Icon(color='blue')
        ).add_to(marker_cluster)

    st_folium(m, width=700, height=500)
else:
    st.warning("No lakes found in selected district.")

