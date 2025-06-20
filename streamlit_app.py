import streamlit as st
import pandas as pd

# Load the CSV data
file_path = 'HDI_lake_district.csv'
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# Clean data types
df['STATE'] = df['STATE'].astype(str)
df['District'] = df['District'].astype(str)

# Sidebar: State selection
st.sidebar.subheader("Select State")
sorted_states = sorted(df['STATE'].unique())
selected_state = st.sidebar.selectbox("Choose a State:", sorted_states)

# Sidebar: District selection based on state
filtered_districts = df[df['STATE'] == selected_state]['District'].unique()
sorted_districts = sorted(filtered_districts)
selected_district = st.sidebar.selectbox("Choose a District:", sorted_districts)

# Filter lakes based on state and district
filtered_lakes = df[(df['STATE'] == selected_state) & (df['District'] == selected_district)]
lake_ids = sorted(filtered_lakes['Lake_id'].unique())

# Sidebar: Multi-select Lake IDs
st.sidebar.subheader("Select Lake IDs")
selected_lake_ids = st.sidebar.multiselect("Choose Lake IDs:", lake_ids)

# Submit button
submit = st.sidebar.button("Submit")

# Display selected lake IDs and CSV download
if submit:
    if selected_lake_ids:
        st.subheader("Selected Lake IDs")
        st.write(selected_lake_ids)

        # Create DataFrame and download button
        selected_df = pd.DataFrame(selected_lake_ids, columns=["Lake_id"])
        csv_data = selected_df.to_csv(index=False)

        st.download_button(
            label="Download Selected Lake IDs as CSV",
            data=csv_data,
            file_name="selected_lake_ids.csv",
            mime="text/csv"
        )
    else:
        st.warning("Please select at least one Lake ID.")
