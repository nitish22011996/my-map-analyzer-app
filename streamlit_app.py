

import streamlit as st
import requests
import os
import base64

# Title
st.title("🧠 Map Analyzer with AI")

# File uploader
uploaded_file = st.file_uploader("Upload a map image (PNG only)", type=["png"])

# Prompt input
user_prompt = st.text_area("Enter your analysis prompt")

# Show image
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Map", use_column_width=True)

# Analyze button
if uploaded_file and user_prompt:
    if st.button("Analyze"):

        # Convert image to base64
        image_bytes = uploaded_file.read()
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")

        # Prepare API call
        api_key = os.getenv("WEBUI_API_KEY")
        if not api_key:
            st.error("API key is missing. Please set it in Streamlit Secrets.")
        else:
            url = "https://lab123.tail7bcbe3.ts.net/api/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            # Prompt with image and user message
            payload = {
                "model": "deepseek-r1:1.5b",
                "messages": [
                    {
                        "role": "user",
                        "content": f"{user_prompt}\n\n[Image attached below]",
                    }
                ],
                "image": {
                    "name": "map.png",
                    "type": "image/png",
                    "base64": encoded_image
                }
            }

            try:
                response = requests.post(url, headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()['choices'][0]['message']['content']
                st.subheader("📝 AI Analysis Result")
                st.markdown(result)

            except Exception as e:
                st.error(f"Error: {str(e)}")
