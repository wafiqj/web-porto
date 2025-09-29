import streamlit as st

st.set_page_config(page_title="CV - Wafiq Jaisyurrahman")

st.write("Redirecting to my CV...")
# https://drive.google.com/file/d/1f1Is1HIsUQhdf5J3gbUgKFwMicl6D4e7/view?usp=sharing
# Ganti ID di bawah ini dengan ID Google Drive kamu
cv_link = "https://drive.google.com/file/d/1f1Is1HIsUQhdf5J3gbUgKFwMicl6D4e7/view?usp=sharing"

# Redirect otomatis
st.markdown(f"""
<meta http-equiv="refresh" content="0; url={cv_link}">
If you're not redirected automatically, <a href="{cv_link}">click here to view my CV</a>.
""", unsafe_allow_html=True)
