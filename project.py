import streamlit as st
from streamlit_chat import message

st.title("Project")

with st.container(border=True):
    col1, col2 = st.columns(2)

    with col1:
        st.image("project_big_data.jpg", width=320)
    
    with col2:
        st.markdown("#### Big Data Pipeline")
        st.write("Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type ")

with st.container(border=True):
    col1, col2 = st.columns(2)

    with col1:
        st.image("project_car_detect.png", width=320)
    
    with col2:
        st.markdown("#### CompVis Implementation")
        st.write("Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type ")

with st.container(border=True):
    col1, col2 = st.columns(2)

    with col1:
        st.image("project_sales_dash.png", width=320)
    
    with col2:
        st.markdown("#### Sales Dashboard Real-time")
        st.write("Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type ")