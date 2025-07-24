import streamlit as st

with st.form("contact_form", border=0):
    st.title("Contact Me")
    name = st.text_input("Name", placeholder="Your Name")
    email = st.text_input("Email", placeholder="Your Email")
    message = st.text_area("Message", placeholder="Your Message")
    
    submit_button = st.form_submit_button("Send Message")
    
    if submit_button:
        if name and email and message:
            st.success(f"Thank you {name}! Your message has been sent.")
        else:
            st.error("Please fill in all fields.")