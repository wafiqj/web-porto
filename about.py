import streamlit as st


# st.set_page_config(
#     page_title="Portofolio Data Anda",
#     page_icon="📊", # Anda bisa ganti dengan emoji lain atau path ke ikon
#     layout="centered", # atau "wide" jika Anda ingin layout lebih lebar
#     initial_sidebar_state="expanded" # "auto", "expanded", "collapsed"
# )

# st.title("Welcome! 👋🏻")
# st.markdown("---")
st.image("photo.jpg", width=200)
st.markdown("#### Wafiq Jaisyurrahman, :small[(uo)] S.Kom. &mdash; *Data Enthusiast*")

st.markdown("""
Hey! I’m Wafiq Jaisyurrahman. I studied Information Systems at Telkom University, and I’ve been diving deep into the world of data, AI, deep learning, and computer vision ever since.
I love building smart solutions that actually work in real life — from dashboards to machine learning models. Most of the time, I’m working with Python, spreadsheets, Power BI, Azure, and MySQL.
*Always curious, always learning*.
""")

st.markdown("---") # Garis pemisah

st.write("Take a look at some of my work below!")

# Tombol CTA
if st.button("My Projects"):
    st.switch_page("project.py")
    # Di aplikasi multi-halaman, Anda akan menggunakan st.switch_page("pages/projects.py")
    # atau mekanisme navigasi lainnya.

st.markdown("---")
st.write("Let's connect on [LinkedIn](https://www.linkedin.com/in/wafiq-jaisyurrahman/) or [GitHub](https://github.com/wafiqj)!")