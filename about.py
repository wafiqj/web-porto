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
st.subheader("You can call me Wafiq.")

st.write("""
Selamat datang di portofolio data saya!
Saya bersemangat dalam mengubah data mentah menjadi insight yang berarti dan solusi inovatif.
Di sini Anda akan menemukan berbagai proyek yang menunjukkan keahlian saya dalam analisis data,
pemodelan machine learning, dan visualisasi interaktif.
""")

# Ganti URL placeholder dengan URL foto Anda sendiri, atau letakkan foto di folder proyek
# dan gunakan st.image("nama_foto.jpg")

st.markdown("---") # Garis pemisah

st.write("Tertarik dengan pekerjaan saya? Mari jelajahi proyek-proyek saya!")

# Tombol CTA
if st.button("Lihat Proyek Saya"):
    st.switch_page("project.py")
    # Di aplikasi multi-halaman, Anda akan menggunakan st.switch_page("pages/projects.py")
    # atau mekanisme navigasi lainnya.

st.markdown("---")
st.write("Hubungi saya di [LinkedIn Anda](https://www.linkedin.com/in/nama-anda/) atau [GitHub Anda](https://github.com/nama-anda/)")