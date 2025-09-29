import streamlit as st

st.set_page_config(layout="wide")
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,200..800;1,200..800&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

def home():
    st.markdown("""
    <style>
    pre, code {
        font-size: 16px !important;
    }
    </style>
""", unsafe_allow_html=True)


    st.title("Welcome!")

    st.code("""class Wafiq:
    def __init__(self):
        self.name = "Wafiq Jaisyurrahman"
        self.background = "Sistem Informasi, Telkom University"
        self.interests = ["Big Data", "Data Engineering", "AI", "Deep Learning"]

    def say_hi(self):
        print(f"Hi, I'm {self.name} 👋 Let's build something cool together!")

wafiq = Wafiq()
wafiq.say_hi()
""", language="python")
    
    st.divider()
    st.markdown("""
###### Want to know more about my background, skills, and experience?
Feel free to explore my CV below.""")
    st.link_button("View CV", url="https://drive.google.com/file/d/1QmNksv7ooi24c0losWWB5GhvquMPBN1D/view?usp=sharing", type="secondary")

pg = st.navigation({
    "Portofolio":[
    st.Page(home, title="Home", icon="🏠"),
    st.Page("waps.py", title="WAPS", icon="🧠"),
    st.Page("about.py", title="About", icon="🧒🏻"),
    st.Page("project.py", title="Project", icon="💻"),
    st.Page("contact.py", title="Contact", icon="📲")],
    "Playground":[
    st.Page("playground/number-guess-debug.py", title="Guess the Number", icon="🔢"),
    st.Page("playground/mood-detect.py", title="Mood Detector", icon="🧐"),
    st.Page("playground/quran-game-ori.py", title="Quran Game", icon="📖"),
    # st.Page("playground/wall-of-fame.py", title="Wall of Fame", icon="🖼️"),
    ]
    })

pg.run()