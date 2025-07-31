import streamlit as st
import json

st.title("Project")

@st.dialog("Project Details", width="large")
def dialog(project_key):
    project = next((p for p in projects if p["button_key"] == project_key), None)
    if project:
        st.markdown(f"# {project['title']}")
        st.image(project["image"])
        if "tech_stack" in project:
            badges = " ".join([f":blue-badge[{tech}]" for tech in project["tech_stack"]])
            st.markdown(badges)

        custom_style = """
<style>
.justify-text p {
    text-align: justify;
}
</style>
"""
        st.markdown(custom_style, unsafe_allow_html=True)
        st.markdown(f'<div class="justify-text">\n\n{project["description"]}\n\n</div>', unsafe_allow_html=True)
        # st.write(project["description"])
        # text = ''.join(f'<p>{para}</p>' for para in project["description"].split('\n\n'))
        # st.markdown(f'<div style="text-align: justify;">{text}</div>', unsafe_allow_html=True)
        if "link" in project:
            st.link_button("Github Repo", url=project["link"])
    else:
        st.error("Project not found.")

# Load project data from JSON file
with open("assets/project_data.json", "r") as f:
    projects = json.load(f)


DESCRIPTION_LIMIT = 150

# Display projects in rows of max 3 columns
for i in range(0, len(projects), 3):
    row_projects = projects[i:i+3]
    cols = st.columns(3)
    for idx, project in enumerate(row_projects):
        with cols[idx].container(border=True):
            st.image(project["image"])
            st.markdown(f"#### {project['title']}")

            display_description = project["description"]
            if len(display_description) > DESCRIPTION_LIMIT:
                display_description = display_description[:DESCRIPTION_LIMIT] + "..."
            st.write(display_description)

            colA, colB = st.columns([0.6, 0.4])
            if "tech_stack" in project:
                badges = " ".join([f":blue-badge[{tech}]" for tech in project["tech_stack"]])
                colA.markdown(badges)

            # print(len(project["tech_stack_logos"]))
            # colsa = st.columns(len(project["tech_stack_logos"]))
            # if "tech_stack_logos" in project:
            #     for idex, logo in enumerate(project["tech_stack_logos"]):
            #         colsa[idex].image(logo, width=24, use_container_width=False)

            if colB.button("View Project", key=project["button_key"]):
                dialog(project["button_key"])