import streamlit as st
from brd_reader import brd_reader
from test_case_generator2 import test_case_generator2

def show_logo():
    st.image("Bank_Muscat_logo.png", width=200)

st.set_page_config(layout="wide")
st.sidebar.title("Navigation")
#selection = st.sidebar.radio("Go to", ["Welcome 🏠", "Test Case Generator 🧪", "Test Case Generator 2 🧪🧪"])
selection = st.sidebar.radio("Go to", ["Welcome 🏠", "Test Case Generator 🧪", "BRD Reader 🧪🧪"])
# Navigation logic
if selection == "Welcome 🏠":
    show_logo()  # 👈 Add this
    # Page title
    st.markdown("<h1 style='text-align: center;'>Welcome</h1>", unsafe_allow_html=True)
    st.write("""
        Welcome to the GenAI Testcase Generation application! 
        This powerful tool leverages advanced AI capabilities to streamline the creation of 
        test cases directly from user stories. By automating this crucial yet time-consuming 
        task, our tool helps developers and QA engineers save valuable time, reduce human error,
        and ensure comprehensive test coverage. Experience how this tool can enhance your 
        development workflow.
    """)
# elif selection == "Test Case Generator 🧪":
#     test_case_generator()
elif selection == "Test Case Generator 🧪":
    test_case_generator2()
elif selection == "BRD Reader 🧪🧪":
    brd_reader()

