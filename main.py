import streamlit as st
from modules.show_attendance import show_attendance
import os
import pandas as pd
from datetime import datetime
import time

st.set_page_config(page_title="Face Recognition Attendance", layout="centered")
st.markdown("<h1 style='text-align: center;'>🎓 AI Face Attendance System</h1><hr>", unsafe_allow_html=True)

# ---------------------- ADD FACE ----------------------
st.subheader("➕ Add New Face")
name = st.text_input("Enter name:")
add_btn = st.button("Start Camera for Adding Face")

if add_btn:
    if name.strip() == "":
        st.warning("⚠️ Please enter a name first.")
    else:
        st.success(f"Starting camera to add face for: {name}")
        os.system(f"python modules/add_face.py {name}")
        time.sleep(2)
        if os.path.exists("status.txt"):
                    with open("status.txt", "r") as f:
                        message = f.read()
                    st.success(message)
                    os.remove("status.txt")
                    
               


# ---------------------- MARK ATTENDANCE ----------------------
st.subheader("🟢 Mark Attendance")
if st.button("Start Camera to Mark Attendance"):
    st.success("Launching webcam for attendance...")
    os.system("python -m modules.recognize")
    if os.path.exists("status.txt"):
        for _ in range(5):
            try:
                with open("status.txt", "r") as f:
                    message = f.read()
                st.success(message)
                os.remove("status.txt")
                break
            except PermissionError:
                time.sleep(0.3) 


# ---------------------- SHOW ATTENDANCE ----------------------
st.subheader("📋 Show Attendance")
if st.button("Show Today’s Records"):
    try:
        show_attendance()
    except pd.errors.EmptyDataError:
        st.warning("Attendance file is empty.")
