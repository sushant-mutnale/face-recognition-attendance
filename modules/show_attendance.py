import pandas as pd
import streamlit as st
import os
from datetime import datetime

def show_attendance():
    st.subheader("📋 Today's Attendance")

    today = datetime.now().strftime("%Y-%m-%d")
    filename = "attendance.csv"

    if os.path.exists(filename):
        df = pd.read_csv(filename)
        df_today = df[df["Date"] == today]

        if df_today.empty:
            st.warning("No attendance marked today.")
        else:
            st.dataframe(df_today)
    else:
        st.info("No attendance file found yet.")
