# ==============================================================================
#  Schedulu: Agentic AI Study Companion (V6.2 - Final Date Fix)
#  app.py - Main Streamlit Application File
# ==============================================================================

# --- 1. Import Necessary Libraries ---
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, date
import google.generativeai as genai
import os
import numpy as np
import json
from ics import Calendar, Event

# --- 2. Configure the Application Page ---
st.set_page_config(
    page_title="Schedulu Agent",
    page_icon="🧠",
    layout="wide"
)

# --- 3. Gemini API Configuration ---
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
except (KeyError, Exception):
    st.error("🚨 Gemini API Key not found. Please add it to your `.streamlit/secrets.toml` file.")
    model = None

# --- 4. Helper Functions ---

def generate_topics_from_subject(subject, model):
    if not model: return ""
    prompt = (f"You are an expert curriculum planner. For '{subject}', list important study topics. "
              f"List each topic on a new line without numbers or bullets.")
    try:
        with st.spinner(f"🤖 Agent is researching topics..."):
            response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating topics: {e}")
        return ""

def get_mcq_quiz(topic, subject, model):
    if not model: return None
    prompt = (f"You are a quiz generator. For the topic '{topic}' in the subject '{subject}', "
              f"create a JSON array of 5 multiple-choice questions to test understanding. "
              f"Each JSON object must have three keys: 'question' (string), 'options' (an array of 4 strings), "
              f"and 'correct_answer' (a string that exactly matches one of the options). "
              f"Ensure the JSON is valid.")
    try:
        response = model.generate_content(prompt)
        cleaned_json = response.text.strip().replace("```json", "").replace("```", "")
        quiz_data = json.loads(cleaned_json)
        return quiz_data
    except (json.JSONDecodeError, Exception) as e:
        st.error(f"Failed to generate a valid quiz for '{topic}'. The AI may have returned an unexpected format.")
        return None

# Persistence Functions
def save_plan(df):
    df.to_csv('plan.csv', index=False)

# --- MODIFIED: The first fix is here ---
def load_plan():
    """Loads the study plan and ensures the 'Date' column is the correct type."""
    if os.path.exists('plan.csv'):
        try:
            df = pd.read_csv('plan.csv')
            # This is the key change: convert the column to datetime, then to standard python date objects.
            df['Date'] = pd.to_datetime(df['Date']).dt.date
            return df
        except Exception:
            return None # Handle corrupted or empty file
    return None

# --- MODIFIED: The second fix is here ---
def create_ics_file(schedule_df, subject):
    """Creates the string content for a downloadable .ics calendar file."""
    c = Calendar()
    for _, row in schedule_df.iterrows():
        e = Event()
        e.name = f"Study {subject}: {row['Topic']}"
        # Because we fixed the data type at the source, this line is now simple and correct.
        e.begin = row['Date']
        e.make_all_day()
        c.events.add(e)
    return str(c)

# --- 5. State Management and Initial Load ---
if 'plan_df' not in st.session_state:
    st.session_state.plan_df = load_plan()
if 'topic_for_quiz' not in st.session_state:
    st.session_state.topic_for_quiz = None
if 'quiz_data' not in st.session_state:
    st.session_state.quiz_data = None

# --- 6. UI: Sidebar ---
# (Unchanged)
with st.sidebar:
    st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=200)
    st.title("Schedulu Agent")
    st.divider()

    with st.expander("🚀 New Plan Setup", expanded=(st.session_state.plan_df is None)):
        subject = st.text_input("Subject", placeholder="e.g., Quantum Physics", key="subject_input")
        if st.button("🤖 Suggest Topics with AI", use_container_width=True):
            if subject:
                st.session_state.topics_string = generate_topics_from_subject(subject, model)
            else:
                st.warning("Please enter a subject first.")
        topics_str = st.text_area("📝 Study Topics", value=st.session_state.get('topics_string', ''), height=200)
        exam_date = st.date_input("Exam Date", min_value=datetime.now() + timedelta(days=1))
        
        if st.button("✨ Generate New Plan", type="primary", use_container_width=True):
            topics = [topic.strip() for topic in topics_str.split('\n') if topic.strip()]
            if topics:
                days_to_study = (exam_date - date.today()).days
                dates = [date.today() + timedelta(days=i) for i in range(days_to_study)]
                schedule_chunks = np.array_split(topics, days_to_study) if days_to_study > 0 else [topics]
                plan_data = []
                for i, topic_list in enumerate(schedule_chunks):
                    for topic in topic_list:
                        plan_data.append({'Date': dates[i], 'Topic': topic, 'Status': False})
                st.session_state.plan_df = pd.DataFrame(plan_data)
                save_plan(st.session_state.plan_df)
                st.rerun()

    if st.session_state.plan_df is not None:
        st.divider()
        st.header("⚙️ Plan Controls")
        
        ics_data = create_ics_file(st.session_state.plan_df, st.session_state.get('subject_input', 'Studies'))
        st.download_button(
            label="🗓️ Download Calendar File (.ics)",
            data=ics_data,
            file_name="study_schedule.ics",
            mime="text/calendar",
            use_container_width=True
        )
        
        if st.button("🗑️ Delete Plan & Start Over", use_container_width=True):
            if os.path.exists('plan.csv'): os.remove('plan.csv')
            st.session_state.plan_df = None
            st.rerun()

# --- 7. UI: Main Application Body ---
# (Unchanged)
st.title("🧠 Schedulu: Your Agentic Study Companion")
st.markdown("""
Welcome to **Schedulu**, your intelligent partner in exam preparation. This isn't just a scheduler; it's an **agentic AI** designed to proactively help you learn.

-   **AI-Powered Planning**: Tell it your subject, and the agent researches and suggests key topics.
-   **Interactive Checklist & Saved Progress**: Track your progress with a day-by-day plan. Your work is saved, so you can close the browser and your progress will be here when you return.
-   **MCQ Reflection Quizzes**: When you complete a topic, the agent generates a 5-question multiple-choice quiz to test your knowledge.
-   **Calendar Export**: Download your entire study plan as an `.ics` file, ready to import into Google Calendar, Apple Calendar, or Outlook.
""")

if st.session_state.plan_df is None:
    st.info("👋 Get started by creating a new plan in the sidebar!")
else:
    plan_df = st.session_state.plan_df
    
    if st.session_state.topic_for_quiz:
        if st.session_state.quiz_data is None:
            st.session_state.quiz_data = get_mcq_quiz(st.session_state.topic_for_quiz, st.session_state.get('subject_input', 'Studies'), model)
        if st.session_state.quiz_data:
            @st.dialog(f"🧠 Quiz: {st.session_state.topic_for_quiz}")
            def run_quiz():
                st.write("Test your understanding of the topic you just completed.")
                user_answers = {}
                for i, q in enumerate(st.session_state.quiz_data):
                    user_answers[i] = st.radio(q['question'], q['options'], key=f"q_{i}", index=None)
                if st.button("Submit Answers", type="primary"):
                    score = sum(1 for i, q in enumerate(st.session_state.quiz_data) if user_answers[i] == q['correct_answer'])
                    st.session_state.last_score = f"{score} / {len(st.session_state.quiz_data)}"
                    st.session_state.topic_for_quiz = None
                    st.session_state.quiz_data = None
                    st.rerun()
            run_quiz()
        else:
            st.session_state.topic_for_quiz = None

    if 'last_score' in st.session_state and st.session_state.last_score:
        st.success(f"Last quiz score: {st.session_state.last_score}")
        st.session_state.last_score = None

    st.header("✅ Your Study Checklist")
    edited_df = st.data_editor(
        plan_df,
        column_config={"Status": st.column_config.CheckboxColumn("Done?", default=False),
                       "Date": st.column_config.DateColumn("Date", format="D MMM YYYY")},
        disabled=["Date", "Topic"], hide_index=True, use_container_width=True, key="data_editor"
    )

    if not edited_df.equals(plan_df):
        st.session_state.plan_df = edited_df
        save_plan(edited_df)
        newly_completed = edited_df[edited_df["Status"] & ~plan_df["Status"]]
        if not newly_completed.empty:
            st.session_state.topic_for_quiz = newly_completed.iloc[0]['Topic']
            st.rerun()

    st.divider()
    st.header("📈 Progress at a Glance")
    total_topics = len(plan_df)
    completed_count = plan_df['Status'].sum()
    progress = completed_count / total_topics if total_topics > 0 else 0
    st.progress(progress, text=f"{progress:.0%} Complete ({completed_count} of {total_topics} topics)")