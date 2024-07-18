import random
import sqlite3
import os
import base64
from functools import lru_cache

DATABASE_PATH = "./databases/srec.db"
VIDEO_FOLDER = "./static/video_interview"

@lru_cache(maxsize=100)
def file_to_b64(filename, folder=VIDEO_FOLDER):
    try:
        file_path = os.path.join(folder, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, "rb") as file:
            file_content = file.read()
            base64_string = base64.b64encode(file_content).decode("utf-8")
        return base64_string
    except Exception as e:
        print(f"Error converting file to base64: {e}")
        return None

def fetch_questions(question_type):
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM question_data WHERE question_type LIKE ?"
        cursor.execute(query, (question_type,))
        return cursor.fetchall()

def select_random_questions(questions, count=1):
    return random.sample(questions, min(count, len(questions)))

def format_question(question):
    return {
        "id": question[0],
        "question_type": question[1],
        "question": question[2],
        "answer": question[3],
        "video_data": file_to_b64(question[4])
    }

def get_interview_questions():
    try:
        basic_questions = fetch_questions("Basic questions")
        behavioral_questions = fetch_questions("Behavioral%")
        salary_questions = fetch_questions("Salary%")

        result = {
            "basic_questions": [format_question(q) for q in select_random_questions(basic_questions)],
            "behavioral_questions": [format_question(q) for q in select_random_questions(behavioral_questions, 2)],
            "salary_questions": [format_question(q) for q in select_random_questions(salary_questions)]
        }
        
        return result
    
    except Exception as e:
        print(f"Error fetching interview questions: {str(e)}")
        return None