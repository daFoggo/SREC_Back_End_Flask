import random
import sqlite3

def get_interview_questions():
    try:
        conn = sqlite3.connect("./databases/srec.db")
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM question_data where question_type = 'Basic questions'")
        basic_questions = cursor.fetchall()
        cursor.execute('SELECT * FROM question_data where question_type LIKE "Behavioral%"')
        behavioral_questions = cursor.fetchall()
        cursor.execute('SELECT * FROM question_data where question_type LIKE "Salary%"')
        salary_questions = cursor.fetchall()

        conn.close()

        random_basic_index = random.randint(0, len(basic_questions) - 1)
        random_behavioral_indices = random.sample(range(0, len(behavioral_questions)), 2)
        random_salary_index = random.randint(0, len(salary_questions) - 1)

        basic_question = basic_questions[random_basic_index]
        behavioral_question_1 = behavioral_questions[random_behavioral_indices[0]]
        behavioral_question_2 = behavioral_questions[random_behavioral_indices[1]]
        salary_question = salary_questions[random_salary_index]

        result = {
            "basic_questions": [{
                "id": basic_question[0],
                "question_type": basic_question[1],
                "question": basic_question[2],
                "answer": basic_question[3]
            }],
            "behavioral_questions": [{
                "id": behavioral_question_1[0],
                "question_type": behavioral_question_1[1],
                "question": behavioral_question_1[2],
                "answer": behavioral_question_1[3]
            },
            {
                "id": behavioral_question_2[0],
                "question_type": behavioral_question_2[1],
                "question": behavioral_question_2[2],
                "answer": behavioral_question_2[3]
            }],
            "salary_questions": [{
                "id": salary_question[0],
                "question_type": salary_question[1],
                "question": salary_question[2],
                "answer": salary_question[3]
            }]
        }
        
        return result
    
    except Exception as e:
        print(f"Error fetching interview questions: {str(e)}")
        return None
