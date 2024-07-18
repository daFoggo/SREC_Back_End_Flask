import os
import sqlite3
import json
import random
import string
import secrets
import uuid
import base64
import joblib
import numpy as np
from functools import lru_cache

from flask import Flask, request, jsonify, make_response
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    jwt_required,
    get_jwt_identity,
)
from flask_cors import CORS, cross_origin
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from datetime import datetime
from flask_mail import Mail, Message

from extract_cv import extract_cv
from get_jobs import get_jobs

from junior_get_code import junior_get_code
from middle_get_code import middle_get_code
from senior_get_code import senior_get_code
from coding_conventions_checker import get_error, coding_conventions_checker

from get_interview_questions import get_interview_questions
from core_answer_matching import answer_matching
from virtual_interview_analyze import prediction
from flask import Response
from flask import render_template

load_dotenv()

app = Flask(__name__)
app.json.sort_keys = False

app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 465
app.config["MAIL_USE_SSL"] = True
app.config["MAIL_USERNAME"] = "srecproduct@gmail.com"
app.config["MAIL_PASSWORD"] = os.getenv("GMAIL_APP_PASSWORD")
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY")

jwt = JWTManager(app)
CORS(app, resources={r"*": {"origins": "*"}})

mail = Mail(app)


def get_db_connection():
    conn = sqlite3.connect("./databases/srec.db")
    conn.row_factory = sqlite3.Row
    return conn


# @app.route("/register", methods=["POST"])
# def register():
#     data = request.get_json()
#     full_name = data.get("full_name")
#     email = data.get("email")
#     password = data.get("password")
#     role = data.get("role", "candidate")
#     job_level = data.get("job_level", "none")
#     hashed_password = generate_password_hash(password, method="pbkdf2:sha256")

#     conn = get_db_connection()
#     cursor = conn.cursor()

#     if role == "recruiter":
#         cursor.execute("SELECT * FROM recruiters WHERE email = ?", (email,))
#         existing_user = cursor.fetchone()
#         if existing_user:
#             conn.close()
#             return jsonify({"msg": "This email is already registered"}), 409
#         cursor.execute(
#             "INSERT INTO recruiters (full_name, email, password, role) VALUES (?, ?, ?, ?, ?)",
#             (full_name, email, hashed_password, role),
#         )
#         conn.commit()
#         return jsonify({"msg": "Recruiter registered successfully"}), 200
#     else:
#         cursor.execute("SELECT * FROM candidates WHERE email = ?", (email,))
#         existing_user = cursor.fetchone()
#         if existing_user:
#             conn.close()
#             return jsonify({"msg": "This email is already registered"}), 409
#         cursor.execute(
#             "INSERT INTO candidates (full_name, email, password, role, level) VALUES (?, ?, ?, ?, ?, ?)",
#             (full_name, email, hashed_password, role, job_level),
#         )
#         conn.commit()
#         return jsonify({"msg": "Candidate registered successfully"}), 200


@app.route("/login/recruiter", methods=["POST"])
@cross_origin()
def login_recruiter():
    data = request.get_json()
    user_name = data.get("userName")
    password = data.get("password")

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT * FROM recruiters
        WHERE user_name = ?
        """,
        (user_name,),
    )
    user = cursor.fetchone()
    conn.close()

    if user and user["password"] == password:
        current_time = datetime.utcnow()
        access_token = create_access_token(
            identity={
                "id": user["recruiter_id"],
                "user_name": user["user_name"],
                "role": "recruiter",
                "full_name": user["full_name"],
                "current_time": current_time,
            }
        )
        return jsonify(access_token=access_token), 200
    else:
        return jsonify({"msg": "Invalid username or password"}), 401


@app.route("/login/candidate", methods=["POST"])
@cross_origin()
def login_candidate():
    data = request.get_json()
    user_name = data.get("userName")
    password = data.get("password")

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT * FROM candidates
        WHERE user_name = ?
        """,
        (user_name,),
    )
    user = cursor.fetchone()
    conn.close()

    if user and user["password"] == password:
        current_time = datetime.utcnow()
        access_token = create_access_token(
            identity={
                "id": user["candidate_id"],
                "user_name": user["user_name"],
                "role": "candidate",
                "job_level": user["level"].lower(),
                "full_name": user["full_name"],
                "current_time": current_time,
            }
        )
        return jsonify(access_token=access_token), 200
    else:
        return jsonify({"msg": "Invalid username or password"}), 401


@app.route("/logout", methods=["POST"])
@jwt_required()
def logout():

    return jsonify({"msg": "User logged out successfully"}), 200


def role_required(role):
    def wrapper(fn):
        @wraps(fn)
        @jwt_required()
        def decorated_view(*args, **kwargs):
            claims = get_jwt_identity()
            if claims["role"] != role:
                return (
                    jsonify({"msg": "Access forbidden: insufficient permissions"}),
                    403,
                )
            return fn(*args, **kwargs)

        return decorated_view

    return wrapper


import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from flask import jsonify, request
from flask_cors import cross_origin
import logging
from threading import Lock

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

assessment_lock = Lock()


# CV-matching module
def generate_password():
    alphabet = string.ascii_letters + string.digits + string.punctuation
    password = "".join(secrets.choice(alphabet) for i in range(12))
    return password


def generate_username(full_name, email):
    username_from_email = email.split("@")[0]
    username_from_email = "".join(filter(str.isalnum, username_from_email)).lower()
    random_suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=5))
    username = f"{username_from_email}_{random_suffix}"

    return username


def generate_candidate_id():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM candidates")
    count = cursor.fetchone()[0]
    conn.close()

    return f"CDD{count + 1}"


@app.route("/get-jobs", methods=["GET"])
@cross_origin()
def get_job():
    result = get_jobs()
    return jsonify(result), 200


@app.route("/get-ranked-candidates", methods=["POST", "GET"])
@cross_origin()
def cvs_matching():
    id = request.get_json().get("id")

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT job_path FROM jobs WHERE id = ?", (id,))
    job_path = cursor.fetchall()[0][0]

    data_path = os.path.join(job_path, "ranked_candidates.json")
    if not os.path.isfile(data_path):
        candidates = extract_cv(job_path, id)
        with open(
            os.path.join(job_path, "ranked_candidates.json"), "w", encoding="utf-8"
        ) as file:
            json.dump(candidates, file)
    with open(data_path, "r") as file:
        data = json.load(file)
        conn.close()
        return jsonify(data), 200
    conn.close()
    return jsonify({"error": "Not found data"}), 425


@app.route("/update-cvs", methods=["POST"])
@cross_origin()
def update_cvs():
    id = request.get_json().get("id")
    files = request.files.getlist("file")

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT job_path FROM jobs WHERE id = ?", (id,))
    job_path = cursor.fetchall()[0][0]
    update_path = os.path.join(os.path.abspath("__file__"), f"update_cvs_for_{id}")
    if not os.path.exists(update_path):
        os.makedirs(update_path)
    for file in files:
        file.save(job_path)
        file.save(update_path)
    update_candidates = extract_cv(update_path, id)
    with open(os.path.join(job_path, "ranked_candidates.json"), "r") as file:
        candidates = json.load(file)
        candidates.append(update_candidates)
        with open(os.path.join(job_path, "ranked_candidates.json"), "w") as f:
            json.dump(candidates, f)
    conn.close()
    os.remove(update_path)
    return "success"


from flask import jsonify, request, render_template_string
from flask_mail import Message
from flask_cors import cross_origin
import sqlite3
import uuid


@app.route("/generate-account-and-send-email", methods=["POST"])
@cross_origin()
def generate_account_and_send_email():
    data = request.get_json()
    candidates = data.get("candidates", [])
    recruiter_id = data.get("recruiter_id")
    job_id = data.get("job_id")
    job_level = data.get("job_level")

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        for candidate in candidates:
            full_name = candidate.get("name")
            email = candidate.get("gmail")
            role = "candidate"
            level = job_level
            user_name = generate_username(full_name, email)
            password = generate_password()

            cursor.execute(
                """
                INSERT INTO candidates 
                (candidate_id, full_name, email, user_name, password, role, level)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    generate_candidate_id(),
                    full_name,
                    email,
                    user_name,
                    password,
                    role,
                    level,
                ),
            )

            cursor.execute(
                "SELECT candidate_id FROM candidates WHERE email = ?", (email,)
            )
            candidate_id = cursor.fetchone()[0]

            cursor.execute(
                """
                SELECT COUNT(*) 
                FROM link_recruiter_with_candidate 
                WHERE recruiter_id = ? AND candidate_id = ? AND job_id = ?
                """,
                (recruiter_id, candidate_id, job_id),
            )
            result = cursor.fetchone()

            if result[0] == 0:
                link_id = str(uuid.uuid4())
                cursor.execute(
                    """
                    INSERT INTO link_recruiter_with_candidate 
                    (link_id, recruiter_id, candidate_id, job_id) 
                    VALUES (?, ?, ?, ?)
                    """,
                    (link_id, recruiter_id, candidate_id, job_id),
                )
                conn.commit()

            matching_id = str(uuid.uuid4())
            matching_score = candidate.get("cv_matching") * 100
            cursor.execute(
                """
                INSERT OR REPLACE INTO cv_matching_scores 
                (matching_id, candidate_id, job_id, matching_score)
                VALUES (?, ?, ?, ?)
                """,
                (matching_id, candidate_id, job_id, matching_score),
            )
            conn.commit()

            msg = Message(
                "Your SREC Account Information",
                sender="srecproduct@gmail.com",
                recipients=[email],
            )

            msg.html = render_template(
                "account_info.html",
                full_name=full_name,
                user_name=user_name,
                password=password,
            )

            mail.send(msg)

        conn.close()
        return (
            jsonify(
                {
                    "msg": "Candidates accounts created successfully, linked with recruiter, and emails sent"
                }
            ),
            200,
        )

    except sqlite3.Error as e:
        conn.rollback()
        conn.close()
        return jsonify({"msg": f"Database error: {str(e)}"}), 500

    except Exception as e:
        conn.rollback()
        conn.close()
        return jsonify({"msg": f"Unexpected error: {str(e)}"}), 500


# Code assessment module
@app.route("/get-code-assessment-scores", methods=["POST"])
@cross_origin()
def get_code_assessment_scores(**kwargs):
    logger.debug("Starting get_code_assessment_scores function")
    data = request.get_json()
    logger.debug(f"Received data: {data}")
    candidate_id = data.get("candidate_id")
    level = data.get("job_level")

    logger.debug(f"Received request for candidate_id: {candidate_id}, level: {level}")

    if candidate_id is None or level is None:
        return jsonify({"msg": "Candidate ID and job level are required"}), 400

    if level not in ["junior", "middle", "senior"]:
        return jsonify({"msg": "Invalid job level"}), 400

    assessment_data = []
    problem_data = {}

    with assessment_lock:
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            logger.debug("Checking existing records")
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM code_assessment_scores
                WHERE candidate_id = ? 
                """,
                (candidate_id,),
            )
            count = cursor.fetchone()[0]
            logger.debug(
                f"Found {count} existing records for candidate_id: {candidate_id}"
            )

            if count == 0:
                logger.debug("No existing records found, creating new assessment")

                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM code_assessment_scores
                    WHERE candidate_id = ? AND status = 0   
                    """,
                    (candidate_id,),
                )
                recheck_count = cursor.fetchone()[0]

                if recheck_count == 0:
                    if level == "junior":
                        code_data = junior_get_code()
                    elif level == "middle":
                        code_data = middle_get_code()
                    elif level == "senior":
                        code_data = senior_get_code()

                    code_assessment_scores_data = []
                    for i in range(3):
                        assessment_id = str(uuid.uuid4())
                        problem_id = code_data[f"test_{i + 1}"]["ID"]
                        code_assessment_scores_data.append(
                            (
                                str(assessment_id),
                                candidate_id,
                                problem_id,
                                0,
                                "",
                                "",
                                False,
                            )
                        )
                        problem_data[problem_id] = {
                            "id": problem_id,
                            "source": code_data[f"test_{i + 1}"]["source"],
                            "name": code_data[f"test_{i + 1}"]["name"],
                            "description": code_data[f"test_{i + 1}"]["description"],
                            "public_input": code_data[f"test_{i + 1}"]["public_input"],
                            "public_output": code_data[f"test_{i + 1}"][
                                "public_output"
                            ],
                            "gen_input": code_data[f"test_{i + 1}"]["gen_input"],
                            "gen_output": code_data[f"test_{i + 1}"]["gen_output"],
                            "difficulty": code_data[f"test_{i + 1}"]["difficulty"],
                            "second": code_data[f"test_{i + 1}"]["second"],
                            "nano": code_data[f"test_{i + 1}"]["nano"],
                            "memory_limit_bytes": code_data[f"test_{i + 1}"][
                                "memory_limit_bytes"
                            ],
                        }

                    logger.debug(
                        f"Prepared {len(code_assessment_scores_data)} records for insertion"
                    )

                    cursor.executemany(
                        """
                        INSERT INTO code_assessment_scores (assessment_id, candidate_id, problem_id, assessment_score, code_convention_comment, code_convention_score, status)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        code_assessment_scores_data,
                    )

                    conn.commit()
                    logger.debug(
                        f"Inserted {len(code_assessment_scores_data)} new records"
                    )

                    assessment_data = code_assessment_scores_data
                else:
                    logger.debug(
                        f"Records were created by another process. Found {recheck_count} records on recheck."
                    )
            else:
                cursor.execute(
                    """
                    SELECT *
                    FROM code_assessment_scores a
                    JOIN code_data b ON a.problem_id = b.id
                    WHERE a.candidate_id = ? AND a.status = 0
                    """,
                    (candidate_id,),
                )
                existing_data = cursor.fetchall()
                print(existing_data)

                for index, row in enumerate(existing_data):
                    print(row[2])
                    assessment_data.append(
                        {
                            "assessment_id": row[0],
                            "candidate_id": row[1],
                            "problem_id": row[2],
                            "assessment_score": row[3],
                            "code_convention_comment": row[4],
                            "code_convention_score": row[5],
                            "status": row[6],
                        }
                    )
                    problem_data[f"{index}_{row[2]}"] = {
                        "id": row[7],
                        "source": row[8],
                        "name": row[9],
                        "description": row[10],
                        "public_input": row[11],
                        "public_output": row[12],
                        "gen_input": row[13],
                        "gen_output": row[14],
                        "difficulty": row[15],
                        "second": row[16],
                        "nano": row[17],
                        "memory_limit_bytes": row[18],
                    }

            cursor.execute(
                """
                SELECT COUNT(*)
                FROM code_assessment_scores
                WHERE candidate_id = ? AND status = 0
                """,
                (candidate_id,),
            )

            incomplete_count = cursor.fetchone()[0]
            if incomplete_count == 3:
                current_problem_index = 1
            elif incomplete_count == 2:
                current_problem_index = 2
            elif incomplete_count == 1:
                current_problem_index = 3
            else:
                current_problem_index = 4

            logger.debug(
                f"Final incomplete_count: {incomplete_count}, current_problem_index: {current_problem_index}"
            )

        except Exception as e:
            conn.rollback()
            logger.error(f"Error processing assessment scores: {str(e)}")
            return (
                jsonify({"msg": f"Error processing assessment scores: {str(e)}"}),
                500,
            )

        finally:
            conn.close()

    logger.debug(
        f"Returning response with {len(assessment_data)} assessment data entries"
    )
    print(problem_data.keys())
    return (
        jsonify(
            {
                "assessment_data": assessment_data,
                "current_problem_index": current_problem_index,
                "problem_data": problem_data,
            }
        ),
        200,
    )


@app.route("/submit-code-assessment-scores", methods=["PUT"])
@cross_origin()
def submit_code_assessment_scores():
    data = request.get_json()
    assessment_id = data.get("assessment_id")
    assessment_score = data.get("assessment_score")
    status = data.get("status")
    code_solution = data.get("code_solution")
    language = data.get("selected_language")

    if (
        assessment_id is None
        or assessment_score is None
        or status is None
        or code_solution is None
        or language is None
    ):
        return jsonify({"msg": "Data insufficiency"}), 400

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        code_convention_score, code_convention_comment = coding_conventions_checker(
            language, code_solution
        )

        cursor.execute(
            """
            UPDATE code_assessment_scores   
            SET 
            assessment_score = ?,
            code_convention_comment = ?,
            code_convention_score = ?,
            status = ?
            WHERE assessment_id = ?
            """,
            (
                assessment_score,
                code_convention_comment,
                code_convention_score,
                status,
                assessment_id,
            ),
        )

        if cursor.rowcount == 0:
            conn.rollback()
            return jsonify({"msg": "Update failed. Assessment ID not found."}), 404

        conn.commit()
    except Exception as e:
        return jsonify({"msg": f"Database error: {str(e)}"}), 500
    finally:
        cursor.close()
        conn.close()

    return jsonify({"msg": "Update successful"}), 200


@app.route("/get-final-code-score", methods=["POST"])
@cross_origin()
def get_final_code_score():
    data = request.get_json()
    candidate_id = data.get("candidate_id")
    if candidate_id is None:
        return jsonify({"msg": "Candidate ID is required"}), 400

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT *, 
                   (SELECT SUM(assessment_score) 
                    FROM code_assessment_scores 
                    WHERE candidate_id = ? AND status = 1) AS final_score
            FROM code_assessment_scores
            WHERE candidate_id = ? AND status = 1
            """,
            (candidate_id, candidate_id),
        )

        rows = cursor.fetchall()
        if not rows:
            return jsonify({"msg": "No assessments found for the candidate"}), 404

        final_score = rows[0]["final_score"]

        assessment_data = [
            {column: row[column] for column in row.keys() if column != "final_score"}
            for row in rows
        ]

        return jsonify({"assessment_data": assessment_data, "final_score": final_score})

    except Exception as e:
        return jsonify({"msg": str(e)}), 500

    finally:
        conn.close()


# Survey module :
with open("./static/survey/data.json", "r", encoding="utf-8") as f:
    data = json.load(f)
categories = {
    "EXT": {f"EXT{i}": [] for i in range(1, 11)},
    "EST": {f"EST{i}": [] for i in range(1, 11)},
    "OPN": {f"OPN{i}": [] for i in range(1, 11)},
    "AGR": {f"AGR{i}": [] for i in range(1, 11)},
    "CSN": {f"CSN{i}": [] for i in range(1, 11)},
}
for item in data:
    category = item["Type"][:3]
    subcategory = item["Type"]
    if category in categories and subcategory in categories[category]:
        categories[category][subcategory].append(item)


@app.route("/random_questions", methods=["GET"])
def get_random_questions():
    selected_questions = []
    for category, subcategories in categories.items():
        for subcategory, questions in subcategories.items():
            if questions:
                selected_questions.extend(random.sample(questions, 1))
    if len(selected_questions) != 50:
        return (
            jsonify(
                {"error": f"Expected 50 questions, but got {len(selected_questions)}"}
            ),
            400,
        )
    final_questions = []
    main_categories = {"EXT": [], "EST": [], "OPN": [], "AGR": [], "CSN": []}
    for question in selected_questions:
        category = question["Type"][:3]
        if category in main_categories:
            main_categories[category].append(question)
    for category, questions in main_categories.items():
        if len(questions) >= 4:
            final_questions.extend(random.sample(questions, 4))
    if len(final_questions) != 20:
        return (
            jsonify(
                {"error": f"Expected 20 questions, but got {len(final_questions)}"}
            ),
            400,
        )
    for idx, question in enumerate(final_questions, start=1):
        question["ID"] = f"Q{idx}"
    return jsonify(final_questions)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON format or empty data"}), 400

        # Đảm bảo data là một list các item
        if not isinstance(data, list):
            data = [data]

        features = []
        for item in data:
            try:
                answers = item["Answer"]
                features.append([answers[f"Q{i+1}"] for i in range(20)])
            except KeyError as e:
                return jsonify({"error": f"Missing key in Answer: {str(e)}"}), 400

        features = np.array(features)

        kmeans_model = joblib.load("./static/survey/kmeans_model.joblib")
        cluster_means = joblib.load("./static/survey/cluster_means.joblib")

        cluster_labels = kmeans_model.predict(features)
        mul_lr = joblib.load("./static/survey/logis.joblib")

        result = []
        for item, label in zip(data, cluster_labels):
            means = cluster_means.iloc[label]

            cluster_data = means[
                [
                    "openess",
                    "neuroticism",
                    "conscientiousness",
                    "agreeableness",
                    "extroversion",
                ]
            ]
            cluster_array = cluster_data.values.reshape(1, -1)
            cluster_probabilities = mul_lr.predict_proba(cluster_array)[0]

            label_probabilities = []
            for lr_label, prob in zip(mul_lr.classes_, cluster_probabilities):
                label_probabilities.append(
                    {"label": lr_label, "probability": float(prob)}
                )

            item_with_label = item.copy()
            item_with_label["cluster"] = int(label)
            item_with_label["probabilities"] = label_probabilities
            result.append(item_with_label)

        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Virtual-Interview-Module

interview_lock = Lock()


@lru_cache(maxsize=100)
def file_to_b64(file_path):
    try:
        with open(file_path, "rb") as file:
            file_content = file.read()
            base64_string = base64.b64encode(file_content).decode("utf-8")
        return base64_string
    except Exception as e:
        print(f"Error converting file to base64: {e}")
        return None


def fetch_questions(question_type):
    with sqlite3.connect("./databases/srec.db") as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM question_data WHERE question_type LIKE ?"
        cursor.execute(query, (question_type,))
        return cursor.fetchall()


@app.route("/get-virtual-interview-scores", methods=["POST", "GET"])
@cross_origin()
def get_virtual_interview_scores():
    logger.debug("Starting get_virtual_interview_scores function")
    data = request.get_json()
    candidate_id = data.get("candidate_id")

    logger.debug(f"Received request for candidate_id: {candidate_id}")

    if candidate_id is None:
        return jsonify({"msg": "Candidate ID is required"}), 400

    interview_data = []
    question_data = {
        "basic_questions": [],
        "behavioral_questions": [],
        "salary_questions": [],
    }
    current_question_index = 0

    with interview_lock:
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            logger.debug("Checking existing records")
            cursor.execute(
                "SELECT COUNT(*) FROM virtual_interview_scores WHERE candidate_id = ?",
                (candidate_id,),
            )
            count = cursor.fetchone()[0]
            logger.debug(
                f"Found {count} existing records for candidate_id: {candidate_id}"
            )

            if count == 0:
                logger.debug("No existing records found, creating new interview")
                fetched_question_data = get_interview_questions()
                logger.debug(f"Question data retrieved: {fetched_question_data}")

                virtual_interview_scores_data = []
                for category in [
                    "basic_questions",
                    "behavioral_questions",
                    "salary_questions",
                ]:
                    if category in fetched_question_data:
                        for question in fetched_question_data[category]:
                            interview_id = str(uuid.uuid4())
                            virtual_interview_scores_data.append(
                                (
                                    interview_id,
                                    candidate_id,
                                    question["id"],
                                    "",
                                    "",
                                    "",
                                    False,
                                )
                            )
                            question_data[category].append(question)

                logger.debug(
                    f"Prepared {len(virtual_interview_scores_data)} records for insertion"
                )

                if virtual_interview_scores_data:
                    cursor.executemany(
                        """
                        INSERT INTO virtual_interview_scores 
                        (interview_id, candidate_id, question_id, analysis_id, video_path, speech_to_text, status)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        virtual_interview_scores_data,
                    )

                    conn.commit()
                    logger.debug(
                        f"Inserted {len(virtual_interview_scores_data)} new records"
                    )
                else:
                    logger.error("No interview scores data prepared for insertion")

                interview_data = [
                    {
                        "interview_id": row[0],
                        "candidate_id": row[1],
                        "question_id": row[2],
                        "analysis_id": row[3],
                        "video_path": row[4],
                        "speech_to_text": row[5],
                        "status": row[6],
                    }
                    for row in virtual_interview_scores_data
                ]
            else:
                logger.debug("Fetching existing records")
                cursor.execute(
                    """
                    SELECT *
                    FROM virtual_interview_scores
                    WHERE candidate_id = ? AND status = 0
                    """,
                    (candidate_id,),
                )
                existing_data = cursor.fetchall()
                logger.debug(f"Fetched {len(existing_data)} existing records")

                interview_data = [
                    {
                        "interview_id": row[0],
                        "candidate_id": row[1],
                        "question_id": row[2],
                        "analysis_id": row[3],
                        "video_path": row[4],
                        "speech_to_text": row[5],
                        "status": row[6],
                    }
                    for row in existing_data
                ]

                for row in existing_data:
                    cursor.execute(
                        """
                        SELECT *
                        FROM question_data
                        WHERE question_id = ?
                        """,
                        (row[2],),
                    )
                    logger.debug(f"Fetching question data for question_id: {row[2]}")
                    question = cursor.fetchone()
                    if question:
                        question_obj = {
                            "question_id": row[2],
                            "question_type": question[1],
                            "question": question[2],
                            "answer": question[3],
                            "video_data": file_to_b64(question[4]),
                        }
                        logger.debug(f"Question data fetched: {question_obj}")
                        if "Basic questions" in question[1]:
                            question_data["basic_questions"].append(question_obj)
                        elif "Behavioral questions" in question[1]:
                            question_data["behavioral_questions"].append(question_obj)
                        elif "Salary questions" in question[1]:
                            question_data["salary_questions"].append(question_obj)

            cursor.execute(
                """
                SELECT COUNT(*)
                FROM virtual_interview_scores
                WHERE candidate_id = ? AND status = 0
                """,
                (candidate_id,),
            )

            incomplete_count = cursor.fetchone()[0]
            if incomplete_count == 4:
                current_question_index = 1
            elif incomplete_count == 3:
                current_question_index = 2
            elif incomplete_count == 2:
                current_question_index = 3
            elif incomplete_count == 1:
                current_question_index = 4
            else:
                current_question_index = 5

            logger.debug(
                f"Final incomplete_count: {incomplete_count}, current_question_index: {current_question_index}"
            )
        except Exception as e:
            conn.rollback()
            logger.error(f"Error processing interview scores: {str(e)}")
            logger.error(f"Error details: {e.__class__.__name__}: {str(e)}")
            return jsonify({"msg": f"Error processing interview scores: {str(e)}"}), 500

        finally:
            conn.close()

    logger.debug(
        f"Preparing response with {len(interview_data)} interview data entries"
    )

    response_data = {
        "question_data": question_data,
        "current_question_index": current_question_index,
        "interview_data": interview_data,
    }

    return jsonify(response_data), 200


import traceback


def b64_to_file(base64_string, filename, folder="./static/recorded_videos"):
    try:
        if not os.path.exists(folder):
            os.makedirs(folder)
        file_path = os.path.join(folder, filename)
        with open(file_path, "wb") as file:
            file.write(base64.b64decode(base64_string))
        return file_path
    except Exception as e:
        print(f"Error saving video file: {e}")
        return None


@app.route("/submit-virtual-interview-scores", methods=["PUT"])
@cross_origin()
def submit_virtual_interview_scores():
    data = request.get_json()
    interview_id = data.get("interview_id")
    candidate_id = data.get("candidate_id")
    current_question_index = data.get("current_question_index")
    status = data.get("status")
    video_data = data.get("video_data")

    if candidate_id is None or status is None:
        return jsonify({"msg": "Data insufficiency"}), 400

    pathFile = b64_to_file(
        video_data,
        f"{candidate_id}_{current_question_index}_capture.mp4",
        folder="./static/recorded_videos",
    )
    if pathFile is None:
        return jsonify({"msg": "Failed to save video file"}), 500

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE virtual_interview_scores 
            SET video_path = ?, status = ?
            WHERE interview_id = ?
            """,
            (pathFile, status, interview_id),
        )
        conn.commit()
        return jsonify({"msg": "Interview score submitted successfully"}), 200
    except Exception as e:
        conn.rollback()
        error_message = str(e)
        stack_trace = traceback.format_exc()
        print(f"Database error: {error_message}")
        print(f"Stack trace: {stack_trace}")
        return jsonify({"msg": "Database error", "error": error_message}), 500
    finally:
        if conn:
            conn.close()


@app.route("/get-interview-questions")
@cross_origin()
def get_interview_questions_route():
    questions = get_interview_questions()
    return questions


@app.route("/virtual-interview-analyze", methods=["POST", "GET"])
def analyze():
    conn = None
    try:
        data = request.get_json()
        candidate_id = data.get("candidate_id")
        current_question_index = data.get("current_question_index")
        interview_id = data.get("interview_id")
        question_id = int(data.get("question_id"))

        if not interview_id or not question_id:
            return jsonify({"error": "interview_id and question_id are required"}), 400

        filepath = f"./static/recorded_videos/{candidate_id}_{current_question_index}_capture.mp4"
        logger.debug(f"Analyzing video file: {filepath}")
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT answer
            FROM question_data
            WHERE question_id = ?
            """,
            (question_id,),
        )

        core_answer_row = cursor.fetchone()
        if not core_answer_row:
            return jsonify({"error": "Invalid question_id provided"}), 400

        core_answer = core_answer_row["answer"]

        analyze_result = prediction(filepath)
        analyze_result_json = json.dumps(analyze_result)
        cosine_sim = answer_matching(filepath, core_answer)

        id = str(uuid.uuid4())

        cursor.execute(
            """
            INSERT INTO video_analysis_data (analysis_id, prediction_data, answer_matching_data)
            VALUES (?, ?, ?)
            """,
            (id, analyze_result_json, cosine_sim),
        )

        cursor.execute(
            """
            UPDATE virtual_interview_scores
            SET analysis_id = ?
            WHERE interview_id = ?
            """,
            (id, interview_id),
        )

        conn.commit()

        return jsonify({"msg": "Inserted data successfully"}), 200

    except Exception as e:
        if conn:
            conn.rollback()
        return jsonify({"error": f"Unable to analyze the video: {str(e)}"}), 420
    finally:
        if conn:
            conn.close()


# Summary module
@app.route("/get-recruiter-with-candidates", methods=["POST"])
@cross_origin()
def get_recruiter_with_candidates():
    data = request.get_json()
    recruiter_id = data.get("recruiter_id")
    job_id = data.get("job_id")

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT c.candidate_id, c.full_name, c.email, c.level
        FROM candidates c
        JOIN link_recruiter_with_candidate l
        ON c.candidate_id = l.candidate_id
        WHERE l.recruiter_id = ? AND l.job_id = ?
        """,
        (recruiter_id, job_id),
    )

    rows = cursor.fetchall()
    candidates = [dict(row) for row in rows]

    conn.close()

    return jsonify(candidates), 200


@app.route("/get-summary-cv-matching", methods=["POST"])
@cross_origin()
def get_summary_cv_matching():
    data = request.get_json()
    candidate_id = data.get("candidate_id")
    job_id = data.get("job_id")

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT matching_score FROM cv_matching_scores
        WHERE candidate_id = ? AND job_id = ?
        """,
        (candidate_id, job_id),
    )
    matching_score = cursor.fetchone()

    cursor.execute(
        """
        SELECT * FROM candidates
        WHERE candidate_id = ?
        """,
        (candidate_id,),
    )
    candidate_data = cursor.fetchone()

    conn.close()

    matching_score_dict = (
        {"matching_score": matching_score[0]} if matching_score else None
    )
    candidate_data_dict = dict(candidate_data) if candidate_data else None

    return jsonify(
        {"matching_score": matching_score_dict, "candidate_data": candidate_data_dict}
    )


@app.route("/get-summary-virtual-interview", methods=["POST"])
@cross_origin()
def get_summary_virtual_interview():
    data = request.get_json()
    candidate_id = data.get("candidate_id")

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT 
            vis.*, 
            vad.* 
        FROM 
            virtual_interview_scores vis
        LEFT JOIN 
            video_analysis_data vad ON vis.analysis_id = vad.analysis_id
        WHERE 
            vis.candidate_id = ? AND vis.status = 1
        """,
        (candidate_id,),
    )
    interview_data = cursor.fetchall()

    conn.close()

    result = []
    for row in interview_data:
        result.append(dict(zip([column[0] for column in cursor.description], row)))

    return jsonify(result)


if __name__ == "__main__":
    app.run()
