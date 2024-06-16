import os
import sqlite3
from flask import Flask, request, jsonify
from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required, get_jwt_identity
)
from flask_cors import CORS, cross_origin
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from datetime import datetime

from junior_get_code import junior_get_code
from middle_get_code import middle_get_code
from senior_get_code import senior_get_code


load_dotenv()

app = Flask(__name__)
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY")
jwt = JWTManager(app)
CORS(app, resources={r"*": {"origins": "*"}})

def get_db_connection():
    conn = sqlite3.connect("./databases/srec.db")
    conn.row_factory = sqlite3.Row
    return conn

@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    first_name = data.get("first_name")
    last_name = data.get("last_name")
    email = data.get("email")
    password = data.get("password")
    role = data.get("role", "candidate")
    job_level = data.get("job_level", "none")
    hashed_password = generate_password_hash(password, method="pbkdf2:sha256")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    existing_user = cursor.fetchone()
    if existing_user:
        conn.close()
        return jsonify({"msg": "This email is already registered"}), 409
    
    try:
        cursor.execute("INSERT INTO users (first_name, last_name, email, password, role, level) VALUES (?, ?, ?, ?, ?, ?)",
                       (first_name, last_name, email, hashed_password, role, job_level))
        conn.commit()
        return jsonify({"msg": "User registered successfully"}), 200
    finally:
        conn.close()
    
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email=?", (email,))
    user = cursor.fetchone()
    conn.close()
    
    if user and check_password_hash(user["password"], password):
        current_time = datetime.utcnow()
        access_token = create_access_token(identity={"email": email, "role": user["role"], "job_level": user["level"], "first_name": user["first_name"], "last_name": user["last_name"], "current_time": current_time})
        return jsonify(access_token=access_token), 200
    else:
        return jsonify({"msg": "Invalid email or password"}), 401

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
                return jsonify({"msg": "Access forbidden: insufficient permissions"}), 403
            return fn(*args, **kwargs)
        return decorated_view
    return wrapper

@app.route("/codeProblem", methods=["GET"])
@role_required("candidate")
def candidate_code_problem():
    return jsonify({"msg": "Welcome to the code challenge"})

@app.route("/cv_matching", methods=["GET"])
@role_required("recruiter")
def recruiter_cv_matching():
    return jsonify({"msg": "Welcome to the CV matching ranking"})

@app.route('/get-junior-code', methods=['get'])
@cross_origin()
def get_junior_code():
    return jsonify(junior_get_code()), 200

@app.route('/get-middle-code', methods=['get'])
@cross_origin()
def get_middle_code():
    return jsonify(middle_get_code()), 200

@app.route('/get-senior-code', methods=['get'])
@cross_origin()
def get_senior_code():
    return jsonify(senior_get_code()), 200

if __name__ == "__main__":
    app.run()
