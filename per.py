from flask import Flask, jsonify,request
import json
import random
import joblib
import numpy as np

app = Flask(__name__)

with open('./static/survey/data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
categories = {
    'EXT': {f'EXT{i}': [] for i in range(1, 11)},
    'EST': {f'EST{i}': [] for i in range(1, 11)},
    'OPN': {f'OPN{i}': [] for i in range(1, 11)},
    'AGR': {f'AGR{i}': [] for i in range(1, 11)},
    'CSN': {f'CSN{i}': [] for i in range(1, 11)}
}
for item in data:
    category = item['Type'][:3]
    subcategory = item['Type']
    if category in categories and subcategory in categories[category]:
        categories[category][subcategory].append(item)
@app.route('/random_questions', methods=['GET'])
def get_random_questions():
    selected_questions = []
    for category, subcategories in categories.items():
        for subcategory, questions in subcategories.items():
            if questions:
                selected_questions.extend(random.sample(questions, 1)) 
    if len(selected_questions) != 50:
        return jsonify({"error": f"Expected 50 questions, but got {len(selected_questions)}"}), 400
    final_questions = []
    main_categories = {'EXT': [], 'EST': [], 'OPN': [], 'AGR': [], 'CSN': []}
    for question in selected_questions:
        category = question['Type'][:3]
        if category in main_categories:
            main_categories[category].append(question)
    for category, questions in main_categories.items():
        if len(questions) >= 4:
            final_questions.extend(random.sample(questions, 4))
    if len(final_questions) != 20:
        return jsonify({"error": f"Expected 20 questions, but got {len(final_questions)}"}), 400
    for idx, question in enumerate(final_questions, start=1):
        question['ID'] = f"Q{idx}"
    return jsonify(final_questions)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON format or empty data'}), 400
        features = []
        for item in data:
            try:
                features.append([item[f'Q{i+1}'] for i in range(20)])
            except KeyError as e:
                return jsonify({'error': f'Missing key in JSON data: {str(e)}'}), 400
        features = np.array(features)
        kmeans = joblib.load('./static/survey/kmeans_model.joblib')
        label_mapping = {
            0: 'Extraverted',
            1: 'Extraverted',
            2: 'Extraverted',
            3: 'Responsible',
            4: 'Extraverted',
            5: 'Lively',
            6: 'Extraverted',
            7: 'Serious',
            8: 'Extraverted',
            9: 'Dependable',
            10: 'Serious',
            11: 'Serious',
            12: 'Extraverted'
            }
        labels = kmeans.predict(features)
        mapped_labels = [label_mapping.get(int(label), 'Unknown') for label in labels]
        result = []
        for item, label in zip(data, mapped_labels):
            item_with_label = item.copy()  
            item_with_label['label'] = label  
            result.append(item_with_label)  
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500