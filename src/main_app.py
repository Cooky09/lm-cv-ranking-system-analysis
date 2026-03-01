from flask import Flask, request, jsonify, send_file
from celery import Celery
from redis import Redis
import os
import requests
import logging
import pandas as pd
from app_cv import process_cv_file, calculate_context_score

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = './uploads'
app.config['broker_url'] = 'redis://localhost:6379/0'
app.config['result_backend'] = 'redis://localhost:6379/0'
app.config['CSV_FOLDER'] = './csv_exports'

celery = Celery("main_app", broker=app.config['broker_url'])
celery.conf.update(app.config)

redis_client = Redis(host='localhost', port=6379, db=0)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['CSV_FOLDER'], exist_ok=True)

logging.basicConfig(level=logging.INFO)

@app.route('/api/upload-cvs', methods=['POST'])
def upload_cv():
    cv_path = "C:/Users/Mahua Mukhopadhyay/Desktop/Resumes"
    files = []
    file_names = []

    for filename in os.listdir(cv_path):
        file_path = os.path.join(cv_path, filename)
        if os.path.isfile(file_path):
            files.append(('file', open(file_path, "rb")))
            file_names.append(filename)

    url = "http://127.0.0.1:5000/api/upload-endpoint"  
    response = requests.post(url, files=files)

    for _, file in files:
        file.close()

    logging.info(f"Status code: {response.status_code}")
    logging.info("Response text: Files uploaded successfully")

    return jsonify({
        "status_code": response.status_code,
        "response_text": "Files uploaded successfully",
        "file_names": file_names
    })

@celery.task(bind=True)
def analyze_cvs_task(self, file_paths, sample_jd):
    logging.info("Starting CV analysis task")
    results = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            logging.warning(f"File not found: {file_path}")
            continue
        try:
            logging.info(f"Processing file: {file_path}")
            cv_file, text, name, designation, experience, education, skills = process_cv_file(file_path)
            context_score = calculate_context_score(text, sample_jd)
            context_score_percentage = f"{context_score * 100:.2f}%"  # Convert to percentage
            results.append({
                "Resume Name": cv_file,
                "Name": name,
                "Context Score": context_score_percentage,  # Ensure correct format
            })
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")
            self.update_state(state='FAILURE', meta=str(e))
            return {'error': str(e)}
    
    return results

@app.route('/api/analyze-cvs', methods=['POST'])
def analyze_cvs():
    logging.info("Received request for analyzing CVs")
    data = request.json
    if not data or 'file_paths' not in data or 'sample_jd' not in data:
        return jsonify({"error": "Invalid input"}), 400
    
    file_paths = data['file_paths']
    sample_jd = data['sample_jd']
    
    task = analyze_cvs_task.apply_async(args=[file_paths, sample_jd])
    return jsonify({"task_id": task.id}), 202

@app.route('/api/analyze-cvs/<task_id>', methods=['GET'])
def get_analysis_result(task_id):
    logging.info(f"Checking status of task {task_id}")
    task = analyze_cvs_task.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'result': task.result
        }
    else:
        response = {
            'state': task.state,
            'status': str(task.info)
        }

    return jsonify(response)

@app.route('/api/export-csv/<task_id>', methods=['GET'])
def export_csv(task_id):
    logging.info(f"Exporting CSV for task {task_id}")
    task = analyze_cvs_task.AsyncResult(task_id)
    if task.state != 'SUCCESS':
        return jsonify({"error": "Task not completed or failed"}), 400
    
    results = task.result
    if not results:
        return jsonify({"error": "No results to export"}), 400

    df = pd.DataFrame(results)
    csv_path = os.path.join(app.config['CSV_FOLDER'], f"{task_id}.csv")
    df.to_csv(csv_path, index=False)

    return send_file(csv_path, as_attachment=True, download_name=f"{task_id}.csv")

if __name__ == '__main__':
    app.run(debug=True)