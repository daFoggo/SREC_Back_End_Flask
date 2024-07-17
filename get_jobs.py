import os
import sqlite3
from extract_job import extract_job

def get_jobs():
    conn = sqlite3.connect('./databases/srec.db')
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM jobs')
    jobs = cursor.fetchall()
    
    result = []
    for job in jobs:
        id = job[0]
        title = job[1]
        level = job[2]
        description = job[4]
        data = job[5]
        categories = job[6]
        job_path = job[7]
        data_string = job[8]

        number_of_candidates = len([entry for entry in os.listdir(job_path)])
        cursor.execute('UPDATE jobs SET number_of_candidates = ? WHERE id = ?', (number_of_candidates, id))
        conn.commit()

        if data is None or categories is None or data_string is None:
            job_data_string, job_data, categories = extract_job(description)
            cursor.execute('UPDATE jobs SET (data_string, data, categories) = (?, ?, ?) WHERE id = ?', (job_data_string, str(job_data), str(categories), id))
            conn.commit()
        result.append({'id': id,
                       'title': title,
                       'level': level,
                       'number_of_candidates': number_of_candidates,
                       'description': description,
                       'data': str(data), 
                       'categories': str(categories),
                       'job_path': job_path,
                       'data_string': data_string})
    conn.close()
    return result

    
