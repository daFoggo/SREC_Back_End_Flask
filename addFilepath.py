import os
import sqlite3

video_folder = './static/video_interview'


conn = sqlite3.connect('./databases/srec.db') 
cursor = conn.cursor()


video_files = os.listdir(video_folder)

for video_file in video_files:
    question_id = os.path.splitext(video_file)[0]  
    video_path = os.path.join(video_folder, video_file)
    
    cursor.execute('''
        UPDATE question_data 
        SET video_pathname = ? 
        WHERE question_id = ?
    ''', (video_path, question_id))

conn.commit()
conn.close()

print("Cập nhật hoàn tất.")