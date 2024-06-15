import sqlite3
import os

def create_database():
    current_dir = os.path.dirname(__file__)
    
    db_path = os.path.abspath(os.path.join(current_dir, '..', 'databases', 'code_data.db'))
    
    if not os.path.isfile(db_path):
        print(f"Database file '{db_path}' does not exist.")
        return

    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        first_name TEXT NOT NULL,
        last_name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'candidate',
        level TEXT NOT NULL DEFAULT 'none'
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_challenges (
        id INTEGER PRIMARY KEY,
        user_id INTEGER NOT NULL,
        challenge_id INTEGER NOT NULL,
        status TEXT NOT NULL,
        score INTEGER NOT NULL DEFAULT 0,
        FOREIGN KEY (user_id) REFERENCES users(id),
        FOREIGN KEY (challenge_id) REFERENCES code_data(id)
    )
    ''')

    connection.commit()
    connection.close()

if __name__ == '__main__':
    create_database()
