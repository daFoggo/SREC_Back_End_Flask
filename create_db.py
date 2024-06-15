import sqlite3

def create_database():
    connection = sqlite3.connect('users.db')
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

    connection.commit()
    connection.close()

if __name__ == '__main__':
    create_database()
