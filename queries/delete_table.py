import os
import sqlite3

def delete_tables():
    current_dir = os.path.dirname(__file__)
    
    db_path = os.path.abspath(os.path.join(current_dir, '..', 'databases', 'code_data.db'))
    
    if not os.path.isfile(db_path):
        print(f"Database file '{db_path}' does not exist.")
        return

    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    try:
        cursor.execute('DROP TABLE IF EXISTS user_challenges')
        cursor.execute('DROP TABLE IF EXISTS users')
        connection.commit()
        print("Tables 'users' and 'user_challenges' deleted successfully.")
    except sqlite3.Error as e:
        print(f'An error occurred: {e}')
    finally:
        connection.close()

if __name__ == '__main__':
    delete_tables()
