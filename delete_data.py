import sqlite3

def delete_user():
    connection = sqlite3.connect('users.db')
    cursor = connection.cursor()
    
    cursor.execute(f"DELETE FROM users")
    
    connection.commit()
    connection.close()
    
if __name__ == '__main__':
    delete_user()
    