import sqlite3
import random

def senior_get_code():
    conn = sqlite3.connect('./databases/srec.db')
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM code_data WHERE difficulty IN (2, 10, 11, 12)')
    medium_tests = cursor.fetchall()

    cursor.execute('SELECT * FROM code_data WHERE difficulty IN (3, 6, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23)')
    hard_tests = cursor.fetchall()

    conn.close()

    random_medium_index = random.randint(0, len(medium_tests) - 1)
    random_hard_index = random.sample(range(0, len(hard_tests)), 2)

    test_1 = medium_tests[random_medium_index]
    test_2 = hard_tests[random_hard_index[0]]
    test_3 = hard_tests[random_hard_index[1]]

    result = {
        'test_1': {
            'ID': test_1[0],
            'source': test_1[1],
            'name': test_1[2],
            'description': test_1[3],
            'public_input': test_1[4],
            'public_output': test_1[5],
            'gen_input': test_1[6],
            'gen_output': test_1[7],
            'difficulty': test_1[8],
            'second': test_1[9],
            'nano': test_1[10],
            'memory_limit_bytes': test_1[11],
        },
        'test_2': {
            'ID': test_2[0],
            'source': test_2[1],
            'name': test_2[2],
            'description': test_2[3],
            'public_input': test_2[4],
            'public_output': test_2[5],
            'gen_input': test_2[6],
            'gen_output': test_2[7],
            'difficulty': test_2[8],
            'second': test_2[9],
            'nano': test_2[10],
            'memory_limit_bytes': test_2[11],
        },
        'test_3': {
            'ID': test_3[0],
            'source': test_3[1],
            'name': test_3[2],
            'description': test_3[3],
            'public_input': test_3[4],
            'public_output': test_3[5],
            'gen_input': test_3[6],
            'gen_output': test_3[7],
            'difficulty': test_3[8],
            'second': test_3[9],
            'nano': test_3[10],
            'memory_limit_bytes': test_3[11],
        },
    }    
    
    return result
