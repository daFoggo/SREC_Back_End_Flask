import sqlite3
import random


def middle_get_code():
    conn = sqlite3.connect("./databases/srec.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM code_data WHERE difficulty = 1 OR difficulty = 7 OR difficulty = 8"
    )
    easy_tests = cursor.fetchall()

    cursor.execute(
        "SELECT * FROM code_data WHERE difficulty = 2 OR difficulty = 10 OR difficulty = 11 OR difficulty = 12"
    )
    medium_tests = cursor.fetchall()

    cursor.execute(
        "SELECT * FROM code_data WHERE difficulty = 3 OR difficulty = 6 OR difficulty = 13 OR difficulty = 14 OR difficulty = 15 OR difficulty = 16 OR difficulty = 17 OR difficulty = 19 OR difficulty = 20 OR difficulty = 21 OR difficulty = 22 OR difficulty = 23"
    )
    hard_tests = cursor.fetchall()
    conn.close()

    random_easy_index = random.sample(range(0, len(easy_tests)), 1)
    random_medium_index = random.sample(range(0, len(medium_tests)), 1)
    random_hard_index = random.sample(range(0, len(hard_tests)), 1)

    test_1 = easy_tests[random_easy_index[0]]
    test_2 = medium_tests[random_medium_index[0]]
    test_3 = hard_tests[random_hard_index[0]]
    result = {
        "test_1": {
            "ID": test_1[0],
            "source": test_1[1],
            "name": test_1[2],
            "description": test_1[3],
            "public_input": test_1[4],
            "public_output": test_1[5],
            "gen_input": test_1[6],
            "gen_output": test_1[7],
            "difficulty": test_1[8],
            "second": test_1[9],
            "nano": test_1[10],
            "memory_limit_bytes": test_1[11],
        },
        "test_2": {
            "ID": test_2[0],
            "source": test_2[1],
            "name": test_2[2],
            "description": test_2[3],
            "public_input": test_2[4],
            "public_output": test_2[5],
            "gen_input": test_2[6],
            "gen_output": test_2[7],
            "difficulty": test_2[8],
            "second": test_2[9],
            "nano": test_2[10],
            "memory_limit_bytes": test_2[11],
        },
        "test_3": {
            "ID": test_3[0],
            "source": test_3[1],
            "name": test_3[2],
            "description": test_3[3],
            "public_input": test_3[4],
            "public_output": test_3[5],
            "gen_input": test_3[6],
            "gen_output": test_3[7],
            "difficulty": test_3[8],
            "second": test_3[9],
            "nano": test_3[10],
            "memory_limit_bytes": test_3[11],
        },
    }
    return result
