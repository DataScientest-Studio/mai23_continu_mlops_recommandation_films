test_db_schema = """
DROP TABLE IF EXISTS users;
CREATE TABLE users (
    userid INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    password TEXT NOT NULL);
"""