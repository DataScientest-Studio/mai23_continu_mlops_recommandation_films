import sqlite3
import datetime

with open("../schema.sql", "r") as file:
    schema = file.read()
# print(schema)    
connection = sqlite3.connect('../database.db')


cursor = connection.cursor()

cursor.executescript("""
    DROP TABLE IF EXISTS users;
    CREATE TABLE users (
        userid INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT NOT NULL,
        password TEXT NOT NULL);
    """)
# cursor.executescript(schema)

user0 = (0,
        "Thomas",
        "thomas@e.mail",
        "abadpassword"
        )

cursor.execute(f"INSERT INTO users VALUES (?,?,?,?)", user0)
#print (user[1])

user1 = (1,
         "Anthony",
         "anthony@e.mail",
        "abadpassword1"
        )
#print (user2[1])

cursor.execute(f"INSERT INTO users (userid, name, email, password) VALUES (?,?,?,?)",user1)

cursor.execute("SELECT * FROM users")
print(cursor.fetchall())

cursor.executescript("""
    DROP TABLE IF EXISTS ratings;
    CREATE TABLE ratings (
        ratingid INTEGER PRIMARY KEY,
        userid INTEGER NOT NULL,
        movieid INTEGER NOT NULL,
        rating INTEGER NOT NULL)
    """)

rating0 = (0,
        0,
        420293,
        4
        )

cursor.execute(f"INSERT INTO ratings VALUES (?,?,?,?)", rating0)

rating1 = (1,
        1,
        884732,
        5
        )
cursor.execute(f"INSERT INTO ratings VALUES (?,?,?,?)", rating1)

cursor.execute("SELECT * FROM ratings")
print(cursor.fetchall())

cursor.executescript("""
    DROP TABLE IF EXISTS activity_log;
    CREATE TABLE activity_log (
        entryid INTEGER PRIMARY KEY,
        logged_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, 
        userid INTEGER NOT NULL, 
        activity TEXT NOT NULL, 
        response_code INTEGER NOT NULL, 
        response_message TEXT NOT NULL, 
        result TEXT NOT NULL);
    """)

connection.commit()
connection.close()