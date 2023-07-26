import sqlite3

with open("schema.sql", "r") as file:
    schema = file.read()
# print(schema)    
connection = sqlite3.connect('database.db')


cursor = connection.cursor()

cursor.executescript("""
    DROP TABLE IF EXISTS users;
    DROP TABLE IF EXISTS activity_log;
    DROP TABLE IF EXISTS log;
    CREATE TABLE users (
        userid INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT NOT NULL,
        password TEXT NOT NULL);
    CREATE TABLE activity_log (entryid INTEGER PRIMARY KEY,
        logged_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, 
        userid INTEGER NOT NULL, 
        activity TEXT NOT NULL, 
        response_code INTEGER NOT NULL, 
        response_message TEXT NOT NULL, 
        result TEXT NOT NULL);
    """)
# cursor.executescript(schema)

user0 = (0,
        "Thomas",
        "thomas@e.mail",
        "abadpassword"
        )

cursor.execute(f"INSERT INTO users VALUES (?,?,?,?)",user0)
#print (user[1])
user1 = ("Anthony",
         "anthony@e.mail",
        "abadpassword1"
        )
#print (user2[1])
cursor.execute(f"INSERT INTO users (name, email, password) VALUES (?,?,?)",user1)

cursor.execute("SELECT * FROM users")
print(cursor.fetchall())

connection.commit()
connection.close()