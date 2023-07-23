import sqlite3

with open("schema.sql", "r") as file:
    schema = file.read()
# print(schema)    
connection = sqlite3.connect('database.db')

# cursor = connection.cursor()
# 
# cursor.execute("DROP TABLE IF EXISTS users")
# cursor.execute("DROP TABLE IF EXISTS log")
# 
# cursor.execute("CREATE TABLE users (userid INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL, password TEXT NOT NULL)")
# cursor.execute("CREATE TABLE log (entryid INTEGER PRIMARY KEY AUTOINCREMENT,logged_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, userid INTEGER NOT NULL, activity TEXT NOT NULL, response_code INTEGER NOT NULL, response_message TEXT NOT NULL, result TEXT NOT NULL)")                    


cursor = connection.cursor()

cursor.executescript("""
    DROP TABLE IF EXISTS users;
    DROP TABLE IF EXISTS activity_log;
    DROP TABLE IF EXISTS log;
    CREATE TABLE users (
        userid INTEGER PRIMARY KEY,
        username TEXT NOT NULL,
        password TEXT NOT NULL);
    CREATE TABLE activity_log (entryid INTEGER PRIMARY KEY AUTOINCREMENT,
        logged_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, 
        userid INTEGER NOT NULL, 
        activity TEXT NOT NULL, 
        response_code INTEGER NOT NULL, 
        response_message TEXT NOT NULL, 
        result TEXT NOT NULL);
    """)
# cursor.executescript(schema)

user = (0,
        "Thomas",
        "abadpassword"
        )
cursor.execute(f"INSERT INTO users VALUES (?,?,?)",user)
#print (user[1])
user2 = ("Anthony",
        "abadpassword2"
        )
#print (user2[1])
cursor.execute(f"INSERT INTO users (username, password) VALUES (?,?)",user2)

cursor.execute("SELECT * FROM users")
print(cursor.fetchall())

connection.commit()
connection.close()