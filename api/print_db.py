import sqlite3

with open("../schema.sql", "r") as file:
    schema = file.read()
# print(schema)    
connection = sqlite3.connect('../database.db')


cursor = connection.cursor()

cursor.execute("SELECT * FROM ratings")
print(cursor.fetchall())

connection.commit()
connection.close()