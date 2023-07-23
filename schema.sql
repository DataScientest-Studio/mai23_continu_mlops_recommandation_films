"""
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS activity_log;
DROP TABLE IF EXISTS log;
CREATE TABLE users (
    userid INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    password TEXT NOT NULL);
CREATE TABLE activity_log (entryid INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, 
    userid INTEGER NOT NULL, 
    activity TEXT NOT NULL, 
    response_code INTEGER NOT NULL, 
    response_message TEXT NOT NULL, 
    result TEXT NOT NULL)
"""