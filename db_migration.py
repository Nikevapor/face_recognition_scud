from __future__ import print_function
import mysql.connector
from mysql.connector import errorcode

DB_NAME = 'face_recognition'

TABLES = {}
TABLES['faces'] = (
    "CREATE TABLE `faces` ("
    "  `id` int(11) NOT NULL AUTO_INCREMENT,"
    "  `name` varchar(128) NOT NULL,"
    "  `photo` varchar(128) NOT NULL,"
    "  `encoding` blob NOT NULL,"
    "  `created_date` datetime NOT NULL,"
    "  `updated_date` datetime NOT NULL,"
    "  PRIMARY KEY (`id`), UNIQUE KEY `faces_name` (`name`)"
    ") ENGINE=InnoDB")

TABLES['detections'] = (
    "CREATE TABLE `detections` ("
    "  `id` int(11) NOT NULL AUTO_INCREMENT,"
    "  `date` datetime NOT NULL,"
    "  `face_id` int(11) NOT NULL,"
    "  PRIMARY KEY (`id`),"
    "  CONSTRAINT `fk_detection_faces` FOREIGN KEY (`face_id`) "
    "     REFERENCES `faces` (`id`) ON DELETE CASCADE"
    ") ENGINE=InnoDB")

cnx = mysql.connector.connect(
    host='localhost',
    user="root",
    passwd="1995604H",
    db="face_recognition"
    # port=8886
)
cursor = cnx.cursor()

def create_database(cursor):
    try:
        cursor.execute(
            "CREATE DATABASE {} DEFAULT CHARACTER SET 'utf8'".format(DB_NAME))
    except mysql.connector.Error as err:
        print("Failed creating database: {}".format(err))
        exit(1)


try:
    cursor.execute("USE {}".format(DB_NAME))
except mysql.connector.Error as err:
    print("Database {} does not exists.".format(DB_NAME))
    if err.errno == errorcode.ER_BAD_DB_ERROR:
        create_database(cursor)
        print("Database {} created successfully.".format(DB_NAME))
        cnx.database = DB_NAME
    else:
        print(err)
        exit(1)


for table_name in TABLES:
    table_description = TABLES[table_name]
    try:
        print("Creating table {}: ".format(table_name), end='')
        cursor.execute(table_description)
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
            print("already exists.")
        else:
            print(err.msg)
    else:
        print("OK")

cursor.close()
cnx.close()

#
# try:
#     cnx = mysql.connector.connect(
#         host='localhost',
#         user="root",
#         passwd="1995604H",
#         # port=8886
#     )
#
#     mycursor = cnx.cursor()
#     mycursor.execute("CREATE DATABASE IF NOT EXISTS face_recognition")
#
# except mysql.connector.Error as err:
#     if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
#         print("Something is wrong with your user name or password")
#     elif err.errno == errorcode.ER_BAD_DB_ERROR:
#         print("Database does not exist")
#     else:
#         print(err)
# else:
#     print(cnx)
#
#     cnx.close()