import datetime

import face_recognition
import mysql.connector
import numpy


cnx = mysql.connector.connect(
    host='localhost',
    user="root",
    passwd="1995604",
    db="face_recognition"
    # port=8886
)
cursor = cnx.cursor()

query = "SELECT name, photo, encoding FROM faces"

cursor.execute(query)

known_face_encodings = []
known_face_names = []
for (employee_name, employee_photo, employee_encoding_file) in cursor:
    known_face_encodings.append(employee_encoding_file)
    known_face_names.append(employee_name)

    numpy.set_printoptions(precision = 16)
    b = numpy.loadtxt(employee_encoding_file, dtype = numpy.float64)
    known_face_encodings.append(b)


print(known_face_encodings)

cursor.close()
cnx.close()

