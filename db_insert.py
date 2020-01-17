from __future__ import print_function
from datetime import date, datetime, timedelta
import time
import numpy
import face_recognition
import mysql.connector

cnx = mysql.connector.connect(
    host='localhost',
    user="root",
    passwd="1995604",
    db="face_recognition"
    # port=8886
)

cursor = cnx.cursor()

tomorrow = datetime.now().date() + timedelta(days=1)
now = time.strftime('%Y-%m-%d %H:%M:%S')

add_employee = ("INSERT INTO faces "
               "(`name`, `photo`, `encoding`, `created_date`, `updated_date`) "
               "VALUES (%s, %s, %s, %s, %s)")

# add_salary = ("INSERT INTO detections "
#               "(`date`, face_id) "
#               "VALUES (%(date)s, %(face_id)s)")
#
# employee_name = "Rafael"
# employee_photo = "photos/Rafael.jpg"
# employee_encoding_file = "encoding/Rafael.txt"
# rafa_image = face_recognition.load_image_file(employee_photo)
# rafa_face_encoding = face_recognition.face_encodings(rafa_image)[0]
#
# numpy.savetxt(employee_encoding_file, rafa_face_encoding, fmt='%r')

# print(numpy.array_str(rafa_face_encoding, precision=32, suppress_small=True))
# print(numpy.fromstring(numpy.array_str(rafa_face_encoding, precision=16, suppress_small=True), dtype=float, count=128))
# print(rafa_face_encoding)
# numpy.savetxt(employee_name + '.txt', rafa_face_encoding, fmt='%.28f')
# with open('Rafael2.txt', 'w') as file:
#     encoded_str = numpy.array2string(rafa_face_encoding, precision=32,  suppress_small=True)
#     file.write(encoded_str)



# b = numpy.loadtxt(employee_name + '.txt', dtype=numpy.float)
# numply load text precision for float
# numpy loadtxt with strings and floats
# https://stackoverflow.com/questions/14885660/numpy-loadtxt-rounding-off-numbers
# numpy.set_printoptions(precision = 16)
# b = numpy.loadtxt(employee_name + '.txt', dtype = numpy.float64)
# print(rafa_face_encoding == b)
# print(rafa_face_encoding)
# print(b)





# string1 = numpy.array2string(rafa_face_encoding, precision=32,  suppress_small=True)
# print(string1)
#
# print(numpy.fromstring(string1, dtype=float, count=128))
# print(numpy.frombuffer(string1.encode('latin-1'), count=128))

# data_employee = (employee_name, employee_photo, employee_encoding_file, now, now)



# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("photos/obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a second sample picture and learn how to recognize it.
biden_image = face_recognition.load_image_file("photos/biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

rafa_image = face_recognition.load_image_file("photos/Rafael.jpg")
rafa_face_encoding = face_recognition.face_encodings(rafa_image)[0]

known_face_photos = [
    "photos/obama.jpg",
    "photos/biden.jpg",
    "photos/Rafael.jpg"
]

# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding,
    rafa_face_encoding
]
known_face_names = [
    "Barack Obama",
    "Joe Biden",
    "Rafael"
]

i = 0
for known_face_name in known_face_names:
    employee_name = known_face_name
    employee_photo = known_face_photos[i]
    employee_encoding_file = "encoding/" + known_face_name + ".txt"

    data_employee = (employee_name, employee_photo, employee_encoding_file, now, now)
    cursor.execute(add_employee, data_employee)

    numpy.savetxt(employee_encoding_file, known_face_encodings[i], fmt='%r')

    i += 1


# Insert new employee
# cursor.execute(add_employee, data_employee)
# face_id = cursor.lastrowid
#
# # Insert salary information
# data_salary = {
#   'emp_no': emp_no,
#   'salary': 50000,
#   'from_date': tomorrow,
#   'to_date': date(9999, 1, 1),
# }
# cursor.execute(add_salary, data_salary)
#
# # Make sure data is committed to the database
cnx.commit()

cursor.close()
cnx.close()