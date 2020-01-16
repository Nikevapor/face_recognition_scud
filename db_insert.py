from __future__ import print_function
from datetime import date, datetime, timedelta
import time
import numpy
from tempfile import TemporaryFile
import face_recognition
import mysql.connector

cnx = mysql.connector.connect(
    host='localhost',
    user="root",
    passwd="1995604H",
    db="face_recognition"
    # port=8886
)
cursor = cnx.cursor()

tomorrow = datetime.now().date() + timedelta(days=1)
now = time.strftime('%Y-%m-%d %H:%M:%S')

add_employee = ("INSERT INTO faces "
               "(`name`, `photo`, `encoding`, `created_date`, `updated_date`) "
               "VALUES (%s, %s, %s, %s, %s)")

add_salary = ("INSERT INTO detections "
              "(`date`, face_id) "
              "VALUES (%(date)s, %(face_id)s)")

employee_name = "Rafael"
employee_photo = "photos/Rafael.jpg"
rafa_image = face_recognition.load_image_file(employee_photo)
rafa_face_encoding = face_recognition.face_encodings(rafa_image)[0]

numpy.savetxt(employee_name + '.txt', rafa_face_encoding, fmt='%r')

# print(numpy.array_str(rafa_face_encoding, precision=32, suppress_small=True))
# print(numpy.fromstring(numpy.array_str(rafa_face_encoding, precision=16, suppress_small=True), dtype=float, count=128))
# print(rafa_face_encoding)
# numpy.savetxt(employee_name + '.txt', rafa_face_encoding, fmt='%.28f')
# with open('Rafael2.txt', 'w') as file:
#     encoded_str = numpy.array2string(rafa_face_encoding, precision=32,  suppress_small=True)
#     file.write(encoded_str)



b = numpy.loadtxt(employee_name + '.txt', dtype=float)
# numply load text precision for float
# numpy loadtxt with strings and floats
# https://stackoverflow.com/questions/14885660/numpy-loadtxt-rounding-off-numbers
# numpy.set_printoptions(precision = 17)
# b = numpy.loadtxt(employee_name + '.txt', dtype = numpy.float64)
print(rafa_face_encoding == b)
print(rafa_face_encoding)
print(b)





# string1 = numpy.array2string(rafa_face_encoding, precision=32,  suppress_small=True)
# print(string1)
#
# print(numpy.fromstring(string1, dtype=float, count=128))
# print(numpy.frombuffer(string1.encode('latin-1'), count=128))

# data_employee = ('Rafa', 'photos/Rafael.jpg', rafa_face_encoding, now, now)
#
# # Insert new employee
# cursor.execute(add_employee, data_employee)
# emp_no = cursor.lastrowid
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
# cnx.commit()

cursor.close()
cnx.close()