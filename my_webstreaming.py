# import the necessary packages
# from pyimagesearch.motion_detection import SingleMotionDetector
import json
import os
import sys
import uuid

from imutils.video import VideoStream
from flask import Response, flash
from flask import Flask
from flask import render_template
from flask import request
import threading
import argparse
# import datetime
# import imutils
import time
import cv2
import numpy as np
import face_recognition
import mysql.connector
from werkzeug.utils import redirect
from mysql.connector import errorcode


PHOTO_UPLOAD_FOLDER = 'static/data/photos/'
ENCODING_UPLOAD_FOLDER = 'static/data/encoding/'
DETECTION_UPLOAD_FOLDER = 'static/data/detection_photos/'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)

outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)
app.secret_key = "super secret key"

try:
    cnx = mysql.connector.connect(
        host='localhost',
        user="root",
        passwd="1995604H",
        db="face_recognition"
        # port=8886
    )

    cnx.autocommit = True
    cursor = cnx.cursor()
except mysql.connector.Error as err:
    if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
        print("В доступе к БД отказано")
    elif err.errno == errorcode.ER_BAD_DB_ERROR:
        print("База данных не сущесвует")
    else:
        print(err)

    sys.exit()


known_face_encodings = []
known_face_names = []

query = "SELECT `name`, photo, encoding_file FROM employees " \
        "ORDER BY `created_date` DESC "
cursor.execute(query)

for (employee_name, employee_photo, employee_encoding_file) in cursor:
    known_face_names.append(employee_name)
    np.set_printoptions(precision=16)
    enc = np.loadtxt(ENCODING_UPLOAD_FOLDER + employee_encoding_file, dtype=np.float64)
    known_face_encodings.append(enc)

# initialize the video stream and allow the camera sensor to
# warmup
# vs = VideoStream(usePiCamera=1).start()
vs = VideoStream(src="http://192.168.0.101:4747/mjpegfeed?960x720").start()
time.sleep(3.0)


@app.route('/test')
def test():
    return render_template('test.html')


@app.route('/get_len', methods=['GET', 'POST'])
def get_len():
    name = request.form['name']
    return json.dumps({'len': len(name)})


@app.route('/get_last_detections', methods=['GET'])
def get_last_detections():
    name = request.form['name']
    return json.dumps({'len': len(name)})


@app.route("/")
@app.route("/index")
def index():
    global cursor, known_face_encodings, known_face_names
    query = "SELECT name, position, employees.photo, encoding_file FROM detections " \
            "LEFT JOIN employees ON employees.id = detections.employee_id " \
            "ORDER BY detections.`date` DESC " \
            "LIMIT 5"
    cursor.execute(query)
    last_detected_employees = cursor.fetchall()

    current_detected = []
    last_four_detected = []
    if (len(last_detected_employees) > 0):
        current_detected = last_detected_employees[0]
    if (len(last_detected_employees) > 1):
        last_four_detected = last_detected_employees[1:]


    known_face_encodings = []
    known_face_names = []

    query = "SELECT `name`, photo, encoding_file FROM employees " \
            "ORDER BY `created_date` DESC "
    cursor.execute(query)

    for (employee_name, employee_photo, employee_encoding_file) in cursor:
        known_face_names.append(employee_name)
        np.set_printoptions(precision=16)
        enc = np.loadtxt(ENCODING_UPLOAD_FOLDER + employee_encoding_file, dtype=np.float64)
        known_face_encodings.append(enc)

    return render_template("index.html", current_detected=current_detected, last_four_detected=last_four_detected)


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(), mimetype = "multipart/x-mixed-replace; boundary=frame")


@app.route("/upload_employee", methods=['GET', 'POST'])
def upload_employee():
    if request.method == 'POST':
        global cursor
        name = request.form["name"]
        position = request.form["position"]

        # check if the post request has the file part
        if 'file' not in request.files:
            flash('Ошибка формы', 'danger')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('Файл не выбран', 'danger')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename, file_extension = os.path.splitext(file.filename)
            new_filename = generate_filename()
            photo_filename = new_filename + file_extension
            photo_path = os.path.join(PHOTO_UPLOAD_FOLDER, photo_filename)
            file.save(photo_path)

            image = face_recognition.load_image_file(photo_path)
            encoding = face_recognition.face_encodings(image)
            if len(encoding) > 0:
                encoding = encoding[0]
            else:
                flash("Лицо на фото не найдено", "danger")
                os.remove(photo_path)
                return redirect(request.url)

            encoding_filename = new_filename + ".txt"
            encoding_path = ENCODING_UPLOAD_FOLDER + encoding_filename
            np.savetxt(encoding_path, encoding, fmt='%r')

            try:
                now = time.strftime('%Y-%m-%d %H:%M:%S')
                add_employee = ("INSERT INTO employees "
                                "(`name`, `position`, `photo`, `encoding_file`, `created_date`, `updated_date`) "
                                "VALUES (%s, %s, %s, %s, %s, %s)")
                data_employee = (name, position, photo_filename, encoding_filename, now, now)
                cursor.execute(add_employee, data_employee)
                cnx.commit()
            except mysql.connector.Error as err:
                if err.errno == errorcode.ER_DUP_ENTRY:
                    flash("Это имя уже существует в базе данных", "danger")
                else:
                    flash(err, "error")
            else:
                flash("Сотрудник был успешно добавлен", "success")

    return render_template("upload_employee.html")


@app.route("/registered_employees")
def registered_employees():
    global cursor
    query = "SELECT name, position, photo, created_date FROM employees " \
            "ORDER BY `created_date` DESC "
    cursor.execute(query)
    employees = cursor.fetchall()

    # return the response generated along with the specific media
    # type (mime type)
    return render_template("registered_employees.html", employees=employees)


@app.route("/detected_employees")
def detected_employees():
    global cursor
    query = "SELECT detections.`date`, detections.photo, CONCAT(name, ' - ', position) as employee_info FROM detections " \
            "LEFT JOIN employees ON employees.id = detections.employee_id " \
            "ORDER BY detections.`date` DESC " \
            "LIMIT 5"
    cursor.execute(query)
    detected_employees = cursor.fetchall()
    # return the response generated along with the specific media
    # type (mime type)

    return render_template("detected_employees.html", detected_employees=detected_employees)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_filename():
    return str(uuid.uuid4())


def find_employee_id_by_name(name):
    query = ("SELECT id, name, position, photo FROM employees "
             "WHERE `name` LIKE %s")
    cursor.execute(query, (name,))
    employee = cursor.fetchone()
    if cursor.rowcount == 0:
        return None
    return employee[0]


# def detect_motion(frameCount):
#     # grab global references to the video stream, output frame, and
#     # lock variables
#     global vs, outputFrame, lock
#
#     # initialize the motion detector and the total number of frames
#     # read thus far
#     md = SingleMotionDetector(accumWeight=0.1)
#     total = 0
#     # loop over frames from the video stream
#     while True:
#         # read the next frame from the video stream, resize it,
#         # convert the frame to grayscale, and blur it
#         frame = vs.read()
#         frame = imutils.resize(frame, width=400)
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         gray = cv2.GaussianBlur(gray, (7, 7), 0)
#
#         # grab the current timestamp and draw it on the frame
#         timestamp = datetime.datetime.now()
#         cv2.putText(frame, timestamp.strftime(
#             "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
#
#         # if the total number of frames has reached a sufficient
#         # number to construct a reasonable background model, then
#         # continue to process the frame
#         if total > frameCount:
#             # detect motion in the image
#             motion = md.detect(gray)
#
#             # check to see if motion was found in the frame
#             if motion is not None:
#                 # unpack the tuple and draw the box surrounding the
#                 # "motion area" on the output frame
#                 (thresh, (minX, minY, maxX, maxY)) = motion
#                 cv2.rectangle(frame, (minX, minY), (maxX, maxY),
#                               (0, 0, 255), 2)
#
#         # update the background model and increment the total number
#         # of frames read thus far
#         md.update(gray)
#         total += 1
#
#         # acquire the lock, set the output frame, and release the
#         # lock
#         with lock:
#             outputFrame = frame.copy()


def detect_motion():
    # db_connect()
    # grab global references to the video stream, output frame, and
    # lock variables
    global vs, outputFrame, lock, cursor, known_face_encodings, known_face_names

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    possible_detections = {}

    start = time.time()
    while True:
        # Grab a single frame of video
        frame = vs.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                if face_distances.size > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                face_names.append(name)
                if name in possible_detections:
                    possible_detections[name] += 1
                else:
                    possible_detections[name] = 1

        process_this_frame = not process_this_frame

        print(possible_detections)
        end = time.time()
        if possible_detections:
            if end - start > 5:
                start = time.time()
                # possible_name = max(possible_detections, key=possible_detections.get)
                for detection in possible_detections:
                    if possible_detections[detection] > 30:
                        new_filename = generate_filename()
                        photo_filename = new_filename + ".jpg"
                        photo_path = os.path.join(DETECTION_UPLOAD_FOLDER, photo_filename)
                        cv2.imwrite(photo_path, frame)  # возможно не совсем правильно

                        now = time.strftime('%Y-%m-%d %H:%M:%S')
                        add_detection = ("INSERT INTO detections "
                                         "(`date`, `employee_id`, `photo`) "
                                         "VALUES (%s, %s, %s)")
                        print(find_employee_id_by_name)
                        data_employee = (now, find_employee_id_by_name(detection), photo_filename)
                        cursor.execute(add_detection, data_employee)
                        cnx.commit()

                # if possible_detections[possible_name] > 30:
                #     new_filename = generate_filename()
                #     photo_filename = new_filename + ".jpg"
                #     photo_path = os.path.join(DETECTION_UPLOAD_FOLDER, photo_filename)
                #     cv2.imwrite(photo_path, frame) #возможно не совсем правильно
                #
                #     now = time.strftime('%Y-%m-%d %H:%M:%S')
                #     add_detection = ("INSERT INTO detections "
                #                     "(`date`, `employee_id`, `photo`) "
                #                     "VALUES (%s, %s, %s)")
                #     print(find_employee_id_by_name)
                #     data_employee = (now, find_employee_id_by_name(possible_name), photo_filename)
                #     cursor.execute(add_detection, data_employee)
                #     cnx.commit()

                possible_detections = {}

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            # Draw a label with a name below the face
            if name == "Unknown":
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            else:
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)

            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # http://zabaykin.ru/?p=657
            # создадим белое изображение
            # или можно считать изобрежние с помощью cv2.imread("path_to_file")
            # img = np.zeros((128, 256, 3), np.uint8)
            # img[:, :, :] = 255
            # #
            # font = cv2.FONT_HERSHEY_COMPLEX
            # # # вставка текста красного цвета
            # cv2.putText(img, 'наш произвольный текст', (5, 75), font, 1, color=(0, 0, 255), thickness=2)
            # # cv2.imshow('Result', img)
            # # cv2.waitKey()
            #
            # x_offset = y_offset = 150
            # frame[y_offset:y_offset + img.shape[0], x_offset:x_offset + img.shape[1]] = img

        with lock:
            outputFrame = frame.copy()

def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


# check to see if this is the main thread of execution
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
                    help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
                    help="ephemeral port number of the server (1024 to 65535)")
    # ap.add_argument("-f", "--frame-count", type=int, default=32,
    #                 help="# of frames used to construct the background model")
    args = vars(ap.parse_args())

    # start a thread that will perform motion detection
    # t = threading.Thread(target=detect_motion, args=(
    #     args["frame_count"],))
    t = threading.Thread(target=detect_motion)
    t.daemon = True
    t.start()

    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
            threaded=True, use_reloader=False)


cursor.close()
cnx.close()

# release the video stream pointer
vs.stop()
