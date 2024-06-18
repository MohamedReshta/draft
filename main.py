import cv2
import os
import face_recognition
import datetime
import mediapipe as mp
import mysql.connector
from flask import Flask, render_template, request, redirect
from connection import conn


app = Flask(__name__)

# Connect to MySQL database
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="123456",
    database="face_recognition"
)
mycursor = mydb.cursor()

known_faces = []
known_names = []
today = datetime.date.today().strftime("%d_%m_%Y")

# Face recognition functions
def get_known_encodings():
    global known_faces, known_names
    known_faces = []
    known_names = []

    for filename in os.listdir('static/faces'):
        image = face_recognition.load_image_file(os.path.join('static/faces', filename))
        print(image)
        try:
            encoding = face_recognition.face_encodings(image)[0]
            known_faces.append(encoding)
            known_names.append(os.path.splitext(filename)[0])
        except IndexError as e:
            print("Can't Find Eny Faces")
        # print(known_faces, known_names)

def totalreg():
    return len(os.listdir('static/faces/'))

def extract_attendance():
    results = conn.read(f"SELECT * FROM {today}")
    return results

def mark_attendance(person):
    name = person.split('_')[0]
    roll_no = int(person.split('_')[1])
    current_time = datetime.datetime.now().strftime('%H:%M:%S')
    exists = conn.read(f"SELECT * FROM {today} WHERE roll_no = {roll_no}")
    try:
        conn.insert(f"INSERT INTO {today} VALUES (%s, %s , %s)", (name, roll_no, current_time))
    except Exception as e:
        print(e)

def identify_person():
    video_capture = cv2.VideoCapture(0)
    attendance_marked = False

    while True:
        ret, frame = video_capture.read()
        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        recognized_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            name = 'unknown'
            if True in matches:
                match_indices = [i for i, match in enumerate(matches) if match]
                for index in match_indices:
                    name = known_names[index]
                    recognized_names.append(name)

        if len(recognized_names) > 0:
            for name in recognized_names:
                mark_attendance(name)
            attendance_marked = True

        cv2.imshow('camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or attendance_marked:
            break

    video_capture.release()
    cv2.destroyAllWindows()

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/home')
def home():
    conn.create(f"CREATE TABLE IF NOT EXISTS {today} (name VARCHAR(30), roll_no INT, time VARCHAR(10))")
    userDetails = extract_attendance()
    get_known_encodings()
    return render_template('home.html', l=len(userDetails), today=today.replace("_", "-"), totalreg=totalreg(), userDetails=userDetails)

@app.route('/video_feed', methods=['GET'])
def video_feed():
    get_known_encodings()
    identify_person()
    userDetails = extract_attendance()
    return render_template('home.html', l=len(userDetails), today=today.replace("_", "-"), totalreg=totalreg(), userDetails=userDetails)

@app.route('/add_user', methods=['GET', 'POST'])
def add_user():
    name = request.form['newusername']
    roll_no = request.form['newrollno']
    userimagefolder = 'static/faces'
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        rgb_frame = frame[:, :, ::-1]
        flipped_frame = cv2.flip(frame, 1)
        text = "Press Q to Capture & Save the Image"
        font = cv2.FONT_HERSHEY_COMPLEX
        font_scale = 0.9
        font_color = (0, 0, 200)
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = (frame.shape[0] - 450)
        cv2.putText(flipped_frame, text, (text_x, text_y), font, font_scale, font_color, thickness, cv2.LINE_AA)
        cv2.imshow('Camera', flipped_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            img_name = name + '_' + str(roll_no) + '.jpg'
            cv2.imwrite(userimagefolder + '/' + img_name, flipped_frame)
            break

    video_capture.release()
    cv2.destroyAllWindows()
    userDetails = extract_attendance()
    get_known_encodings()
    return render_template('home.html', l=len(userDetails), today=today.replace("_", "-"), totalreg=totalreg(), userDetails=userDetails)

# Sign language recognition functions
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
tipIds = [4, 8, 12, 16, 20]
screenshots_folder = 'screenshots'
if not os.path.exists(screenshots_folder):
    os.makedirs(screenshots_folder)
frame_count = 0

@app.route('/sign_session')
def sign_session():
    global frame_count
    feedback = None
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        lmList = []

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

                if len(lmList) == 21:
                    fingers = []

                    if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                    for tip in range(1, 5):
                        if lmList[tipIds[tip]][2] < lmList[tipIds[tip] - 2][2]:
                            fingers.append(1)
                        else:
                            fingers.append(0)

                    totalFingers = fingers.count(1)
                    feedback = ["Bad", "Not bad", "Okay", "Good", "Perfect"][totalFingers - 1]

                    cv2.putText(img, feedback, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    mycursor.execute("INSERT INTO sign_language (feedback, name) VALUES (%s, %s)", (feedback, frame_count))
                    mydb.commit()

        # if frame_count % 10 == 0:
        #     screenshot_filename = f"{screenshots_folder}/screenshot_{frame_count}.jpg"
        #     cv2.imwrite(screenshot_filename, img)

        # frame_count += 1
        cv2.imshow('Hand Tracker', img)
        if (cv2.waitKey(1) & 0xFF == ord('q')) | (feedback is not None):
            break

    cap.release()
    cv2.destroyAllWindows()
    mycursor.execute("SELECT * FROM sign_language")
    signs = mycursor.fetchall()
    feedback_counts = {
        "Bad": 0,
        "Not bad": 0,
        "Okay": 0,
        "Good": 0,
        "Perfect": 0
    }
    total_feedback = len(signs)

    for sign in signs:
        feedback = sign[1]
        if feedback in feedback_counts:
            feedback_counts[feedback] += 1

    feedback_percentages = {k: (v / total_feedback) * 100 for k, v in feedback_counts.items()}
    return render_template('sign_language.html', signs=signs, feedback_percentages=feedback_percentages)

# @app.route('/sign-session', methods=['POST', 'GET'])
# def sign():
#     mycursor.execute("SELECT * FROM sign_language")
#     signs = mycursor.fetchall()
#     feedback_counts = {
#         "Bad": 1,
#         "Not bad": 2,
#         "Okay": 3,
#         "Good": 4,
#         "Perfect": 5
#     }
#     total_feedback = len(signs)
#
#     for sign in signs:
#         feedback = sign[1]
#         if feedback in feedback_counts:
#             feedback_counts[feedback] += 1
#
#     feedback_percentages = {k: (v / total_feedback) * 100 for k, v in feedback_counts.items()}
#
#     return render_template('sign_language.html', signs=signs, feedback_percentages=feedback_percentages)


if __name__ == '__main__':
    app.run(debug=True)


