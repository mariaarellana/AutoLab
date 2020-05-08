######## Webcam Object Detection Using Tensorflow-trained Classifier #########

# Import packages
from utils import visualization_utils as vis_util
from utils import label_map_util
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import pymysql
from datetime import date
from datetime import datetime
import socketio
import time
# -----------------------------------------------------------------------------------------------------------------------
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import imutils
# -----------------------------------------------------------------------------------------------------------------------

# Creating a Client Instance
sio = socketio.Client()

# Connecting to a Server
sio.connect("http://localhost:3000/")
@sio.event
def connect(parameter_list):
    print("Se estableció la conexión con el servidor")


print("mi ID de sesion es ", sio.sid)

# Connect this code to Mysql Database
connection = pymysql.connect(
    host='database-1.coxt8euwrxba.us-east-1.rds.amazonaws.com',
    port=int(3306),
    user='admin',
    password='carpediem0599',
    db='database-1')

# ------------------------------------------------------------------------------------------------

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)
# ------------------------------------------------------------------------------------------------

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'entrenamiento1.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 6

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v1.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.compat.v1.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier
# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize webcam feed
video = VideoStream(src=0).start()
time.sleep(2.0)


def detect_motion(frameCount):
    # grab global references to the video stream, output frame, and
    # lock variables
    global outputFrame, lock, sio, connection, num_detections, detection_classes, detection_scores, detection_boxes, image_tensor, sess, video, category_index

    total = 0

    # loop over frames from the video stream
    while True:
        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        frame = video.read()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_expanded = np.expand_dims(frame_rgb, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores,
             detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=4,
            min_score_thresh=0.60)

        if total > frameCount:

            # All the results have been drawn on the frame, so it's time to save in database.
            chooseClass = np.squeeze(classes).astype(np.int32)
            chooseScore = np.squeeze(scores)
            limit = len(chooseClass)
            nowTime = int(time.time())
            formatTime = str(time.ctime())
            for i in range(0, limit):
                if chooseScore[i] > 0.60 and chooseClass[i] < 7:
                    classesDb = int(chooseClass[i])
                    scoresDb = round(float(chooseScore[i]), 2)
                    sql = "INSERT INTO `database-1`.database1 (`id`, `classes`, `scores`, `datetime`, `formatTime`) VALUES (%s, %s, %s, %s, %s)"
                    values = (None, classesDb, scoresDb, nowTime, formatTime)
                    cur = connection.cursor()
                    cur.execute(sql, values)
                    connection.commit()
                    print(chooseClass[i])
                    print(chooseScore[i])
                    print(nowTime)
                    print(formatTime)
                    # Emitting Events Socket.io
                    sio.emit("ClassData", classesDb)
                    sio.emit("ScoreData", scoresDb)
                    sio.emit("formattime", formatTime)
        total += 1
        # acquire the lock, set the output frame, and release the
        # lock
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
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
              bytearray(encodedImage) + b'\r\n')


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# check to see if this is the main thread of execution
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
                    help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
                    help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=32,
                    help="# of frames used to construct the background model")
    args = vars(ap.parse_args())

    # start a thread that will perform motion detection
    t = threading.Thread(target=detect_motion, args=(
        args["frame_count"],))
    t.daemon = True
    t.start()

    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
            threaded=True, use_reloader=False)

# release the video stream pointer
# Clean up
video.stop()
sio.disconnect()
connection.close()
