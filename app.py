# import gevent
# from gevent import monkey
# monkey.patch_all()
from camera.utils import readb64
from camera.camera import video_camera, ip_webcam, live_cam, ThreadingCam, backgroundThread
from deep_sort.deep_sort import DeepSort
from detector.detector import Detector
from people_counter_v1 import PeopleCounterV1, is_equal
from flask import Flask, request, jsonify, send_file, make_response
import argparse
from werkzeug.utils import secure_filename
import os
from flask_socketio import SocketIO, emit, disconnect, join_room, leave_room
import base64
import cv2
import threading


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret:)'
socketio = SocketIO(app, manage_session=False, async_mode='threading', cors_allowed_origins="*")
# logger=True, engineio_logger=True, 

def create_people_counter():
    return PeopleCounterV1(threading_cam, detector, DeepSort(device='cpu', max_dist=0.5, min_confidence=0.5, nms_max_overlap=0.5, max_iou_distance=0.7, max_age=70, n_init=1, nn_budget=100), line=[(0, 120), (320, 120)])

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return {'response': 'No file part'}
    file = request.files['file']
    if file.filename == '':
        return {'response': 'No video selected for uploading'}
    else:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        cam = live_cam(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        threading_cam.set_cam(cam)
        background_thread.set_people_counter(create_people_counter())
        return {'response': 'Video successfully uploaded'}

@app.route('/ip_webcam_url', methods=['POST'])
def ip_webcam_url():
    url = request.json['url']
    cam = ip_webcam(url)
    threading_cam.set_cam(cam)
    background_thread.set_people_counter(create_people_counter())
    return 'success'

@app.route('/get_frame', methods=['GET'])
def get_frame():
    data = background_thread.get()[0]
    return jsonify(data)

@app.route('/get_frame_draw', methods=['GET'])
def get_frame_draw ():
    data = background_thread.get()[1]
    return jsonify(data)

@app.route('/get_frame_data', methods=['GET'])
def get_frame_data():
    data = background_thread.get()[2]
    return jsonify(data)

@app.route('/get_current_webcam_frame', methods=['GET'])
def get_current_webcam_frame():
    img = readb64(current_webcam_frame)
    retval, buffer = cv2.imencode('.jpeg', img)
    response = make_response(buffer.tobytes())
    response.headers['Content-Type'] = 'image/jpeg'
    return response

class custom_webcam(object):
    def __del__(self):
        cv2.destroyAllWindows() 

    def get_frame(self, size=(640, 480)):
        if current_webcam_frame is None:
            return None
        resized = cv2.resize(current_webcam_frame, size, interpolation = cv2.INTER_LINEAR) 
        frame_flip = cv2.flip(resized,1)
        # ret, jpg = cv2.imencode('.jpg', frame_flip)
        return frame_flip


@socketio.on('connect')
def connect_handle():
    # print('connect')
    pass

@socketio.on('disconnect')
def disconnect_handle():
    # print('disconnect')
    pass

@socketio.on('client_disconnecting')
def client_disconnecting_handle():
    pass

@socketio.on('start_stream')
def stream_handle():
    global streamer
    if streamer is not None:
        socketio.emit('stop_stream', to=streamer)
    with lock:
        streamer = request.sid
    cam = custom_webcam()
    threading_cam.set_cam(cam)
    background_thread.set_people_counter(create_people_counter())
    socketio.emit('stream', to=streamer)

@socketio.on('stream')
def stream_handle(json):
    global current_webcam_frame
    with lock:
        current_webcam_frame = json['image_url']

from PIL import Image
from io import BytesIO
import numpy as np

def reduce_image_size(img, new_size=None, optimize=True, quality=95):
    h, w = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    if new_size is None:
        img_pil = img_pil.resize((w,h),Image.ANTIALIAS)
    else:
        img_pil = img_pil.resize(new_size,Image.ANTIALIAS)
    output = BytesIO()
    img_pil.save(output, format="JPEG", optimize=optimize, quality=quality)
    output.seek(0)
    file_bytes = np.asarray(bytearray(output.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img

def send_image(image, is_rq=True):
    img_b64 = None
    if image is not None:
        if is_rq:
            image = reduce_image_size(image, quality=65)
        retval, buffer = cv2.imencode('.jpg', image)
        img_b64 = base64.encodebytes(buffer.tobytes()).decode("utf-8")
    socketio.emit('send_image', {'img_b64': img_b64})

@socketio.on('request_image')
def request_image_handle():
    send_image(background_thread.get()[1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='app.py')
    parser.add_argument('--host', default="0.0.0.0")
    parser.add_argument('--port', default=5000)
    parser.add_argument('--upload_folder', default="./upload_folder")
    opt = parser.parse_args()
    app.config['UPLOAD_FOLDER'] = opt.upload_folder
    threading_cam = ThreadingCam(size=(320, 240))
    detector = Detector()
    background_thread = backgroundThread(create_people_counter())
    streamer = None
    current_webcam_frame = None
    lock = threading.Lock()
    socketio.run(app, host=opt.host, port=opt.port)