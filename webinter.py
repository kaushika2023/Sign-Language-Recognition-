from flask import Flask, render_template, Response, request, url_for
import cv2
from torch.distributed.elastic.multiprocessing.redirects import redirect
from ultralytics import YOLO
import numpy

global capture,switch
capture=0
switch=1

model = YOLO("best.pt")

app = Flask(__name__, template_folder='./templates')

camera = cv2.VideoCapture(0)


def gen_frames():
    global out, capture, rec_frame
    while True:
        ret, frame = camera.read()


        result = model(frame, device="cuda", verbose=False)

        if result:
            annot = result[0].plot()
        else:
            annot = frame

        resize = cv2.resize(annot, (640, 480))

        try:
            r,buffer = cv2.imencode(".jpg", resize)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except Exception as e:
            pass



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global switch, camera
    if request.method == 'POST':
        if request.form.get('stop') == 'STOP':
            if (switch == 1):
                switch = 0
                camera.release()
                cv2.destroyAllWindows()

            else:
                camera = cv2.VideoCapture(0)
                switch = 1

        if request.form.get("aboutus") == "getaboutus":
            return render_template("abs.html")


    elif request.method == 'GET':
        return render_template('index.html')
    return render_template('index.html')


if __name__ == '__main__':
    app.run()