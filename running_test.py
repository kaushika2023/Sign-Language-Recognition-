import cv2
from ultralytics import YOLO

model = YOLO("D:\\pycharm_projects\\epicYolo\\runs\\detect\\train3\\weights\\best.pt")

CAP = cv2.VideoCapture(0)

while True:
    ret, frame = CAP.read()

    result = model(frame, device="cuda", verbose=False)

    if result:
        annotationFrame = result[0].plot()
    else:
        annotationFrame = frame

    resizedFrame = cv2.resize(annotationFrame, (640, 480))

    cv2.imshow("window testing", resizedFrame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

CAP.release()
cv2.destroyAllWindows()
