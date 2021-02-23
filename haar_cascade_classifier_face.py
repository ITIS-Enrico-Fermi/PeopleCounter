import cv2 as cv
import argparse
import logging
import os


def draw_circle(frame, coord):
    (x, y, w, h) = coord
    center = (x + w // 2, y + h // 2)
    return cv.ellipse(frame, center, (w // 2, h // 2), 0, 0, 180, (0, 255, 0), 4)  # 360, color


def detect_and_display(frame, face_cascade: cv.CascadeClassifier) -> None:
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x, y, w, h) in faces:
        frame = draw_circle(frame, (x, y, w, h))
        # ROI
    cv.imshow('Face detection with HCC', frame)  # HCCD - Haar Cascade Classifier


def main(camera: int, face_cascade_name: str) -> None:
    face_cascade = cv.CascadeClassifier()
    face_cascade.load(cv.samples.findFile(face_cascade_name))
    cap = cv.VideoCapture(camera)
    if not cap.isOpened():
        logging.error("Camera video stream can't be opened")
        exit(1)
    while True:
        ret, frame = cap.read()
        if frame is None:
            print('No caputered frame')
            continue

        detect_and_display(frame, face_cascade)

        if cv.waitKey(10) == 27:
            break


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser()
    parser.add_argument('--face_cascade', help='Path to face cascade.',
                        default=os.path.join(os.path.split(os.path.abspath(cv.__file__))[0],
                                             'data/haarcascade_frontalface_alt.xml'))
    parser.add_argument('--camera', help='Camera number.', type=int, default=0)
    args = parser.parse_args()
    fc = args.face_cascade
    c = args.camera
    main(c, fc)
