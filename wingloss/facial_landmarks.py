import os
import csv
import dlib
import cv2
import numpy as np

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.width()
    h = rect.height()
    return (x, y, w, h)

def detect_landmarks(image, detector, predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    landmarks = []
    for rect in rects:
        shape = predictor(gray, rect)
        for i in range(68):
            landmarks.append((shape.part(i).x, shape.part(i).y))
    return landmarks

def add_additional_points(landmarks, img_width, img_height):
    # Add 12 additional points based on image size
    points = [(0, 0), (0, img_height//3), (0, 2*img_height//3), (0, img_height-1),
              (img_width//3, 0), (2*img_width//3, 0), (img_width-1, 0),
              (img_width-1, img_height//3), (img_width-1, 2*img_height//3),
              (img_width-1, img_height-1), (img_width//3, img_height-1),
              (2*img_width//3, img_height-1)]
    landmarks.extend(points)
    return landmarks

def save_landmarks(landmarks, file_path):
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(landmarks)

def process_images(src_folder, dst_folder, predictor):
    detector = dlib.get_frontal_face_detector()
    for root, dirs, files in os.walk(src_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                img_name, _ = os.path.splitext(file)
                output_folder = os.path.join(dst_folder, os.path.relpath(root, src_folder))
                os.makedirs(output_folder, exist_ok=True)
                output_file = os.path.join(output_folder, img_name + '.csv')

                image = cv2.imread(img_path)
                img_height, img_width = image.shape[:2]

                landmarks = detect_landmarks(image, detector, predictor)
                landmarks = add_additional_points(landmarks, img_width, img_height)
                save_landmarks(landmarks, output_file)

if __name__ == "__main__":
    predictor_file = '../shape_predictor_68_face_landmarks.dat'
    src_path = '/home/na/1_Face_morphing/2_data/FRGC-Morphs/frgc/raw_aligned_1024_pairs/'
    dst_path = '/home/na/1_Face_morphing/2_data/FRGC-Morphs/frgc/raw_aligned_1024_pairs_landmarks/'

    predictor = dlib.shape_predictor(predictor_file)
    process_images(src_path, dst_path, predictor)
