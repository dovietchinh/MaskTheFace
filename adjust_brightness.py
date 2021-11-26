import cv2 
import numpy as np
from utils.aux_functions import *
import dlib
from easydict import EasyDict
def get_avg_brightness(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    return np.mean(v)
def change_brightness(img, value=1.0):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    v = value * v
    v[v > 255] = 255
    v = np.asarray(v, dtype=np.uint8)
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def main():
    img2= cv2.imread('chinh.jpg', cv2.IMREAD_COLOR)
    img = img2.copy()
    detector=dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('dlib_models/shape_predictor_68_face_landmarks.dat')
    args = EasyDict()
    args.code = None
    args.mask_type = 'cloth'
    args.predictor = predictor
    args.detector = detector
    args.verbose = False
    args.pattern = False
    args.color = False
    face_locations = args.detector(img, 1)
    face_location = face_locations[0]
    shape = args.predictor(img, face_location)
    shape = face_utils.shape_to_np(shape)
    face_landmarks = shape_to_landmarks(shape)
    face_location = rect_to_bb(face_location)
    six_points,angle = get_six_points(face_landmarks, img)
    face_landmark = face_landmarks
    left_eye = face_landmark["left_eye"]   # return a center point of left eye
    right_eye = face_landmark["right_eye"]
    left_eye_mid = np.mean(np.array(left_eye), axis=0)
    right_eye_mid = np.mean(np.array(right_eye), axis=0)
    eye_line_mid = (left_eye_mid + right_eye_mid) / 2
    cfg = read_cfg(config_filename="masks/masks.cfg", mask_type='cloth', verbose=False)
    mask_line = np.float32([cfg.mask_a, cfg.mask_b, cfg.mask_c, cfg.mask_f, cfg.mask_e, cfg.mask_d])   
    
    img3 = change_brightness(img,value=1.5)
    img4 = change_brightness(img,value=0.5)
    out_img,mask = mask_face(img3, face_location, six_points, angle, args, type="surgical_green") 
    out_img2,mask = mask_face(img2, face_location, six_points, angle, args, type="surgical_green") 
    out_img4,mask = mask_face(img4, face_location, six_points, angle, args, type="surgical_green") 
    #cv2.namedWindow('a', cv2.WINDOW_NORMAL)
    #cv2.namedWindow('b', cv2.WINDOW_NORMAL)
    #cv2.namedWindow('c', cv2.WINDOW_NORMAL)
    #cv2.imshow('a',out_img2)
    #cv2.imshow('b',out_img4)
    #cv2.imshow('c',out_img)
    cv2.imwrite('a.jpg',out_img2)
    cv2.imwrite('b.jpg',out_img4)
    cv2.imwrite('c.jpg',out_img)

    #k = cv2.waitKey(0)
def main2():
    #cv2.namedWindow('a', cv2.WINDOW_NORMAL)
    #cv2.namedWindow('b', cv2.WINDOW_NORMAL)
    mask_path = 'chinh.jpg'
    img = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    brightness = get_avg_brightness(img)
    print(brightness)
    img2 = change_brightness(img,value=0.5)
    brightness = get_avg_brightness(img)
    print(brightness)
    cv2.imwrite('chinh1.jpg',img)
    cv2.imwrite('chinh2.jpg',img2)
    #cv2.imshow('a',img)
    #cv2.imshow('b',img2)
    #k = cv2.waitKey(0)


if __name__ =='__main__':
    main()