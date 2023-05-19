

import cv2 
import dlib
import numpy as np
import time

# initialize face detector and landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("D:/Downloads/blink_to_comms-master/shape_predictor_68_face_landmarks.dat")

# define the blinking threshold
EAR_THRESHOLD = 0.25
# define the duration threshold for a single blink
BLINK_DURATION_THRESHOLD = 0.2 # in seconds

# initialize variables for blink detection
prev_time = 0
blink_start_time = 0
blinking = False

# define the Morse code dictionary
morse_dict = {'.-': 'A', '1000': 'B', '1010': 'C', '100': 'D', '0': 'E', '0010': 'F', '110': 'G', '0000': 'H', '00': 'I', '0111': 'J', '101': 'K', '0100': 'L', '11': 'M', '10': 'N', '111': 'O', '0110': 'P', '1101': 'Q', '010': 'R', '000': 'S', '1': 'T', '001': 'U', '0001': 'V', '011': 'W', '1001': 'X', '1011': 'Y', '1100': 'Z', '01111': '1', '00111': '2', '00011': '3', '00001': '4', '00000': '5', '10000': '6', '11000': '7', '11100': '8', '11110': '9', '11111': '0'}

# function to translate blink pattern to Morse code
def blink_to_morse(blink_pattern):
    return morse_dict.get(blink_pattern, '')

# initialize video capture
cap = cv2.VideoCapture(0)

while True:
    # capture frame-by-frame
    ret, frame = cap.read()
    
    # detect faces in the frame
    faces = detector(frame)
    
    # iterate over the faces
    for face in faces:
        # get the landmarks for the face
        landmarks = predictor(frame, face)
        
        # get the left and right eye landmarks
        left_eye_landmarks = np.array([(landmarks.part(36).x, landmarks.part(36).y), (landmarks.part(37).x, landmarks.part(37).y), (landmarks.part(38).x, landmarks.part(38).y), (landmarks.part(39).x, landmarks.part(39).y), (landmarks.part(40).x, landmarks.part(40).y), (landmarks.part(41).x, landmarks.part(41).y)], np.int32)
        right_eye_landmarks = np.array([(landmarks.part(42).x, landmarks.part(42).y), (landmarks.part(43).x, landmarks.part(43).y), (landmarks.part(44).x, landmarks.part(44).y), (landmarks.part(45).x, landmarks.part(45).y),(landmarks.part(46).x, landmarks.part(46).y), (landmarks.part(47).x, landmarks.part(47).y)], np.int32)

            # calculate eye aspect ratio (EAR) for left eye
        left_eye_aspect_ratio = (np.linalg.norm(left_eye_landmarks[1]-left_eye_landmarks[5]) + np.linalg.norm(left_eye_landmarks[2]-left_eye_landmarks[4])) / (2 * np.linalg.norm(left_eye_landmarks[0]-left_eye_landmarks[3]))
        
        # calculate eye aspect ratio (EAR) for right eye
        right_eye_aspect_ratio = (np.linalg.norm(right_eye_landmarks[1]-right_eye_landmarks[5]) + np.linalg.norm(right_eye_landmarks[2]-right_eye_landmarks[4])) / (2 * np.linalg.norm(right_eye_landmarks[0]-right_eye_landmarks[3]))
        
        # calculate average eye aspect ratio (EAR)
        avg_eye_aspect_ratio = (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2
        
        # check if the eyes are closed
        if avg_eye_aspect_ratio < EAR_THRESHOLD:
            if not blinking:
                # if this is the first frame the eyes are closed, set the start time
                blink_start_time = time.time()
                blinking = True
        else:
            if blinking:
                # if this is the first frame the eyes are open, calculate the blink duration
                blink_duration = time.time() - blink_start_time
                
                # check if the blink duration is long enough to count as a blink
                if blink_duration > BLINK_DURATION_THRESHOLD:
                    # calculate the blink pattern based on the blink duration
                    blink_pattern = '1' if blink_duration > BLINK_DURATION_THRESHOLD * 3 else '0'
                    
                    # add the blink pattern to the message
                    message += blink_pattern
                    
                    # translate the blink pattern to Morse code
                    morse_code = blink_to_morse(message)
                    
                    # print the decoded Morse code
                    print(morse_code)
                
                # reset the blinking variables
                message = ''
                blinking = False
        
        # draw the eye landmarks on the frame
        cv2.polylines(frame, [left_eye_landmarks], True, (0, 255, 255), 2)
        cv2.polylines(frame, [right_eye_landmarks], True, (0, 255, 255), 2)
        
        # draw the EAR value on the frame
        cv2.putText(frame, f"EAR: {avg_eye_aspect_ratio:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()