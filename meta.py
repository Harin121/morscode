import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe hand tracking
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    # Define alphabet mapping
    alphabet = {
         0: 'A',
         1: 'B',
         2: 'C',
         3: 'D',
         4: 'E',
         5: 'F',
         6: 'G',
         7: 'H',
         8: 'I',
         9: 'J',
        10: 'K',
        11: 'L',
        12: 'M',
        13: 'N',
        14: 'O',
        15: 'P',
        16: 'Q',
        17: 'R',
        18: 'S',
        19: 'T',
        20: 'U',
        21: 'V',
        22: 'W',
        23: 'X',
        24: 'Y',
        25: 'Z',

    }

    while cap.isOpened():
        # Read frames from the video stream
        success, image = cap.read()
        if not success:
            break

        # Convert the image to RGB and pass it to the hand tracking model
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw hand landmarks on the image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the coordinates of the thumb and index finger
                thumb_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
                thumb_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
                index_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
                index_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y

                # Check if the thumb is to the left of the index finger
                if thumb_x < index_x:
                    # Calculate the distance between the thumb and index finger
                    distance = ((thumb_x - index_x)**2 + (thumb_y - index_y)**2)**0.5

                    # Map the distance to a letter of the alphabet
                    letter_index = int(distance / 0.04)
                    if letter_index > 25:
                        letter_index = 25

                    # Print the corresponding letter to the console
                    print(alphabet[letter_index])

        # Display the resulting image
        cv2.imshow('Hand Gestures', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()