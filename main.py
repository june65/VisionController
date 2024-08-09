import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)

video = cv2.VideoCapture('data/data3.mp4')

if not video.isOpened():
    print("Error: Could not open video file.")
    exit()

while True:
    ret, frame = video.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = face_mesh.process(rgb_frame)

    hand_result = hands.process(rgb_frame)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:

            left_eye_landmarks = [face_landmarks.landmark[i] for i in [33, 133, 159, 145, 153, 144, 163]]
            left_eye_center = (
                int(sum([lm.x for lm in left_eye_landmarks]) / len(left_eye_landmarks) * frame.shape[1]),
                int(sum([lm.y for lm in left_eye_landmarks]) / len(left_eye_landmarks) * frame.shape[0])
            )
            cv2.circle(frame, left_eye_center, 5, (0, 255, 0), -1)

            right_eye_landmarks = [face_landmarks.landmark[i] for i in [362, 263, 387, 373, 380, 374, 386]]
            right_eye_center = (
                int(sum([lm.x for lm in right_eye_landmarks]) / len(right_eye_landmarks) * frame.shape[1]),
                int(sum([lm.y for lm in right_eye_landmarks]) / len(right_eye_landmarks) * frame.shape[0])
            )
            cv2.circle(frame, right_eye_center, 5, (0, 255, 0), -1)


    if hand_result.multi_hand_landmarks:
        for hand_landmarks in hand_result.multi_hand_landmarks:
            thumb_tip = hand_landmarks.landmark[4]
            thumb_tip_coords = (int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0]))
            cv2.circle(frame, thumb_tip_coords, 5, (255, 0, 0), -1)

            index_finger_tip = hand_landmarks.landmark[8]
            index_finger_tip_coords = (int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0]))
            cv2.circle(frame, index_finger_tip_coords, 5, (0, 0, 255), -1)

            cv2.putText(frame, f'Thumb: {thumb_tip_coords}', (thumb_tip_coords[0], thumb_tip_coords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, f'Index: {index_finger_tip_coords}', (index_finger_tip_coords[0], index_finger_tip_coords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('Eye Tracker', frame)

    if cv2.waitKey(1) == 27:
        break
   
video.release()
cv2.destroyAllWindows()
