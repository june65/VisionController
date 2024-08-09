import cv2
import torch
import mediapipe as mp
import numpy as np
from torchvision.transforms import Compose, ToPILImage, Resize, ToTensor, Normalize

model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.eval()

# GPU 사용 여부 설정
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")
midas.to(device)

transform = Compose([
    ToPILImage(),
    Resize((256, 256)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)

video = cv2.VideoCapture('data/data3.mp4')

if not video.isOpened():
    print("Error: Could not open video file.")
    exit()

stream = torch.cuda.Stream()

while True:
    ret, frame = video.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 이미지 전처리
    input_image = transform(rgb_frame).to(device).unsqueeze(0) 
    
    with torch.cuda.stream(stream):
        with torch.no_grad():
            prediction = midas(input_image)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
    depth_map = prediction.cpu().numpy()
    pad_value_top = depth_map[0, :]
    pad_value_bottom = depth_map[-1, :]
    pad_value_left = depth_map[:, 0]
    pad_value_right = depth_map[:, -1]
    pad_size = 50
    depth_map = np.pad(depth_map, ((pad_size, pad_size), (pad_size, pad_size)), mode='edge')

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
            left_eye_depth = depth_map[left_eye_center[1]+pad_size//2, left_eye_center[0]+pad_size//2]
            cv2.putText(frame, f'Depth: {left_eye_depth:.2f}', (left_eye_center[0], left_eye_center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            right_eye_landmarks = [face_landmarks.landmark[i] for i in [362, 263, 387, 373, 380, 374, 386]]
            right_eye_center = (
                int(sum([lm.x for lm in right_eye_landmarks]) / len(right_eye_landmarks) * frame.shape[1]),
                int(sum([lm.y for lm in right_eye_landmarks]) / len(right_eye_landmarks) * frame.shape[0])
            )
            cv2.circle(frame, right_eye_center, 5, (0, 255, 0), -1)
            right_eye_depth = depth_map[right_eye_center[1]+pad_size//2, right_eye_center[0]+pad_size//2]
            cv2.putText(frame, f'Depth: {right_eye_depth:.2f}', (right_eye_center[0], right_eye_center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if hand_result.multi_hand_landmarks:
        for hand_landmarks in hand_result.multi_hand_landmarks:
            thumb_tip = hand_landmarks.landmark[4]
            thumb_tip_coords = (int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0]))
            thumb_depth = depth_map[thumb_tip_coords[1]+pad_size//2, thumb_tip_coords[0]+pad_size//2]
            cv2.circle(frame, thumb_tip_coords, 5, (255, 0, 0), -1)
            cv2.putText(frame, f'Thumb Depth: {thumb_depth:.2f}', (thumb_tip_coords[0], thumb_tip_coords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            index_finger_tip = hand_landmarks.landmark[8]
            index_finger_tip_coords = (int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0]))
            index_finger_depth = depth_map[index_finger_tip_coords[1]+pad_size//2, index_finger_tip_coords[0]+pad_size//2]
            cv2.circle(frame, index_finger_tip_coords, 5, (0, 0, 255), -1)
            cv2.putText(frame, f'Index Depth: {index_finger_depth:.2f}', (index_finger_tip_coords[0], index_finger_tip_coords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('Eye Tracker', frame)

    if cv2.waitKey(1) == 27:
        break

video.release()
cv2.destroyAllWindows()
