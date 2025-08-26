import cv2, mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

img = cv2.imread('idoso1.png')
with mp_pose.Pose(static_image_mode=True, model_complexity=2,
                  enable_segmentation=True) as pose:
    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

annotated = img.copy()
mp_drawing.draw_landmarks(
    annotated, results.pose_landmarks,
    mp_pose.POSE_CONNECTIONS,
    landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
)
cv2.imwrite('pose_detectada.jpg', annotated)
