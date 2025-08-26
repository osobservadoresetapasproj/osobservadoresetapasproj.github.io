import cv2
import mediapipe as mp
import numpy as np
import math


def calculate_angle(a, b, c):
    """Calculates the angle (in degrees) at point b given three points a, b, c."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    # Normalize vectors
    ba_norm = np.linalg.norm(ba)
    bc_norm = np.linalg.norm(bc)
    if ba_norm == 0 or bc_norm == 0:
        return 180.0
    cosine_angle = np.dot(ba, bc) / (ba_norm * bc_norm)
    # Clamp to handle numerical errors
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = math.degrees(math.acos(cosine_angle))
    return angle


def classify_pose(landmarks):
    """
    Classify pose as standing, sitting or lying based on landmarks.
    landmarks: list of mediapipe landmark objects with .x, .y coordinates (normalized).
    Returns a string: 'Em pé', 'Sentada' or 'Deitada'.
    """
    # Compute bounding box dimensions
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    # Heuristic for lying: width significantly larger than height
    # Use a threshold ratio; adjust as necessary
    if width > height * 1.3:
        return 'Deitada'

    # Compute knee angles
    mp_pose = mp.solutions.pose
    # Left leg
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    # Right leg
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

    # Compute angles
    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
    avg_knee_angle = (left_knee_angle + right_knee_angle) / 2.0

    # Heuristics: if knees are bent (angle < 160 degrees), assume sitting
    if avg_knee_angle < 160:
        return 'Sentada'
    else:
        return 'Em pé'


def main():
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    img = cv2.imread('teste.jpeg')
    if img is None:
        raise FileNotFoundError('Não foi possível abrir pessoa.jpg')

    with mp_pose.Pose(static_image_mode=True, model_complexity=2,
                      enable_segmentation=False) as pose:
        # Converter imagem de BGR para RGB
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    annotated = img.copy()
    if results.pose_landmarks:
       mp_drawing.draw_landmarks(
          annotated, results.pose_landmarks,
          mp_pose.POSE_CONNECTIONS,
          landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
       )
       pose_label = classify_pose(results.pose_landmarks.landmark)
       print(pose_label)
       # Desenha o texto na imagem
       font = cv2.FONT_HERSHEY_SIMPLEX
       text_position = (30, 40)
       font_scale = 1.0
       thickness = 2
       # Contorno preto
       cv2.putText(annotated, pose_label, text_position, font,
                   font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
       # Texto branco
       cv2.putText(annotated, pose_label, text_position, font,
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    else:
       print('Nenhuma pessoa detectada.')
   
   cv2.imwrite('pose_detectada.jpg', annotated)

if __name__ == '__main__':
    main()
