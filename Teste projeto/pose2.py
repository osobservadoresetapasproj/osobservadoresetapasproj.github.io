import cv2
import mediapipe as mp
import numpy as np
import math

def calculate_angle(a, b, c):
    """Calcula o ângulo (em graus) no ponto b dados três pontos a, b e c."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    ba_norm = np.linalg.norm(ba)
    bc_norm = np.linalg.norm(bc)
    if ba_norm == 0 or bc_norm == 0:
        return 180.0
    cosine_angle = np.dot(ba, bc) / (ba_norm * bc_norm)
    # Limita o cosseno para evitar erros numéricos
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = math.degrees(math.acos(cosine_angle))
    return angle

def classify_pose(landmarks):
    """
    Classifica a pose como 'Em pé', 'Sentada' ou 'Deitada' com base nos marcos.
    """
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    width  = max(xs) - min(xs)
    height = max(ys) - min(ys)

    # Se o corpo estiver mais horizontal (largura maior que 1.3x a altura), assume deitado.
    if width > height * 1.3:
        return 'Deitada'

    mp_pose = mp.solutions.pose
    # Coordenadas normalizadas dos marcos dos quadris, joelhos e tornozelos.
    left_hip   = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    left_knee  = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    right_hip   = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    right_knee  = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

    # Calcula os ângulos dos joelhos esquerdo e direito.
    left_angle  = calculate_angle(left_hip, left_knee, left_ankle)
    right_angle = calculate_angle(right_hip, right_knee, right_ankle)
    avg_knee_angle = (left_angle + right_angle) / 2.0

    # Se os joelhos estiverem significativamente dobrados (<160°), assume-se sentado.
    if avg_knee_angle < 160:
        return 'Sentada'
    else:
        return 'Em pé'

mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

img = cv2.imread('teste4.jpg')
with mp_pose.Pose(static_image_mode=True, model_complexity=2,
                  enable_segmentation=False) as pose:
    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

annotated = img.copy()
if results.pose_landmarks:
    mp_drawing.draw_landmarks(
        annotated, results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
    )
    # Classifica e imprime a posição no prompt
    classificacao = classify_pose(results.pose_landmarks.landmark)
    print(classificacao)
else:
    print('Nenhuma pessoa detectada.')

cv2.imwrite('pose_detectada.jpg', annotated)
