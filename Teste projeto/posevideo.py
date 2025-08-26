# ====================================================================
# Título do projeto: Monitoramento automatizado de indivíduos em situação de vulnerabilidade
# Nome do programa: posevideo.py
# Exemplo de chamada (Linux): python3 posevideo.py
# Autores:
#  - Jorge Luiz Pinto Junior — RA: 11058715 — CEO
#  - Marcos Baldrigue Andrade — RA: 11201921777 — CFO (Financeiro)
#  - Guilherme Eduardo Pereira — RA: 11201720498 — CPO (Desenvolvimento)
# Data: 2025-07-18
# ====================================================================

import cv2
import time
import mediapipe as mp
import numpy as np
import math
import requests
import os
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
MENSAGEM = 'Pessoa observada esta de pé ou saiu do alcance de visão'

# === PARÂMETROS DE CALIBRAÇÃO ===
fs = cv2.FileStorage("calibration.xml", cv2.FILE_STORAGE_READ)
camera_matrix = fs.getNode("camera_matrix").mat()
dist_coeffs = fs.getNode("distortion_coefficients").mat()
fs.release()

def enviar_mensagem(texto):
    url = f'https://api.telegram.org/bot{TOKEN}/sendMessage'
    payload = {
        'chat_id': CHAT_ID,
        'text': texto
    }

    response = requests.post(url, data=payload)
    
    if response.status_code == 200:
        print('✅ Mensagem enviada com sucesso!')
    else:
        print(f'❌ Erro ao enviar mensagem. Código: {response.status_code}')
        print(response.text)

# === Funções de cálculo e classificação ===

def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    ba, bc = a - b, c - b
    ba_norm, bc_norm = np.linalg.norm(ba), np.linalg.norm(bc)
    if ba_norm == 0 or bc_norm == 0:
        return 180.0
    cos_ang = np.dot(ba, bc) / (ba_norm * bc_norm)
    cos_ang = np.clip(cos_ang, -1.0, 1.0)
    return math.degrees(math.acos(cos_ang))

def classify_pose(landmarks):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    width, height = max(xs) - min(xs), max(ys) - min(ys)
    if width > height * 1.3:
        return 'Deitada'
    mp_pose = mp.solutions.pose
    left_hip   = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,  landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    left_knee  = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    right_hip   = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,  landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    right_knee  = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    left_angle  = calculate_angle(left_hip, left_knee, left_ankle)
    right_angle = calculate_angle(right_hip, right_knee, right_ankle)
    if (left_angle + right_angle) / 2.0 < 160:
        return 'Sentada'
    return 'Em pé'

# === Loop principal com monitoramento de tempo ===

def main():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError('Não foi possível acessar a webcam.')

    em_pe_start = None
    ausente_start = None
    em_pe_alertado = False
    ausente_alertado = False
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('Falha ao capturar frame da webcam.')
                break
                
            # === CORRIGE DISTORÇÃO USANDO CALIBRAÇÃO ===
            frame_undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)
            
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            annotated = frame.copy()
            label = 'Ausente'
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    annotated, results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
                )
                label = classify_pose(results.pose_landmarks.landmark)

                # Reinicia o estado de ausência
                ausente_start = None
                ausente_alertado = False

                # Detecção de "em pé"
                if label == 'Em pé':
                    if em_pe_start is None:
                        em_pe_start = time.time()
                    elif not em_pe_alertado and (time.time() - em_pe_start) >= 5:
                        enviar_mensagem("Pessoa está em pé por 5 segundos!")
                        em_pe_alertado = True
                else:
                    em_pe_start = None
                    em_pe_alertado = False
            else:
                # Ninguém na imagem
                em_pe_start = None
                em_pe_alertado = False

                if ausente_start is None:
                    ausente_start = time.time()
                elif not ausente_alertado and (time.time() - ausente_start) >= 5:
                    enviar_mensagem("Ninguém detectado na câmera por 5 segundos!")
                    ausente_alertado = True                    
                # sobrepõe o rótulo
            font, pos, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, (20,30), 1.0, 2
            cv2.putText(annotated, label, pos, font, scale, (0,0,0), thickness+2, cv2.LINE_AA)
            cv2.putText(annotated, label, pos, font, scale, (255,255,255), thickness, cv2.LINE_AA)
            cv2.imshow('Pose Detection (press q to quit)', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()

if __name__ == '__main__':
    main()

