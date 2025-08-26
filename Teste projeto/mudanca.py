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