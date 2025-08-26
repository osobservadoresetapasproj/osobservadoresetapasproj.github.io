import cv2
import numpy as np
import glob

# Definindo as dimensões do tabuleiro de xadrez
CHECKERBOARD = (6, 8)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Vetores para armazenar os pontos 3D e 2D
objpoints = []
imgpoints = []

# Coordenadas do mundo para os pontos 3D
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Carregando as imagens
images = glob.glob('*.jpg')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                    cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imwrite("tab_" + fname, img)

cv2.destroyAllWindows()

# Calibração
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Impressão dos resultados
print("Camera matrix : \n", mtx)
print("Distortion coefficients : \n", dist)
print("Rotation Vectors : \n", rvecs)
print("Translation Vectors : \n", tvecs)

# Salvando os parâmetros em um arquivo XML
fs = cv2.FileStorage("calibration.xml", cv2.FILE_STORAGE_WRITE)
fs.write("camera_matrix", mtx)
fs.write("distortion_coefficients", dist)

# Salva os vetores de rotação e translação (como listas de matrizes)
fs.write("rotation_vectors", np.array(rvecs))
fs.write("translation_vectors", np.array(tvecs))

fs.release()
print("Parâmetros de calibração salvos em 'calibration.xml'")
