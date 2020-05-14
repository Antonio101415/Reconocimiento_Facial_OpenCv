# HeimdallEYE USO:
# python pi_face_recognition.py --cascade haarcascade_frontalface_default.xml --encodings encodings.pickle

# Importamos los paquetes / librerias necesarios
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
import os 

# Contruimos el analizador de argumentos y analizamos estos argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
	help = "path to where the face cascade resides")
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
args = vars(ap.parse_args())

# Cargamos la caras conocidas y las incrustaciones junto Haar de OpenCV (Cascada para deteccion de rostros)
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(args["encodings"], "rb").read())
detector = cv2.CascadeClassifier(args["cascade"])

# Inizalizamos la transmision de video y permitimos activar el sensor de Camara
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

time.sleep(2.0)


fps = FPS().start()

# Bucle sobre cuadros de la secuencia del archivo de video
while True:
	# Toma el fotograma de la secuencia de video y cambia el tamaño a 500px (Para poder que nuestra raspberry acelere el procesamiento)
	frame = vs.read()
	frame = imutils.resize(frame, width=500)
	
	# Convierte el marco de entrada (1) BGR a escala de grises para la deteccion de la cara y (2) de BGR pasamos a RGB para el reconocimiento facial 
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# Detectamos caras en el marco de escala de grises 
	rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)

	# OpenCV  devuelve las coordenadas del cuadro delimitador en orden  (x,y,w,h) 
	# Pero los necesitamos en el orden (Arriba , derecha , abajo , izquierda ) , Por lo que necesitamos reogarnizarlo

	boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

	#Calcula la incustraciones faciales para cada cuadro delimitador de cara 
	encodings = face_recognition.face_encodings(rgb, boxes)
	names = []

	# Bucle sobre las incrustaciones faciales
	for encoding in encodings:
		# Intente hacer coincidir cada cara en la imagen de entrada con nuestra coincidencia  , no es asi es Desconocido 
		matches = face_recognition.compare_faces(data["encodings"],
			encoding)
		name = "DESCONOCIDO"
			
		

		# Verificamos si hemos encontrado una coincidencia con nuestra BD
		if True in matches:
			# Encontramos los indices de todas las caras que coinciden y luego inicializamos
			#Un diccionario para contar el numero totoal de veces que cada cara fue emparejado
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			# Recorre los indices coincidentes y mantiene un recuento de cada cara reconocida en BD
			for i in matchedIdxs:
				name = data["names"][i]
			

				counts[name] = counts.get(name, 0) + 1
			

			# Determinamos la cara reconocida con el mayor numero de casos 
			# En Python es poco problable el caso de empate 
			# Seleccionara la primera entrada del diccionario
			name = max(counts, key=counts.get)
		
		# Actualizamos la lista de nombre
		names.append(name)
	#Esta en prueba :
	#if name  == data["antonio"] :
	#	os.system("echo Antonio | festival --tts")


	# Realizamos un bucle para el reconocimiento facial
	for ((top, right, bottom, left), name) in zip(boxes, names):
		# Dibujamos el nombre de la cara pronosticada en el frame 
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, (0, 255, 0), 2)

	# Imprimimos el Frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# Para salir del bucle solo pulse la letra q y saldra
	if key == ord("q"):
		break

	# Actualizamos el FPS COUNTER
	fps.update()

# Paramos el tiempo de actualizacion
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# Y realizamos la limpieza del Frame
cv2.destroyAllWindows()
vs.stop()
