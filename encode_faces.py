#  HeimdallEYE
# Cuando compilamos o codificamos en un dispositivo de bajo recursos de GPU tarda en cargar
# python encode_faces.py --dataset dataset --encodings encodings.pickle --detection-method cnn
# Para codificar mas rapido en Raspberry usamos este:
# python encode_faces.py --dataset dataset --encodings encodings.pickle --detection-method hog

# Importamos los paquetes necesarios
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

# Contruimos el analizador de argumentos sobre nuestra base de datos y analizamos estos argumentos mediante los diferectes metodos
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# Toma la ruta a las imagenes de entrada en nuestro cojunto de base de datos
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

# Inicializa la lista de codificaciones conocidas sobre nuestra base de datos 
knownEncodings = []
knownNames = []

# Recorre las rutas de las Imagenes
for (i, imagePath) in enumerate(imagePaths):
	# Extraer el nombre de la persona en la rutad de la imagen
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	# Carga la imagen de entrada y convertida desde RGB (Pedido desde OPENCV)
	# Ordenador por DLIB (RGB)
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# Dectecta la coordenadas de ejes (x,y) de los cuadros que son los limitadores
	# Que corresponde a cada cara en la entrada de imagen en nuestra camara
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])

	# Calcula la incustacion facial para la cara
	encodings = face_recognition.face_encodings(rgb, boxes)

	# Recorremos las codificaciones
	for encoding in encodings:
		# Agregamos cada codificacion + nombre a nuestro cojunto de nombre conocidos codificados
		
		knownEncodings.append(encoding)
		knownNames.append(name)

# Imprimimos la codificaciones Faciales + nombre en el disco 
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()
