#Iniciamos el proyecto inicial

#Importamos las librerias
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

#Ahora cargamos el modelo preentrenado desde el archivo
model=load_model("Keras/keras_model.h5")

#Iniciamos la camara
cap=cv2.VideoCapture(0)
class_labels=['WebCam','Teclados']

while True:
  #Capturar los de frame de la camara
  ret,frame=cap.read()
  
  #Preprocesamiento de los frames para clasificar
  img=cv2.resize(frame,(224,224))
  img_tensor=image.img_to_array(img)
  img_tensor=np.expand_dims(img_tensor,axis=0)
  img_tensor/=255.
  
  #Uso del modelo de clasificaciones de frame
  prediction=model.predict(img_tensor)
  
  #Obtenemos la etiquetas de clases para el fotograma
  class_index=np.argmax(prediction[0]) 
  
 #Obtenemos la clase de el label para el frame
  class_label=class_labels[class_index]
  print(class_label)
  
  #Mostrar el fotograma con la etiqueta de clases
  cv2.putText(frame,class_label,(10,30),cv2.FONT_HERSHEY_SIMPLEX,
              1.0,(255,255,255),3)
  cv2.imshow('Reconocimiento de perifericos',frame)
 
 #Haremos una función para que el bucle se termine presionando "s"
  if cv2.waitKey(1) & 0xFF== ord('s'):
      break
  
  #Realizamos la última parte
  #Libera la camara web y cierra todas las ventanas
  cap.release()
  cv2.destroyAllWindows()
   