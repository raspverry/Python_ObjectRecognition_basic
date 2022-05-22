import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

'''
filename = './Images/dog.jpg'
im = Image.open(filename)
im = im.resize((224,224))
#im.show()

#Tensorflow

# deep learning module weight - pre-trained
#mobile = tf.keras.applications.mobilenet.MobileNet()  #v1
mobile = tf.keras.applications.mobilenet_v2.MobileNetV2()

imgArray = np.array(im)
final_image = np.expand_dims(imgArray, axis =0) ## need fourth dimension
final_image = tf.keras.applications.mobilenet.preprocess_input(final_image)
print(final_image.shape)
prediction = mobile.predict(final_image)
result = tf.keras.applications.imagenet_utils.decode_predictions(prediction)
print(result)

'''

#using opencv, set-up model


config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
fronzen_model = 'frozen_inference_graph.pb' 

model = cv2.dnn_DetectionModel(fronzen_model, config_file)

class_labels = []
file_name = 'coco.names'
with open(file_name, 'rt') as fpt:
    class_labels = fpt.read().rstrip('\n').split('\n')

model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)


#------------------------------------------------------------------------

#Image Capture

img = cv2.imread('./Images/dog.jpg')
#cv2.imshow('image', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#cv2.waitKey()



classIndex, confidence, bbox = model.detect(img, confThreshold=0.5)
font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
for ClassInd, conf, boxes in zip(classIndex.flatten(), confidence.flatten(), bbox):
    cv2.rectangle(img, boxes, (255,0,0), 2)
    cv2.putText(img, class_labels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40), font, fontScale=font_scale, color=(0,255,0), thickness=3)

#cv2.imshow('image', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#cv2.waitKey()



#-------------------------------------------------------------------------
#Video capture

#load the video
#cap = cv2.VideoCapture('name of video')

#load webcam
cap = cv2.VideoCapture(0)

#Check video is opened correctly
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError('Cannot open video')

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret, frame = cap.read()
    
    ClassIndex, confidence, bbox = model.detect(frame, confThreshold = 0.55)
    
    
    if(len(ClassIndex) != 0):
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if(ClassInd <= 91):
                cv2.rectangle(frame, boxes, (255,0,0), 2)
                cv2.putText(frame, class_labels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40), font, fontScale=font_scale, color=(0,255,0), thickness=3)

    cv2.imshow('Object detection', frame)
    
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()