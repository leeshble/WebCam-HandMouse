import cv2
import pyautogui
from tkinter import *
import time
import numpy as np
from IPython.display import display

from six import BytesIO

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from tensorflow.keras import models

root = Tk()
monitor_height = root.winfo_screenheight()
monitor_width = root.winfo_screenwidth()
print("Your screen size is  %d x %d" %(monitor_width, monitor_height))

labelmap_path = './labels/label_map.pbtxt'

category_index = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)

print("Loading Model...")
tf.keras.backend.clear_session()
model = tf.saved_model.load(f'./model/saved_model')

def load_image_into_numpy_array(path):
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  input_tensor = tf.convert_to_tensor(image)
  input_tensor = input_tensor[tf.newaxis,...]

  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)

  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

  if 'detection_masks' in output_dict:
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict

def run_inference(model, cap):
    while cap.isOpened():
        ret, image_np = cap.read()
        # Actual detection.
        output_dict = run_inference_for_single_image(model, image_np)
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=8)
        math(output_dict)
        cv2.imshow('object_detection', cv2.resize(image_np, (800, 600)))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

def math(output_dict):
    accuracy = output_dict['detection_scores'][0]
    if accuracy >= 0.5:
      label = output_dict['detection_classes'][0]
      if label == 1:
        position_data_y_min = output_dict['detection_boxes'][0][0].tolist()
        position_data_x_min = output_dict['detection_boxes'][0][1].tolist()

        position_data_y_max = output_dict['detection_boxes'][0][2].tolist()
        position_data_x_max = output_dict['detection_boxes'][0][3].tolist()

        position_data_x = (position_data_x_max + position_data_x_min)/2
        position_data_y = (position_data_y_max + position_data_y_min)/2
        screen_x_temp = int(position_data_x * monitor_width)
        screen_x = monitor_width - screen_x_temp
        screen_y = int(position_data_y * monitor_height)
        print("mouse pos = %d x %d" %(screen_x, screen_y))
        #print(y)
        pyautogui.moveTo(screen_x, screen_y)
      if label == 2:
        pyautogui.click()
        print("clicked")


cap = cv2.VideoCapture(0) # or cap = cv2.VideoCapture("<video-path>")
run_inference(model, cap)
cap.release()
cv2.destroyAllWindows()