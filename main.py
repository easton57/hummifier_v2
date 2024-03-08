import os
import cv2
import logging
import matplotlib

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from datetime import datetime
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.builders import model_builder
from object_detection.utils import visualization_utils as viz_utils

# Change backend for image analysis
matplotlib.use('TkAgg')


def main():
	""" Driver for the model and program """
	model_name = "hummifier_model"
	model_annotations = "Tensorflow/workspace/annotations/label_map.pbtxt"

	choice = input("Would you like infer on live video or photo?\n\n"
	               "1. Live video\n"
	               "2. Photo\n"
	               "3. Video\n"
	               "0. Quit\n")

	if choice == "1" or choice.lower() == "live video":
		video_infer(model_name, model_annotations, 0)
	elif choice == "2" or choice.lower() == "photo":
		# Print available photos in the folder and get input
		print("The available images are below. Please add files to the test_images directory in the project files\n")

		image_list = os.listdir("test_images")
		for i in range(len(image_list)):
			print(f"{i + 1}: {image_list[i]}")

		image_choice = int(input("\nInput which image number you'd like to analyze: "))

		# Call the inference function
		image_infer(model_name, model_annotations, image_list[image_choice - 1])
	elif choice == "3" or choice.lower() == "video":
		# Print available photos in the folder and get input
		print("The available videos are below. Please add files to the test_images directory in the project files\n")

		video_list = os.listdir("test_videos")
		for i in range(len(video_list)):
			print(f"{i + 1}: {video_list[i]}")

		video_choice = int(input("\nInput which video number you'd like to analyze: "))

		# Call the inference function
		video_infer(model_name, model_annotations, video_list[video_choice - 1])
	elif choice == "0" or choice.lower() == "quit":
		exit()
	else:
		print("Invalid option. Please try again.\n\n")

	main()


def image_infer(model_name, model_annotations, image_name):
	""" Function to infer hummingbird species from images """
	models_dir = "Tensorflow/workspace/models/"
	saved_model = os.path.join(models_dir, model_name + "/export/saved_model")

	# Load saved model
	model = tf.saved_model.load(saved_model)

	# Load labels
	category_index = label_map_util.create_category_index_from_labelmap(model_annotations, use_display_name=True)

	print('Running inference for {}... '.format(image_name), end='')

	image_path = os.path.join('test_images', image_name)
	image_np = np.array(Image.open(image_path))

	# The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
	input_tensor = tf.convert_to_tensor(image_np)

	# The model expects a batch of images, so add an axis with `tf.newaxis`.
	input_tensor = input_tensor[tf.newaxis, ...]

	# input_tensor = np.expand_dims(image_np, 0)
	detections = model(input_tensor)

	# Convert to numpy arrays, and take index [0] to remove the batch dimension.
	num_detections = int(detections.pop('num_detections'))
	detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
	detections['num_detections'] = num_detections

	# detection_classes should be ints.
	detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

	image_np_with_detections = image_np.copy()

	viz_utils.visualize_boxes_and_labels_on_image_array(
		image_np_with_detections,
		detections['detection_boxes'],
		detections['detection_classes'],
		detections['detection_scores'],
		category_index,
		use_normalized_coordinates=True,
		max_boxes_to_draw=1,
		min_score_thresh=.30,
		agnostic_mode=False)

	plt.figure()
	plt.imshow(image_np_with_detections)
	print('Done\n')
	plt.show()


def video_infer(model_name, model_annotations, video_name):
	""" Function to infer from a webcam stream """
	# Set logging
	logging.basicConfig(filename=f"logs/hummifier_visits_{datetime.today().strftime('%Y-%m-%d')}.log", filemode='a',
	                    level=logging.DEBUG, force=True, format='[%(asctime)s] %(name)s %(levelname)s - %(message)s')

	models_dir = "Tensorflow/workspace/models/"

	if video_name != 0:
		video_path = os.path.join('test_videos', video_name)
	else:
		video_path = 0

	path_to_ckpt = os.path.join(models_dir, os.path.join(model_name + "/export/", 'checkpoint/'))
	path_to_cfg = os.path.join(models_dir, os.path.join(model_name + "/export/", 'pipeline.config'))

	# Load pipeline config and build a detection model
	configs = config_util.get_configs_from_pipeline_file(path_to_cfg)
	model_config = configs['model']
	detection_model = model_builder.build(model_config=model_config, is_training=False)

	# Restore checkpoint
	ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
	ckpt.restore(os.path.join(path_to_ckpt, 'ckpt-0')).expect_partial()

	@tf.function
	def detect_fn(image):
		"""Detect objects in image."""

		image, shapes = detection_model.preprocess(image)
		prediction_dict = detection_model.predict(image, shapes)
		detections = detection_model.postprocess(prediction_dict, shapes)

		return detections, prediction_dict, tf.reshape(shapes, [-1])

	# Load the labels
	category_index = label_map_util.create_category_index_from_labelmap(model_annotations, use_display_name=True)

	# Start the webcam stream
	cap = cv2.VideoCapture(video_path)

	wait = 0

	# Loop and read the stream
	while True:
		# Read frame from camera
		ret, image_np = cap.read()

		if not ret:
			print("Can't receive frame (stream end?). Exiting ...")
			break

		input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
		detections, predictions_dict, shapes = detect_fn(input_tensor)

		label_id_offset = 1
		image_np_with_detections = image_np.copy()

		viz_utils.visualize_boxes_and_labels_on_image_array(
			image_np_with_detections,
			detections['detection_boxes'][0].numpy(),
			(detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
			detections['detection_scores'][0].numpy(),
			category_index,
			use_normalized_coordinates=True,
			max_boxes_to_draw=1,
			min_score_thresh=.60,
			agnostic_mode=False)

		# Display output
		cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))

		# if the data changes, write date, time, species to log
		if wait == 0 and detections['num_detections'] > 0:
			logging.info(f"Hummingbird detected: {category_index[image_np_with_detections.shape[2]].get('name')}")
			wait = 100
		elif wait > 0:
			wait -= 1
		else:
			pass

		if cv2.waitKey(25) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	# Some tensorflow logging stuff
	tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

	main()
