'''
How to run: 
python test_accuracy_updated.py --folder path_to_regression_dataset --flag 1_for_separate_classes/0_for_combined_classes

and change the labels accordingly and path to the model files
The labels need to be changed in the dictionary in the functions h5_acc, tflite_acc, tflite_quant_acc, 
pb_acc and in dict_ variable on line 271
'''

import os
import cv2, csv, time, sys, argparse, keras
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pd
from keras.preprocessing import image as IMAGE
import statistics
from keras.models import load_model, Model
from tensorflow.keras.models import load_model
# from keras.applications.mobilenet_v2 import preprocess_input
# from keras.applications.densenet import DenseNet121,preprocess_input
from keras.applications.imagenet_utils import preprocess_input
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#import efficientnet.tfkeras

# from keras import backend as K

# def swish_activation(x):
#         return (K.sigmoid(x) * x)

# model.update({'swish_activation': Activation(swish_activation)})

h5_path="/home/user/Aparajit/EnvironmentSense/program/squeezenet_sai/squeezeNet_15_classes_with_empty.h5"
# tflite_path="/home/aparajit/Downloads/model_testing/densenet_15_classes_with_empty_separate_classes/tfmodel_normal.tflite"
# quantised_path="/home/ayush/Downloads/models/vgg16/tfmodel_quantized.tflite"
# frozen_graph_path="/home/ayush/Downloads/models/vgg16/frozen_inference_graph_opt.pb"

csv_path="/home/user/Aparajit/EnvironmentSense/program/confusion_matrices/squeezenet/"
# csv_path = "/home/user/Aparajit/EnvironmentSense/program"

# h5_path="/home/ayush/Downloads/models/densenet/densenet-15_class_with_empty.h5"
# tflite_path="/home/ayush/Downloads/models/densenet/tfmodel_normal.tflite"
# quantised_path="/home/ayush/Downloads/models/densenet/tfmodel_quantized.tflite"
# frozen_graph_path="/home/ayush/Downloads/models/densenet/frozen_inference_graph_opt.pb"

# csv_path="/home/ayush/Downloads/csv/densenet"


# h5_path="/home/ayush/Downloads/models/efficientnet/efficientnet_15_class_with_empty.h5"
# csv_path="/home/ayush/Downloads/csv/efficientnet"

def write_to_csv(image_names, tflite_detection, pb_detection, tflite_quant_detection, h5_detection, h5_detections_temp, tf_time, pb_time, tf_quant_time, h5_time, h5_time_temp):
	with open(csv_path+'/acc_comparison.csv', 'a') as csv_file:
		csv_writer = csv.writer(csv_file)
		#for image_name, tf_detect, pb_detect, tf_quant_detect, h5_detect, tf_t, pb_t, tf_quant_t, h5_t in zip(image_names, tflite_detection, pb_detection, tflite_quant_detection, h5_detection, tf_time, pb_time, tf_quant_time, h5_time):
		csv_writer.writerow((image_names, tflite_detection, pb_detection, tflite_quant_detection, h5_detection, h5_detections_temp, tf_time, pb_time, tf_quant_time, h5_time, h5_time_temp))
		# csv_writer.writerow((image_names, h5_detection, h5_time))
		
		# csv_writer.writerow(("accuracy of pb", pb_acc))
		# csv_writer.writerow(("accuracy of tflite", tf_acc))
def write_to_csv_h5_tf(image_names,  h5_detection, h5_time):
	with open(csv_path+'/acc_comparison.csv', 'a') as csv_file:
		csv_writer = csv.writer(csv_file)
		#for image_name, tf_detect, pb_detect, tf_quant_detect, h5_detect, tf_t, pb_t, tf_quant_t, h5_t in zip(image_names, tflite_detection, pb_detection, tflite_quant_detection, h5_detection, tf_time, pb_time, tf_quant_time, h5_time):
		# csv_writer.writerow((image_names, tflite_detection, pb_detection, tflite_quant_detection, h5_detection, h5_detections_temp, tf_time, pb_time, tf_quant_time, h5_time, h5_time_temp))
		csv_writer.writerow((image_names, h5_detection, h5_time))
		
		# csv_writer.writerow(("accuracy of pb", pb_acc))
		# csv_writer.writerow(("accuracy of tflite", tf_acc))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    # print (e_x.sum(axis=1, keepdims=1))
    try:
        val=e_x / e_x.sum(axis=1, keepdims=1)
    except Exception as e:
        print (e)
        return(None)
    return val

def h5_acc_temp(model, img):
	x = IMAGE.img_to_array(img)
	# expanding the dimensions from (224,224,3) to (1, 224, 224, 3)
	x = np.expand_dims(x, axis=0)
	images = np.vstack([x])

	#Predicting the result
	# {'book_cover': 0, 'book_page': 1, 'book_page_graphics': 2, 'comic_book': 3, 'empty': 4, 'jrkitfaces': 5, 'jrkitfaces_box': 6, 'jrkitsticks': 7, 'jrkitsticks_box': 8, 'notebook': 9, 'tangram': 10, 'tangram_box': 11, 'workbook': 12, 'worksheet_cover': 13, 'worksheet_page': 14}
	y_out = model.predict(images)
	# y_out = softmax(y_out/1.1532955)
	print(f"length of output results : {len(y_out[0])}")
	values = dict()
	for i in range(len(dict_)):
		values[list(dict_.keys())[i]]=y_out[0][i]
	# values['book_cover'] = y_out[0][0]
	# values['book_page'] = y_out[0][1]
	# values['book_page_graphics'] = y_out[0][2]
	# values['jrkitfaces'] = y_out[0][3]
	# values['jrkitfaces_box'] = y_out[0][4]
	# values['jrkitsticks'] = y_out[0][5]
	# values['jrkitsticks_box'] = y_out[0][6]
	# values['tangram'] = y_out[0][7]
	# values['tangram_box'] = y_out[0][8]
	# values['worksheet_cover'] = y_out[0][9]
	# values['worksheet_page'] = y_out[0][10]

	max_key = max(values, key=lambda k: values[k])
	if values[max_key] < 0.01:
			max_key = "empty"
	print("max", max_key)
	return max_key

def h5_acc(model, img):
	# img=img/255.0
	# img=img-[0.485, 0.456, 0.406]
	# img=img/[0.229, 0.224, 0.225]
	x = IMAGE.img_to_array(img)
	# expanding the dimensions from (224,224,3) to (1, 224, 224, 3)
	x = np.expand_dims(x, axis=0)
	# img=img/255.0
	# img=img-[0.485, 0.456, 0.406]
	# img=img/[0.229, 0.224, 0.225]
	x = preprocess_input(x)
	images = np.vstack([x])

	#Predicting the result
	print(1)
	y_out = model.predict(images)
	print(2)
	print (y_out)
	print(f"length of output results : {len(y_out[0])}")
	values = dict()
	for i in range(len(dict_)):
		values[list(dict_.keys())[i]]=y_out[0][i]

	max_key = max(values, key=lambda k: values[k])
	# if values[max_key] < 0.1:
	# 		max_key = "empty"
	# print("max", max_key)
	return max_key

def tflite_acc(interpreter, input_details, output_details, image):
	# filepath_tflite = 'tfmodel_normal.tflite'
	# interpreter_tf = tf.lite.Interpreter(model_path=filepath_tflite)
	# interpreter_tf.allocate_tensors()

	# # Get input and output tensors.
	# input_details_tf = interpreter_tf.get_input_details()
	# output_details_tf = interpreter_tf.get_output_details()
	# input_shape_tf = input_details_tf[0]['shape']
	image = np.expand_dims((image), axis=0).astype(np.float32)
	image = preprocess_input(image)
	image = np.asarray(image)
	input_data = np.array(image)
	interpreter.set_tensor(input_details_tf[0]['index'], input_data)

	interpreter.invoke()
	y_out = interpreter.get_tensor(output_details[0]['index'])
	print(f"length of output results : {len(y_out[0])}")
	values = dict()
	# print (y_out[0])
	for i in range(len(dict_)):
		values[list(dict_.keys())[i]]=y_out[0][i]
	

	max_key = max(values, key=lambda k: values[k])
	
	return max_key

def tflite_quant_acc(interpreter, input_details, output_details, image):
	# filepath = 'tfmodel_quantized.tflite'
	# interpreter = tf.lite.Interpreter(model_path=filepath)
	# interpreter.allocate_tensors()

	# # Get input and output tensors.
	# input_details = interpreter.get_input_details()
	# output_details = interpreter.get_output_details()
	# input_shape = input_details[0]['shape']
	image = np.expand_dims((image), axis=0).astype(np.float32)
	image = np.asarray(image)
	input_data = np.array(image)
	interpreter.set_tensor(input_details[0]['index'], input_data)

	interpreter.invoke()
	y_out = interpreter.get_tensor(output_details[0]['index'])
	print(f"length of output results : {len(y_out[0])}")
	values = dict()
	for i in range(len(dict_)):
		values[list(dict_.keys())[i]]=y_out[0][i]
	# values['book_cover'] = y_out[0][0]
	# values['book_page'] = y_out[0][1]
	# values['book_page_graphics'] = y_out[0][2]
	# values['jrkitfaces'] = y_out[0][3]
	# values['jrkitfaces_box'] = y_out[0][4]
	# values['jrkitsticks'] = y_out[0][5]
	# values['jrkitsticks_box'] = y_out[0][6]
	# values['tangram'] = y_out[0][7]
	# values['tangram_box'] = y_out[0][8]
	# values['worksheet_cover'] = y_out[0][9]
	# values['worksheet_page'] = y_out[0][10]
	max_key = max(values, key=lambda k: values[k])
	#print("max", max_key)
	if values[max_key] < 0.1:
			max_key = "empty"
	return max_key

def load_graph(frozen_graph_filename):
	# We load the protobuf file from the disk and parse it to retrieve the 
	# unserialized graph_def
	with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())

	# Then, we import the graph_def into a new Graph and returns it 
	with tf.Graph().as_default() as graph:
		# The name var will prefix every op/nodes in your graph
		# Since we load everything in a new graph, this is not needed
		tf.import_graph_def(graph_def, name="prefix")
	return graph

def pb_acc(graph, image):
	# We can verify that we can access the list of operations in the graph
	#for op in graph.get_operations():
	    #print(op.name)
	    # prefix/Placeholder/inputs_placeholder
	    # ...
	    # prefix/Accuracy/predictions
	    
	# We access the input and output nodes 
	x = graph.get_tensor_by_name('prefix/input_1:0')
	y = graph.get_tensor_by_name('prefix/dense_3/Softmax:0')
	#image = image * 1./255
	image = np.expand_dims((image), axis=0).astype(np.float32)
	image = np.asarray(image)
	with tf.Session(graph=graph) as sess:
		# Note: we don't nee to initialize/restore anything
		# There is no Variables in this graph, only hardcoded constants 
		y_out = sess.run(y, feed_dict={
		    x: image
		})
		# I taught a neural net to recognise when a sum of numbers is bigger than 45
		# it should return False in this case
		#print(y_out) # [[ False ]] Yay, it works!
		print(f"length of output results : {len(y_out[0])}")
		values = dict()
		for i in range(len(dict_)):
			values[list(dict_.keys())[i]]=y_out[0][i]
		# values['book_cover'] = y_out[0][0]
		# values['book_page'] = y_out[0][1]
		# values['book_page_graphics'] = y_out[0][2]
		# values['jrkitfaces'] = y_out[0][3]
		# values['jrkitfaces_box'] = y_out[0][4]
		# values['jrkitsticks'] = y_out[0][5]
		# values['jrkitsticks_box'] = y_out[0][6]
		# values['tangram'] = y_out[0][7]
		# values['tangram_box'] = y_out[0][8]
		# values['worksheet_cover'] = y_out[0][9]
		# values['worksheet_page'] = y_out[0][10]
		
		max_key = max(values, key=lambda k: values[k])
		if values[max_key] < 0.1:
			max_key = "empty"	
		print(max_key)
	return max_key

def acc(true_detections, h5_detections, h5_detections_temp, pb_detections, tflite_detections, tflite_quant_detections):
	count_h5, count_h5_temp, count_pb, count_tf, count_tf_q, total = 0, 0, 0, 0, 0, 0

	#for y, y_h5 ''', y_h5_temp, y_pb, y_tf, y_tf_q''' in zip(true_detections, h5_detections)#, h5_detections_temp, pb_detections, tflite_detections, tflite_quant_detections):
	for y, y_h5 in zip(true_detections, h5_detections):
		search = y
		for sublist in combined_classes:
		    if search in sublist:
		    	index_y=combined_classes.index(sublist)
		    	break
		search=y_h5
		for sublist in combined_classes:
		    if search in sublist:
		    	index_y_h5=combined_classes.index(sublist)
		    	break
		# search=y_h5_temp
		# for sublist in combined_classes:
		#     if search in sublist:
		#     	index_y_h5_temp=combined_classes.index(sublist)
		#     	break
		# search=y_pb
		# for sublist in combined_classes:
		#     if search in sublist:
		#     	index_y_pb=combined_classes.index(sublist)
		#     	break
		# search=y_tf
		# for sublist in combined_classes:
		#     if search in sublist:
		#     	index_y_tf=combined_classes.index(sublist)
		#     	break
		# search=y_tf_q
		# for sublist in combined_classes:
		#     if search in sublist:
		#     	index_y_tf_q=combined_classes.index(sublist)
		#     	break
		if index_y==index_y_h5:
			count_h5+=1
		# if index_y==index_y_h5_temp:
		# 	count_h5_temp+=1
		# if index_y==index_y_pb:
		# 	count_pb+=1
		# if index_y==index_y_tf:
		# 	count_tf+=1
		# if index_y==index_y_tf_q:
		# 	count_tf_q+=1
		# if y == "book_cover":
		# 	if y_h5 == "book_cover":
		# 		count_h5 += 1
		# 	if y_h5_temp == "book_cover":
		# 		count_h5_temp += 1
		# 	if y_pb == "book_cover":
		# 		count_pb += 1
		# 	if y_tf == "book_cover":
		# 		count_tf += 1
		# 	if y_tf_q == "book_cover":
		# 		count_tf_q += 1
			
		# elif y == "book_page":
		# 	if y_h5 == "book_page":
		# 		count_h5 += 1
		# 	if y_h5_temp == "book_page":
		# 		count_h5_temp += 1	
		# 	if y_pb == "book_page":
		# 		count_pb += 1
		# 	if y_tf == "book_page":
		# 		count_tf += 1
		# 	if y_tf_q == "book_page":
		# 		count_tf_q += 1
			
		# elif y == "jrkitsticks" or y == "jrkitsticks_box":
		# 	if y_h5 == "jrkitsticks" or y_h5 == "jrkitsticks_box":
		# 		count_h5 += 1
		# 	if y_h5_temp == "jrkitsticks" or y_h5_temp == "jrkitsticks_box" :
		# 		count_h5_temp += 1
		# 	if y_pb == "jrkitsticks" or y_pb == "jrkitsticks_box":
		# 		count_pb += 1
		# 	if y_tf == "jrkitsticks" or y_tf == "jrkitsticks_box":
		# 		count_tf += 1
		# 	if y_tf_q == "jrkitsticks" or y_tf_q == "jrkitsticks_box":
		# 		count_tf_q += 1
			
		# elif y == "jrkitfaces" or y == "jrkitfaces_box":
		# 	if y_h5 == "jrkitfaces" or y_h5 == "jrkitfaces_box":
		# 		count_h5 += 1
		# 	if y_h5_temp == "jrkitfaces" or y_h5_temp == "jrkitfaces_box":
		# 		count_h5_temp += 1
		# 	if y_pb == "jrkitfaces" or y_pb == "jrkitfaces_box":
		# 		count_pb += 1
		# 	if y_tf == "jrkitfaces" or y_tf == "jrkitfaces_box":
		# 		count_tf += 1
		# 	if y_tf_q == "jrkitfaces" or y_tf_q == "jrkitfaces_box":
		# 		count_tf_q += 1
			
		# elif y == "tangram" or y == "tangram_box":
		# 	if y_h5 == "tangram" or y_h5 == "tangram_box":
		# 		count_h5 += 1
		# 	if y_h5_temp == "tangram" or y_h5_temp == "tangram_box" :
		# 		count_h5_temp += 1
		# 	if y_pb == "tangram" or y_pb == "tangram_box":
		# 		count_pb += 1
		# 	if y_tf == "tangram" or y_tf == "tangram_box":
		# 		count_tf += 1
		# 	if y_tf_q == "tangram" or y_tf_q == "tangram_box":
		# 		count_tf_q += 1
			
		# elif y == "worksheet_page" or y == "worksheet_cover":
		# 	if y_h5 == "worksheet_page" or y_h5 == "worksheet_cover":
		# 		count_h5 += 1
		# 	if y_h5_temp == "worksheet_page" or y_h5_temp == "worksheet_cover":
		# 		count_h5_temp += 1
		# 	if y_pb == "worksheet_page" or y_pb == "worksheet_cover":
		# 		count_pb += 1
		# 	if y_tf == "worksheet_page" or y_tf == "worksheet_cover":
		# 		count_tf += 1
		# 	if y_tf_q == "worksheet_page" or y_tf_q == "worksheet_cover":
		# 		count_tf_q += 1
			
		# elif y == "book_page_graphics":
		# 	if y_h5 == "book_page_graphics":
		# 		count_h5 += 1
		# 	if y_h5_temp == "book_page_graphics":
		# 		count_h5_temp += 1
		# 	if y_pb == "book_page_graphics":
		# 		count_pb += 1
		# 	if y_tf == "book_page_graphics":
		# 		count_tf += 1
		# 	if y_tf_q == "book_page_graphics":
		# 		count_tf_q += 1

		# elif y == "comic_book":
		# 	if y_h5 == "comic_book":
		# 		count_h5 += 1
		# 	if y_h5_temp == "comic_book":
		# 		count_h5_temp += 1
		# 	if y_pb == "comic_book":
		# 		count_pb += 1
		# 	if y_tf == "comic_book":
		# 		count_tf += 1
		# 	if y_tf_q == "comic_book":
		# 		count_tf_q += 1
		
		# elif y == "empty":
		# 	if y_h5 == "empty":
		# 		count_h5 += 1
		# 	if y_h5_temp == "empty":
		# 		count_h5_temp += 1
		# 	if y_pb == "empty":
		# 		count_pb += 1
		# 	if y_tf == "empty":
		# 		count_tf += 1
		# 	if y_tf_q == "empty":
		# 		count_tf_q += 1

		# elif y == "workbook":
		# 	if y_h5 == "workbook":
		# 		count_h5 += 1
		# 	if y_h5_temp == "workbook":
		# 		count_h5_temp += 1
		# 	if y_pb == "workbook":
		# 		count_pb += 1
		# 	if y_tf == "workbook":
		# 		count_tf += 1
		# 	if y_tf_q == "workbook":
		# 		count_tf_q += 1

		# elif y == "notebook":
		# 	if y_h5 == "notebook":
		# 		count_h5 += 1
		# 	if y_h5_temp == "notebook":
		# 		count_h5_temp += 1
		# 	if y_pb == "notebook":
		# 		count_pb += 1
		# 	if y_tf == "notebook":
		# 		count_tf += 1
		# 	if y_tf_q == "notebook":
		# 		count_tf_q += 1

		total += 1
	print("=================After combining the results based on user view=================")
	print(f"Accuracy score for h5 : {count_h5/total}")
	print(f"Accuracy score for h5_temperature : {count_h5_temp/total}")
	print(f"Accuracy score for pb : {count_pb/total}")
	print(f"Accuracy score for tf : {count_tf/total}")
	print(f"Accuracy score for tf_quant : {count_tf_q/total}")
	print("================================================================================")

def acc_only_h5(true_detections, h5_detections):
	count_h5 ,total = 0, 0

	#for y, y_h5 ''', y_h5_temp, y_pb, y_tf, y_tf_q''' in zip(true_detections, h5_detections)#, h5_detections_temp, pb_detections, tflite_detections, tflite_quant_detections):
	for y, y_h5 in zip(true_detections, h5_detections):
		search = y
		for sublist in combined_classes:
		    if search in sublist:
		    	index_y=combined_classes.index(sublist)
		    	break
		search=y_h5
		for sublist in combined_classes:
		    if search in sublist:
		    	index_y_h5=combined_classes.index(sublist)
		    	break

		if index_y==index_y_h5:
			count_h5+=1


		total += 1
	print("=================After combining the results based on user view=================")
	print(f"Accuracy score for h5 : {count_h5/total}")
shape = (224, 224)
# img_path, tf_detection, pb_detection, tf_quant_detection, h5_detection, h5_detection_temp, end_tf - start_tf, end_pb - start_pb, end_tf_quant - start_tf_quant, end_h5 - start_h5, end_h5_temp - start_h5_temp)
  
with open(csv_path+"/acc_comparison.csv", "w") as csvfile:
	writer = csv.writer(csvfile)
	writer.writerow(("image name", "h5 detection", "h5 time"))

# {'book_cover': 0, 'book_page': 1, 'book_page_graphics': 2, 'comic_book': 3, 'empty': 4, 'jrkitfaces': 5,
# 'jrkitfaces_box': 6, 'jrkitsticks': 7, 'jrkitsticks_box': 8, 'notebook': 9, 'tangram': 10,
# 'tangram_box': 11, 'workbook': 12, 'worksheet_cover': 13, 'worksheet_page': 14}

ap = argparse.ArgumentParser()
ap.add_argument("-fo", "--folder", required = True, help = "path to the regression dataset")
ap.add_argument("-f", "--flag", required = True, default = 0, help = "1 for separate classes or 0 for combined")
args = vars(ap.parse_args())
#dict_={'Bookcover': 0, 'Bookpagetext': 1, 'CostumeParty': 2, 'JrkitABC': 3, 'Newton': 4, 'SquigleMagic': 5, 'Tangram': 6, 'WorksheetPage': 7}

dict_={
	   'book_cover': 0,
	   'book_page': 1, 
	   'book_page_graphics': 2, 
	   'comic_book': 3, 
	   'empty': 4, 
	   'jrkitfaces': 5, 
	   'jrkitfaces_box': 6, 
	   'jrkitsticks': 7,
	   'jrkitsticks_box': 8, 
	   'notebook': 9,
	   'tangram': 10, 
	   'tangram_box': 11, 
	   'workbook': 12,
	   'worksheet_cover': 13,
	   'worksheet_page': 14
	   }
dict_c={
	   'book_cover': 0,
	   'book_page': 1, 
	   'book_page_graphics': 2, 
	   # 'comic_book': 3, 
	   'empty': 3, 
	   'jrkitfaces': 4, 
	   # 'jrkitfaces_box': 6, 
	   'jrkitsticks': 5,
	   # 'jrkitsticks_box': 8, 
	   # 'notebook': 9,
	   'tangram': 6, 
	   # 'tangram_box': 11, 
	   'workbook': 7,
	   'worksheet_cover': 8,
	   'worksheet_page': 9
	   }


combined_classes=[
                ['book_page'],
                ['comic_book','notebook','book_page_graphics'],
                ['book_cover'],
                ['jrkitfaces_box','jrkitfaces'],
                ['jrkitsticks_box','jrkitsticks'],
                ['tangram','tangram_box'],
                ['workbook'],
                ['worksheet_page'],
                ['worksheet_cover'],
                ['empty']
                ]



# print ("--------- ",combined_classes.index("jrkitsticks"))
# for i in range(len(dict_)):
# 	print ("bblllaahhh = ",list(dict_.keys())[i])
# dict_={0 : 'book_cover', 1 : 'book_page', 2 : 'book_page_graphics', 3 : 'jrkitfaces', 4 : 'jrkitsticks', 5 : 'tangram', 6 : 'worksheet_page'}
images = list()
tflite_detections = list()
# tflite_quant_detections = list()
h5_detections = list()
# h5_detections_temp = list()
# pb_detections = list()
time_taken_tf = list()
# time_taken_tf_quant = list()
# time_taken_pb = list()
time_taken_h5 = list()
# time_taken_h5_temp = list()


folder_location=args["folder"]
print(folder_location)
all_folders=os.listdir(folder_location)
print(all_folders)
total=0
correct=0
y_true=list()
y_pred=list()
labels=[]
labels_names=[]#['book_cover', 'book_page', 'book_page_graphics', 'jrkitfaces', 'jrkitfaces_box', 'jrkitsticks', 'jrkitsticks_box', 'tangram', 'tangram_box', 'worksheet_cover', 'worksheet_page']
#loading h5 model
model = load_model(h5_path)
# model.update({'swish_activation': Activation(swish_activation)})

#for temperature scaling
# last_layer = model.layers.pAs stated before, the problem arises when a large input space is mapped to a small one, causing the derivatives to disappear. In Image 1, this is most clearly seen at when |x| is big. Batch normalization reduces this problem by simply normalizing the input so |x| doesn’t reach the outer edges of thop()
# last_layer.activation = keras.activations.linear
# i = model.input
# o = last_layer(model.layers[-1].output)
# model_1 = Model(inputs=i, outputs=[o])


#loading tflite model
# filepath_tflite = 'model_testing/separate_classes_empty_removed_book_graphics_added/tfmodel_normal.tflite'
# interpreter_tf = tf.lite.Interpreter(model_path=tflite_path)
# interpreter_tf.allocate_tensors()
# # Get input and output tensors.
# input_details_tf = interpreter_tf.get_input_details()
# output_details_tf = interpreter_tf.get_output_details()
# input_shape_tf = input_details_tf[0]['shape']

# # loading tflite quantized model
# # filepath_tflite_q = 'model_testing/separate_classes_empty_removed_book_graphics_added/tfmodel_quantized.tflite'
# interpreter_tf_q = tf.lite.Interpreter(model_path=quantised_path)
# interpreter_tf_q.allocate_tensors()
# # Get input and output tensors.
# input_details_tf_q = interpreter_tf_q.get_input_details()
# output_details_tf_q = interpreter_tf_q.get_output_details()
# input_shape_tf_q = input_details_tf_q[0]['shape']

# #loading pb model
# graph = load_graph(frozen_graph_path)

for folder in all_folders:
	print(folder)
	labels_names.append(folder)
	#print('labels are',labels)
	files=[]
	files=os.listdir(os.path.join(folder_location,folder))
	print("folder", folder)
	
	#image = cv2.resize(cv2.imread('/home/user/srinivas/dummy/worksheet/'+items), shape)
	for file in files:
		y_true.append(folder)
		img_path = os.path.join(folder_location,folder,file)
		print(img_path)
		image=cv2.imread(img_path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		image = cv2.resize(image, shape)
		# image=image/255.0
		# image=image-[0.485, 0.456, 0.406]
		# image=image/[0.229, 0.224, 0.225]
		#tflite detection
		# start_tf = time.time()
		# tf_detection = tflite_acc(interpreter_tf, input_details_tf, output_details_tf, image)
		# end_tf = time.time()

		# pb detection
		# start_pb = time.time()
		# pb_detection = pb_acc(graph, image)
		# end_pb = time.time()
		
		# # tflite quant detection
		# start_tf_quant = time.time()
		# tf_quant_detection = tflite_quant_acc(interpreter_tf_q, input_details_tf_q, output_details_tf_q, image)
		# end_tf_quant = time.time()

		# h5 detections
		start_h5 = time.time()
		h5_detection = h5_acc(model, image)
		end_h5 = time.time()

		# start_h5_temp = time.time()
		# h5_detection_temp = h5_acc_temp(model_1, image)
		# end_h5_temp = time.time()

		# tflite_quant_detections.append(tf_quant_detection)
		# time_taken_tf_quant.append(end_tf_quant - start_tf_quant)
		
		# tflite_detections.append(tf_detection)
		# time_taken_tf.append(end_tf - start_tf)

		# time_taken_pb.append(end_pb-start_pb)
		# pb_detections.append(pb_detection)

		time_taken_h5.append(end_h5 - start_h5)
		h5_detections.append(h5_detection)

		# time_taken_h5_temp.append(end_h5_temp - start_h5_temp)
		# h5_detections_temp.append(h5_detection_temp)

		images.append(file)
		write_to_csv_h5_tf(img_path, h5_detection, end_h5 - start_h5,)
		# write_to_csv(img_path, tf_detection, pb_detection, tf_quant_detection, h5_detection, h5_detection_temp, end_tf - start_tf, end_pb - start_pb, end_tf_quant - start_tf_quant, end_h5 - start_h5, end_h5_temp - start_h5_temp)
  
# Printing the time

# print("time taken for pb: ", statistics.mean(time_taken_pb))
# print("time taken for tf: ", statistics.mean(time_taken_tf))
# print("time taken for tf_quant: ", statistics.mean(time_taken_tf_quant))
print("time taken for h5: ", statistics.mean(time_taken_h5))
# print("time taken for h5 with temperature scaling : ", statistics.mean(time_taken_h5_temp))

'''
c_pb=confusion_matrix(y_true, pb_detections)
print ('Accuracy Score for pb:',accuracy_score(y_true, pb_detections))
print(c_pb)
'''
#print("TRUE FOR TFLITE", tflite_detections)

# final_dict_tf = {}
# final_dict_pb = {}
# final_dict_tf_quant = {}
final_dict_h5 = {}
# final_dict_h5_temp = {}
# c_pb = confusion_matrix(y_true, pb_detections,labels_names)
# c_tflite = confusion_matrix(y_true, tflite_detections,labels_names)
# c_tflite_quant = confusion_matrix(y_true, tflite_quant_detections,labels_names)
# print("+++++++++Y_TRUE+++++++", y_true)
# print("------------LABEL NAMES------------", labels_names)
# print("............................h5 detections................", h5_detections)
c_h5 = confusion_matrix(y_true, h5_detections, labels_names)
# c_h5_temp = confusion_matrix(y_true, h5_detections_temp, labels_names)
i=0
serial_num=[]
for val in dict_.values():
    serial_num.append(all_folders[i])
    # final_dict_tf[all_folders[i]]=c_tflite[:,i]
    # final_dict_pb[all_folders[i]]=c_pb[:,i]
    # final_dict_tf_quant[all_folders[i]]=c_tflite_quant[:, i]
    final_dict_h5[all_folders[i]]=c_h5[:,i]
    # final_dict_h5_temp[all_folders[i]] = c_h5_temp[:, i]
    i+=1

df_1=pd.DataFrame({'labels':labels_names})

# df_pb=pd.DataFrame(final_dict_pb)
# dft_pb=df_1.join(df_pb)

# df_tf = pd.DataFrame(final_dict_tf)
# dft_tf = df_1.join(df_tf)

# df_tf_quant = pd.DataFrame(final_dict_tf_quant)
# dft_tf_quant = df_1.join(df_tf_quant)

df_h5 = pd.DataFrame(final_dict_h5)
dft_h5 = df_1.join(df_h5)

# df_h5_temp = pd.DataFrame(final_dict_h5_temp)
# dft_h5_temp = df_1.join(df_h5_temp)

# dft_tf.to_csv(csv_path+'/confusion_matrix_tf.csv')
# dft_pb.to_csv(csv_path+'/confusion_matrix_pb.csv')
# dft_tf_quant.to_csv(csv_path+"/confusion_matrix_tf_quant.csv")
dft_h5.to_csv(csv_path+"/confusion_matrix_h5.csv")
# dft_h5_temp.to_csv(csv_path+"/confusion_matrix_h5_temp.csv")
print("=============Accuracy score without combining the classes based on user perspective===================")
# print("Accuracy score for tflite :", accuracy_score(y_true, tflite_detections))
# print("Accuracy Score for pb :", accuracy_score(y_true, pb_detections))
# print("Accuracy Score for tflite quant :", accuracy_score(y_true, tflite_quant_detections))
print("Accuracy Score for h5 :", accuracy_score(y_true, h5_detections))
# print("Accuracy Score for h5 with Temperature Scaling : ", accuracy_score(y_true, h5_detections_temp))
print("==========================================================================================")

if args["flag"]:
# 	#acc(y_true, h5_detections, h5_detections_temp, pb_detections, tflite_detections, tflite_quant_detections)
	acc_only_h5(y_true, h5_detections)

#print("confusion", c_tflite)
#print ('Accuracy is :',(correct*100.0)/total)
#write_to_csv(images, pb_detections, tflite_detections, time_taken_pb, time_taken_tf, accuracy_score(y_true, pb_detections), accuracy_score(y_true, tflite_detections))