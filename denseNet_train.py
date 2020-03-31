from keras.layers import Dense, Activation, Flatten, Dropout, GlobalAveragePooling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD,Nadam
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
#from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
# from keras.applications.mobilenet import MobileNet, preprocess_input
# from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.densenet import DenseNet121,preprocess_input
# from  keras.applications.inception_v3 import InceptionV3,preprocess_input
# from keras.applications.resnet_v2 import ResNet50V2,preprocess_input
#from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.models import load_model
import tensorflow as tf
from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.python.tools import optimize_for_inference_lib
import os
# from pelee_net import PeleeNet
# from keras.applications.vgg16 import VGG16, preprocess_input
# from keras.applications.inception_resnet_v2 import InceptionResNetV2,preprocess_input

import efficientnet.keras as efn 

# print (model.summary())

def saving_to_pb():
	output_names = [node.op.name for node in finetune_model.outputs]
	print('output names are',output_names)
	export_dir = '.model_training/'
	graph_filename = save_dir + "training_vgg16.pb"
	sess = K.get_session()
	output_graph_def = tf.graph_util.convert_variables_to_constants(sess,  sess.graph.as_graph_def(),  output_names)
	with tf.gfile.FastGFile(graph_filename, 'wb') as f:
		f.write(output_graph_def.SerializeToString())

	with tf.gfile.GFile(graph_filename, "rb") as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		tf.train.write_graph(graph_def, export_dir, 'training_vgg16.pbtxt', True)

def saving_in_json():
	# serialize model to JSON
	model_json = finetune_model.to_json()
	with open(os.path.join(save_dir,"model.json"), "w") as json_file:
		json_file.write(model_json)

def generate_pbtxt():
	# Read the graph.
	with tf.gfile.FastGFile(os.path.join(save_dir,'frozen_inference_graph_opt.pb'), 'rb') as f:
	    graph_def = tf.GraphDef()
	    graph_def.ParseFromString(f.read())

	# Remove Const nodes.
	for i in reversed(range(len(graph_def.node))):
	    if graph_def.node[i].op == 'Const':
	        del graph_def.node[i]
	    for attr in ['T', 'data_format', 'Tshape', 'N', 'Tidx', 'Tdim',
	                 'use_cudnn_on_gpu', 'Index', 'Tperm', 'is_training',
	                 'Tpaddings']:
	        if attr in graph_def.node[i].attr:
	            del graph_def.node[i].attr[attr]

	# Save as text.
	tf.train.write_graph(graph_def, "", os.path.join(save_dir,"frozen_inference_graph_opt.pbtxt"), as_text=True)


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


def defining_model(base_model, dropout, fc_layers, num_classes):
    # print(base_model.summary())
    
    # print(base_model.summary())

    x = base_model.output
    # print ("baseeee =",x)
    x = GlobalAveragePooling2D()(x)
    # print(x)
    #x = Flatten()(x)
    x = Dense(1024, activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
    x = Dropout(0.7)(x)
    x = Dense(1024, activation='relu')(x) #dense layer 3
    x=Dropout(0.5)(x)
    # x = Dense(128, activation='relu')(x) #dense layer 3

    predictions = Dense(num_classes, activation='softmax')(x) #final layer with softmax activation
    # x=Dense(num_classes)(x)
    # predictions=Activation('softmax')(x)
    finetune_model = Model(inputs = base_model.input, outputs = predictions)
    print ("length of model = ",len(finetune_model.layers))
    for layer in finetune_model.layers[:200]:
    	layer.trainable = False
    print(finetune_model.summary())
	
    return finetune_model

def saving_tflite():
	K.clear_session()
	converter = tf.lite.TFLiteConverter.from_keras_model_file(os.path.join(save_dir,model_name+'.h5'))
	tfmodel = converter.convert()
	open(os.path.join(save_dir,"tfmodel_normal.tflite"), "wb").write(tfmodel)

	converter.optimizations = [tf.lite.Optimize.DEFAULT]
	quantized_model = converter.convert()
	open(os.path.join(save_dir,"tfmodel_quantized.tflite"), "wb").write(quantized_model)


def optimizing(graph_def):
	#graph = 'trying/trying_model.pb'
	# with tf.gfile.FastGFile(graph, 'rb') as f:
	# 	graph_def = tf.GraphDef()
	# 	graph_def.ParseFromString(f.read())
	# 	tf.summary.FileWriter('logs', graph_def)

	inp_node = 'input_1'
	out_node = 'dense_3/Softmax'
	graph_def = optimize_for_inference_lib.optimize_for_inference(graph_def, [inp_node], [out_node], tf.float32.as_datatype_enum)
	graph_def = TransformGraph(graph_def, [inp_node], [out_node], ["sort_by_execution_order"])

	with tf.gfile.FastGFile(os.path.join(save_dir,'frozen_inference_graph_opt.pb'), 'wb') as f:
	    f.write(graph_def.SerializeToString())

# Plot the training and validation loss + accuracy
def plot_training(history):
	acc = history.history['accuracy']
	val_acc = history.history['val_accuracy']
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs = range(len(acc))

	plt.plot(epochs, acc, 'r.', label='acc')
	plt.plot(epochs, val_acc, 'r', label='val_acc')
	plt.title('Training and validation accuracy')
	plt.legend()
	plt.figure()
	plt.plot(epochs, loss, 'r.', label='loss')
	plt.plot(epochs, val_loss, 'r-', label='val_loss')
	plt.title('Training and validation loss')
	plt.legend()
	plt.show()

	plt.savefig(os.path.join(save_dir,'acc_vs_epochs.png'))

height, width = 224, 224

# base_model = MobileNet(weights = 'imagenet', include_top= False, input_shape = (height, width, 3))
# base_model = VGG16(weights = 'imagenet', include_top= False, input_shape = (height, width, 3))
base_model=DenseNet121(include_top=False, weights='imagenet', input_shape=(height,width,3))
# base_model=InceptionV3(include_top=False, weights='imagenet',input_shape=(height,width,3))
# base_model=ResNet50V2(include_top=False,weights='imagenet',input_shape=(height,width,3))
# base_model = efn.EfficientNetB7(weights='imagenet',include_top=False)
#base_model=InceptionResNetV2(include_top=False, weights='imagenet')
# base_model = PeleeNet(input_shape=(224,224,3), use_stem_block=True, n_classes=1000)



save_dir="denseNet"
model_name="densenet_15_classes_with_empty"
# model_name="vgg16-15_class_with_empty"
# model_name="efficientnet_15_class_with_empty"



train_dir = "/home/user/Downloads/scripts/images_dataset/data_auto/train/"#"../training/"
validation_dir = "/home/user/Downloads/scripts/images_dataset/data_auto/val/"#"../validation/"
test_dir = "/home/user/Downloads/scripts/images_dataset/regression/"#"../regression"

Batch_size = 64

train_datagen = ImageDataGenerator(
	# rescale= 1./255,
	# featurewise_center=True,
    # featurewise_std_normalization=True,
    samplewise_center=True,
    samplewise_std_normalization=True,
    rotation_range=90,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
	vertical_flip=True,
    # preprocessing_function = preprocess_input
)

test_datagen = ImageDataGenerator(
	# rescale= 1./255,
	# featurewise_center=True,
    # featurewise_std_normalization=True,
    # featurewise_center=True,
    # featurewise_std_normalization=True,

    # preprocessing_function = preprocess_input
)

train_generator = train_datagen.flow_from_directory("/media/user/DATA1604/d4-data/data_split/train",
	target_size = (height, width),
	batch_size = Batch_size,
	class_mode = 'categorical'
)

labels = (train_generator.class_indices)
# print(labels)
with open(os.path.join(save_dir,"labels.txt"), "w") as f:
	f.write(str(labels))


# validation_generator = train_datagen.flow_from_directory("/home/user/Downloads/scripts/images_dataset_2/data_splitted/val",
# 	target_size =(height, width),
# 	batch_size = Batch_size,
# 	class_mode = 'categorical'
# )
validation_generator = train_datagen.flow_from_directory("/media/user/DATA1604/d4-data/data_split/val",
    target_size =(height, width),
    batch_size = Batch_size,
    class_mode = 'categorical'
)

test_generator = test_datagen.flow_from_directory("/media/user/DATA1604/d4-data/regression",
	target_size = (height, width),
	batch_size = Batch_size,
	class_mode = 'categorical'
)

# class_list=[
#         "non_relevant",
#         "relevant"
#         	        ]
class_list=[
			'book_cover',
			'book_page', 
			'book_page_graphics', 
			'comic_book', 
			'empty', 
			'jrkitfaces', 
			'jrkitfaces_box', 
			'jrkitsticks', 
			'jrkitsticks_box', 
			'notebook', 
			'tangram', 
			'tangram_box', 
			'workbook', 
			'worksheet_cover', 
			'worksheet_page'
			]

# class_list = ["book_page", "book_cover", "worksheet_page", "worksheet_cover", "tangram", "tangram_box","book_page_graphics", "jrkitsticks", "jrkitsticks_box", "jrkitfaces", "jrkitfaces_box"]
FC_LAYERS = [1024, 1024]

dropout = 0.25

finetune_model = defining_model(base_model,
	dropout= dropout,
	fc_layers = FC_LAYERS,
	num_classes= len(class_list))
#print(finetune_model.summary())
#print("==================")
#print(base_model.layers[:-3])






num_epochs = 15
num_train_images = 6555
# num_train_images = 8326

adam = Adam(lr=0.00005, decay=0.000001)
# adam = Adam(lr=0.0003,beta_1=0.9,beta_2=0.95)
# adam = Nadam(lr=0.0003,beta_1=0.9,beta_2=0.95)

#sgd = SGD(lr=0.000001)
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# labels_dict={'non_relevant': 0, 'relevant': 1}
labels_dict={'book_cover': 0, 'book_page': 1, 'book_page_graphics': 2, 'comic_book': 3, 'empty': 4, 'jrkitfaces': 5, 'jrkitfaces_box': 6, 'jrkitsticks': 7, 'jrkitsticks_box': 8, 'notebook': 9, 'tangram': 10, 'tangram_box': 11, 'workbook': 12, 'worksheet_cover': 13, 'worksheet_page': 14}
#saving_to_pb()
print (list(labels_dict.keys()))

finetune_model.compile(adam, loss='categorical_crossentropy', metrics=["accuracy"])
# filepath = "predict one tensor kerasmobilenet_augmented_00003_35/" + "mobilenet_augmented_00003_35.h5"
filepath=os.path.join(save_dir,model_name+".h5")
checkpoint = ModelCheckpoint(filepath, verbose = 1, save_best_only=True)
callback_list = [checkpoint]
history = finetune_model.fit_generator(train_generator, epochs=num_epochs,
	steps_per_epoch = num_train_images//Batch_size,  validation_data = validation_generator,
	validation_steps = validation_generator.samples // Batch_size, shuffle=True, callbacks=callback_list)


K.set_learning_phase(0)
model = load_model(filepath)
frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in finetune_model.outputs])
#tf.train.write_graph(frozen_graph, "trying/", "trying_model.pb", as_text=False)
#tf.train.write_graph(frozen_graph, 'trying/', 'trying_model.pbtxt', as_text=True)
optimizing(frozen_graph)
generate_pbtxt()
saving_in_json()
#print(history)
loss = finetune_model.evaluate_generator(test_generator, steps = 64)

# num_of_test_samples=1943
# Y_pred = model.predict_generator(test_generator, num_of_test_samples // Batch_size+1)
# y_pred = np.argmax(Y_pred, axis=1)
# print('Confusion Matrix')
# print(confusion_matrix(test_generator.classes, y_pred))
# print('Classification Report')
# target_names = ['Cats', 'Dogs', 'Horse']
# print(classification_report(test_generator.classes, y_pred, target_names=list(labels_dict.keys())))

# pred=finetune_model.predict_generator(test_generator,steps=64)
# true=finetune_model.classes
print("Loss Metric: ", loss)


# plot_training(history)
saving_tflite()
