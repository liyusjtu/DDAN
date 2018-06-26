from __future__ import print_function

__author__ = "yli"

import numpy as np
import tensorflow as tf
from GAN_models import *
import glob
import os
from prepare import get_dataset
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "20", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")

tf.flags.DEFINE_float("learning_rate", "5e-5", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_float("optimizer_param", "0.5", "beta1 for Adam optimizer / decay for RMSProp")
tf.flags.DEFINE_float("iterations", "20000", "No. of iterations to train model")
tf.flags.DEFINE_integer("image_size", "64", "Size of actual images, Size of images to be generated at.")
tf.flags.DEFINE_string("optimizer", "RMSProp", "Optimizer to use for training")
tf.flags.DEFINE_integer("gen_dimension", "16", "dimension of first layer in generator")
tf.flags.DEFINE_string("mode", "train", "train / visualize model")
tf.flags.DEFINE_string("save_path", "acc/", "path")
tf.flags.DEFINE_string("save_path_", "acc1/", "path")
tf.flags.DEFINE_string("data", "none", "none")

def main(argv=None):
    save_path = '/DATA3_DB7/data/yli/datasets/'

    query_path, input_path,query_test,input_test  = get_dataset(save_path+'list_attr_celeba.txt','Eyeglasses')
    _,_,query_test_,_ = get_dataset(save_path+'list_attr_celeba.txt','Eyeglasses')
    #assert len(reading_path) == len(test_path)
    assert query_test == query_test_
    print('logdir is :', FLAGS.logs_dir)
    if not os.path.exists(FLAGS.logs_dir):
        os.makedirs(FLAGS.logs_dir)
    #glass
    glass_mask = np.zeros([100,100,3])
    for i in range(8,93):
        for j in range(35,65):
            glass_mask[j][i][:]=1
    glass_crop = [35,0,30,100]


    if FLAGS.mode == "train":
        print('train')

        model = GAN_model(batch_size=FLAGS.batch_size,learning_rate=FLAGS.learning_rate,data_path=input_path,\
                        query_path=query_path,mask=glass_mask,crop_size=glass_crop)
        model.build_model()

        model.initialize_network(FLAGS.logs_dir)
        model.train_model(max_iterations=int(1 + FLAGS.iterations), pretrain_iterations=0)
    elif FLAGS.mode == "test":
        test_save_path = FLAGS.logs_dir+FLAGS.save_path
        test_save_path_ = FLAGS.logs_dir + FLAGS.save_path_
        print('sampling images')
        if FLAGS.data == 'no-sun':
            input_test = sorted(glob.glob('/DATA3_DB7/data/msli/CycleGAN-tensorflow/datasets/classification_test/none2glasses/test_images_1/*jpg'))
        elif FLAGS.data == 'sun':
	        input_test = sorted(glob.glob('/DATA3_DB7/data/yli/feature_separation/dataset/sun_glass/*jpg'))
            #input_test = sorted(glob.glob('/DATA3_DB7/data/msli/CycleGAN-tensorflow/datasets/classification_test/none2glasses/test_images_1/*jpg'))
        print(len(input_test))

        model = GAN_model(batch_size=FLAGS.batch_size,learning_rate=FLAGS.learning_rate,data_path=input_test,\
                query_path=query_path,mask=glass_mask,crop_size=glass_crop)
        model.build_model()

        model.initialize_network(FLAGS.logs_dir)
        if not os.path.exists(test_save_path):
            os.makedirs(test_save_path)
        if not os.path.exists(test_save_path_):
            os.makedirs(test_save_path_)
        print('test num :', len(input_test))
        print('batch num :',len(input_test)//FLAGS.batch_size)
        for i in range(len(input_test)//FLAGS.batch_size):
            print(i)
            model.save_results(test_save_path,str(i)+'-')
            model.test(test_save_path_,str(i)+'-')



if __name__ == "__main__":
    tf.app.run()
