__author__ = 'shekkizh'
# Utils used with tensorflow implemetation
import tensorflow as tf
import numpy as np
import scipy.misc as misc
import os, sys
from six.moves import urllib
import tarfile
import zipfile
from tqdm import trange
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from mpl_toolkits.axes_grid1 import ImageGrid
import PIL

def maybe_download_and_extract(dir_path, url_name, is_tarfile=False, is_zipfile=False):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    filename = url_name.split('/')[-1]
    filepath = os.path.join(dir_path, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write(
                '\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(url_name, filepath, reporthook=_progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
        if is_tarfile:
            tarfile.open(filepath, 'r:gz').extractall(dir_path)
        elif is_zipfile:
            with zipfile.ZipFile(filepath) as zf:
                zip_dir = zf.namelist()[0]
                zf.extractall(dir_path)


def save_image(image, image_size, save_dir, name=""):
    """
    Save image by unprocessing assuming mean 127.5
    :param image:
    :param save_dir:
    :param name:
    :return:
    """
    image += 1
    image *= 127.5
    image = np.clip(image, 0, 255).astype(np.uint8)
    image = np.reshape(image, (image_size, image_size, -1))
    misc.imsave(os.path.join(save_dir, name + "pred_image.png"), image)


def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


def weight_variable_xavier_initialized(shape, constant=1, name=None):
    stddev = constant * np.sqrt(2.0 / (shape[2] + shape[3]))
    return weight_variable(shape, stddev=stddev, name=name)


def weight_variable(shape, stddev=0.02, name=None):
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def get_tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)

def fully_connected(x,out_num,name=None):
    W_shape = [x.get_shape().as_list()[-1], out_num]
    W = weight_variable(W_shape, name=name+'_weights')
    bias_shape = [x.get_shape().as_list()[0], out_num]
    bias = bias_variable(bias_shape, name=name+'_bias')
    return tf.matmul(x, W) + bias  

def conv2d_basic(x, W, bias):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
    return tf.nn.bias_add(conv, bias)


def conv2d_strided(x, W, b,strides=2):
    conv = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)

def conv2d(x, out_channels, size=4,strides=2, stddev=0.2, name=None):
    W = weight_variable(shape = [size,size,x.get_shape().as_list()[-1],out_channels],stddev = stddev, name=name)
    b = tf.zeros([out_channels])
    return conv2d_strided(x, W, b, strides)


def conv2d_transpose_strided(x, W, b, output_shape=None):

    if output_shape is None:
        output_shape = x.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] = W.get_shape().as_list()[2]
    # print output_shape
    conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, 2, 2, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)

def deconv(x,out_channels,add_x=0,add_y=0,strides=2,Print=False,name=None):
    kernel_shape = [4,4,out_channels,x.get_shape().as_list()[-1]]
  
    W = weight_variable(shape=kernel_shape, stddev=0.02,name=name)
    output_shape = x.get_shape().as_list()
    output_shape[1] = output_shape[1]*2 + add_x
    output_shape[2] = output_shape[2]*2 + add_y
    output_shape[3] = out_channels
    if Print:
    	print 'in shape', x.get_shape().as_list()
    	print 'out shape',output_shape
    return tf.nn.conv2d_transpose(x,W,output_shape, strides=[1,strides,strides,1],padding="SAME")


def leaky_relu(x, alpha=0.2, name=None):
    return tf.maximum(alpha * x, x, name)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def local_response_norm(x):
    return tf.nn.lrn(x, depth_radius=5, bias=2, alpha=1e-4, beta=0.75)


def batch_norm(x, n_out, phase_train, scope='bn', decay=0.9, eps=1e-5, stddev=0.02):
    """
    Code taken from http://stackoverflow.com/a/34634291/2267819
    """
    with tf.variable_scope(scope):
        beta = tf.get_variable(name='beta', shape=[n_out], initializer=tf.constant_initializer(0.0)
                               , trainable=True)
        gamma = tf.get_variable(name='gamma', shape=[n_out], initializer=tf.random_normal_initializer(1.0, stddev),
                                trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=decay)

        def mean_var_with_update():
            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)
    return normed

def BN(x,trainable=True,name=None):
	if name:
		with tf.variable_scope(name) as scope:
			return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=None, trainable=trainable)

	return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=None, trainable=trainable)


def process_image(image, mean_pixel, norm):
    return (image - mean_pixel) / norm


def unprocess_image(image, mean_pixel, norm):
    return image * norm + mean_pixel


def add_to_regularization_and_summary(var):
    if var is not None:
        tf.summary.histogram(var.op.name, var)
        tf.add_to_collection("reg_loss", tf.nn.l2_loss(var))


def add_activation_summary(var):
    tf.summary.histogram(var.op.name + "/activation", var)
    tf.summary.scalar(var.op.name + "/sparsity", tf.nn.zero_fraction(var))


def add_gradient_summary(grad, var):
    if grad is not None:
        tf.summary.histogram(var.op.name + "/gradient", grad)

def save_imshow_grid(images, logs_dir, filename, shape):
    """
    Plot images in a grid of a given shape.
    """
    fig = plt.figure(1)
    grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.05)

    size = shape[0] * shape[1]
    for i in trange(size, desc="Saving images"):
        grid[i].axis('off')
        grid[i].imshow(images[i])

    plt.savefig(os.path.join(logs_dir, filename))

def TV_loss(image):
    dy = image[:,:-1,...] - image[:,1:, ...]
    dx = image[:,:,:-1, ...] - image[:,:, 1:, ...]
    size_dy = tf.size(dy, out_type = tf.int32)
    size_dx = tf.size(dx, out_type = tf.int32)
    return tf.nn.l2_loss(dy) / tf.to_float(size_dy) + tf.nn.l2_loss(dx) / tf.to_float(size_dx)

def feature_loss(real, fake):
    size = tf.size(real, out_type = tf.int32)
    return tf.nn.l2_loss(real - fake)/tf.to_float(size)

def montage(M,sep=0,canvas_value=0):
  # row X col X H X W X C
  assert M.ndim==5
  canvas=np.ones((M.shape[0]*M.shape[2]+(M.shape[0]-1)*sep,M.shape[1]*M.shape[3]+(M.shape[1]-1)*sep,M.shape[4]),dtype=M.dtype)*canvas_value
  for i in range(M.shape[0]):
    for j in range(M.shape[1]):
      canvas[i*(M.shape[2]+sep):i*(M.shape[2]+sep)+M.shape[2],j*(M.shape[3]+sep):j*(M.shape[3]+sep)+M.shape[3]]=M[i,j]
  return canvas

def write(opath,I,**kwargs):
  '''
  Given a H x W x 3 RGB image it is clipped to the range [0,1] and
  written to an 8-bit image file.
  '''
  img=PIL.Image.fromarray((I*255).clip(0,255).astype(np.uint8))
  ext=os.path.splitext(opath)[1]
  if ext=='.jpg':
    quality=kwargs['quality'] if 'quality' in kwargs else 95
    img.save(opath,quality=quality,optimize=True)
  elif ext=='.png':
    img.save(opath)
  else:
    # I do not want to save unknown extensions because there is no
    # expectation that the default save options are reasonable.
    raise ValueError('Unknown image extension ({})'.format(ext))


def color_match(A,B):
  '''
  A is a rank 5 tensor (column of original images)
  B is a rank 5 tensor (grid of images)
  '''
  A=np.asarray(A)
  B=np.asarray(B)
  print('Computing color match',A.shape,B.shape)
  m=A.reshape(A.shape[0],1,-1).mean(axis=2)
  m=np.expand_dims(np.expand_dims(np.expand_dims(m,-1),-1),-1)
  s=(A-m).reshape(A.shape[0],1,-1).std(axis=2)
  s=np.expand_dims(np.expand_dims(np.expand_dims(s,-1),-1),-1)
  m2=B.reshape(B.shape[0],B.shape[1],-1).mean(axis=2)
  m2=np.expand_dims(np.expand_dims(np.expand_dims(m2,-1),-1),-1)
  s2=(B-m2).reshape(B.shape[0],B.shape[1],-1).std(axis=2)
  s2=np.expand_dims(np.expand_dims(np.expand_dims(s2,-1),-1),-1)
  return (B-m2)*(s+1e-8)/(s2+1e-8)+m


def Residualblock(x, scope=None):
    channels = x.get_shape().as_list()[-1]
    with tf.variable_scope(scope) as scope:
        conv1 = conv2d(x, out_channels=channels, size=4,strides=1, stddev=0.2, name='conv1')
        bn1 = BN(conv1,trainable=True,name='bn1')
        conv2 = conv2d(tf.nn.relu(bn1), out_channels=channels, size=4,strides=1, stddev=0.2, name='conv2')
        bn2 = BN(conv2,trainable=True,name='bn2')
    return x+bn2