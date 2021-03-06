
# coding: utf-8

# In[1]:


# In[2]:
import tensorflow as tf
import numpy as np
import os.path
import argparse
import sys
import time
import random


# In[3]:

HEIGHT = 480
WIDTH = 640

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 795
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 654

# NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10
# NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1439

FLAGS = None
TRAIN_FILE = './data_b/train.tfrecords'
VALIDATION_FILE = './data_b/test.tfrecords'


# In[4]:

def read_NYUdepthV2(filename_queue):
    """Reads and parses examples from NYU_depth data files(train&test.tfrecords).
    
    Args :
        filename_queue : A queue of strings with the filenames to read from.
        
    Returns:
        An object representing a single example with the following fields:
            height,
            width, 
            image_ch : number of color channel in the image
            dpt_image_ch : number of color channel in the dpt_image
            image,
            dpt_image
            
    """
    
    class NYUdepthV2_Record(object):
        pass
    result = NYUdepthV2_Record()
    
    # Dimensions of the images in the NYUdepthV2 dataset.
    result.height = HEIGHT
    result.width = WIDTH
    result.image_ch = 3
    result.dpt_image_ch = 1
    image_bytes = result.height * result.width * result.image_ch # image
    dpt_image_bytes = result.height * result.width * result.dpt_image_ch # depth image

    
    reader = tf.TFRecordReader() # make reader
    _, serialized_example = reader.read(filename_queue)
    
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw' : tf.FixedLenFeature([], tf.string),
            'depth_raw' : tf.FixedLenFeature([], tf.string),
        })
    
    # Convert from a scalar string tensor to a uint8 tesor and a float tensor.
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    dpt_image = tf.decode_raw(features['depth_raw'], tf.float32)
    
    image.set_shape(image_bytes)
    dpt_image.set_shape(dpt_image_bytes)

    
    # reshpae from [channel * width * height] to [channel, width, height]
    temp_img = tf.reshape(
        tf.strided_slice(image, [0], [image_bytes]), [result.image_ch, result.width, result.height])
    temp_dpt_img = tf.reshape(
        tf.strided_slice(dpt_image, [0], [dpt_image_bytes]), 
                         [result.dpt_image_ch, result.width, result.height])
    
    # Convert from [channel, width, height] to [height, width, channel]
    result.image = tf.transpose(temp_img, [2, 1, 0])
    result.dpt_image = tf.transpose(temp_dpt_img, [2, 1, 0])
    
    return result    


# In[5]:

def _generate_image_and_label_batch(image, dpt_image,
                                    min_queue_examples, batch_size, shuffle):
    """Construct a queued batch of images and labels.
    
    Args :
        image : NYUdepthV2_Record instance.image
        dpt_image : NYUdepthV2_Record instance.dpt_image
        binary_im 
        
    Returns:
        images : images 4D tensor of [batch_size, height, width, 3] size.
        dpt_images = depth images 4D tensor of [batch_size, height, width, 1] size.
    """
    # Create a queue that shuffles that examples, and then read 'batch_size' images + dpt_image from the example queue.
    num_preprocess_threads = 8
    if shuffle:
        images, dpt_images = tf.train.shuffle_batch(
            [image, dpt_image],
            batch_size = batch_size,
            num_threads = num_preprocess_threads,
            capacity = min_queue_examples + 3 * batch_size,
            min_after_dequeue = min_queue_examples)
    else:
        images, dpt_images = tf.train.batch(
            [image, dpt_image],
            batch_size = batch_size,
            num_threads = num_preprocess_threads,
            capacity = min_queue_examples + 3 *batch_size)
    
    # Display the training images in the visualizer.
    tf.summary.image('images', images)
    
    return images, dpt_images


# In[10]:

def distorted_inputs(data_dir, batch_size):
    """Construct distorted(&crop)  input for Depth Estimation training using the Reader ops.
    
    Args:
        data_dir : Path to the tfrecord data directory.
        batch_size : Number of images per batch.
        
    Returns:
        images, dpt_images, depth_binary_images (from _generate_image_and_label_batch Function)
    """
    filenames = [os.path.join(data_dir, 'train_' + str(idx) + '.tfrecords') for idx in range(10)]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file : ' + f)
            
    # Create a queue that produces the filenames to read.
    
    filename_queue = tf.train.string_input_producer(filenames)
    
    # Read example from files in the filename queue.
    single_ex = read_NYUdepthV2(filename_queue)
    reshaped_img = tf.cast(single_ex.image, tf.float32)
    reshaped_dpt_img = tf.cast(single_ex.dpt_image, tf.float32)
    
    height = 240
    width = 320
    seed = random.randint(0, 100)
    # Image processing for training the network. 
    
    reshaped_img.set_shape([height*2, width*2, 3])
    reshaped_dpt_img.set_shape([height*2, width*2, 1])

    # 2. Crop A image into half of size
    distorted_image = tf.image.central_crop(central_fraction=0.5, image=reshaped_img)
    distorted_dpt_image = tf.image.central_crop(central_fraction=0.5, image=reshaped_dpt_img)
    # 3. Randomly flip the image horizontally
    tmp = tf.concat([distorted_image, distorted_dpt_image], axis=2)
    tmp = tf.image.random_flip_left_right(tmp)
    # 4. Randomly flip the image vertically
    tmp = tf.image.random_flip_up_down(tmp)
    tmp.set_shape([height, width, 4])
    # 5. Randomly britgtning & contrasting & hue & saturation
    distorted_image = tmp[:, :, 0:3]
    distorted_dpt_image = tmp[:, :, -1]
    distorted_dpt_image = tf.reshape(distorted_dpt_image, [height, width, 1])
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=20)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
    distorted_image = tf.image.random_hue(distorted_image, max_delta=0.4)
    distorted_image = tf.image.random_saturation(distorted_image, lower=0.2, upper=1.8)
    
    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(distorted_image)
    #float_image = distorted_image
    # Set the shapes of tensors.
    float_image.set_shape([height, width, 3])    
    distorted_dpt_image.set_shape([height, width, 1])
    
    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
    print ("Filling queue with %d Images before starting to train." %min_queue_examples, "This will take a few minutes.")
    
    # Generate a batch of images and dpt_images by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, distorted_dpt_image,
                                          min_queue_examples, batch_size, shuffle=True)

def cent_crop_raw(data_dir, batch_size):
    """Construct distorted(&crop)  input for Depth Estimation training using the Reader ops.
    
    Args:
        data_dir : Path to the tfrecord data directory.
        batch_size : Number of images per batch.
        
    Returns:
        images, dpt_images, depth_binary_images (from _generate_image_and_label_batch Function)
    """
    filenames =[os.path.join(data_dir, 'train_raw.tfrecords')]
    #filenames =[os.path.join(data_dir, 'train.tfrecords')]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file : ' + f)
            
    # Create a queue that produces the filenames to read.
    
    filename_queue = tf.train.string_input_producer(filenames)
    
    # Read example from files in the filename queue.
    single_ex = read_NYUdepthV2(filename_queue)
    reshaped_img = tf.cast(single_ex.image, tf.float32)
    reshaped_dpt_img = tf.cast(single_ex.dpt_image, tf.float32)
    
    height = 240
    width = 320
    seed = random.randint(0, 100)
    # Image processing for training the network. 

    # 2. Crop A image into half of size
    distorted_image = tf.image.central_crop(central_fraction=0.5, image=reshaped_img)
    distorted_dpt_image = tf.image.central_crop(central_fraction=0.5, image=reshaped_dpt_img)
    # 3. Randomly flip the image horizontally
    distorted_image = tf.image.random_flip_left_right(distorted_image, seed=seed)
    distorted_dpt_image = tf.image.random_flip_left_right(distorted_dpt_image, seed=seed)
    # 4. Randomly flip the image vertically
    distorted_image = tf.image.random_flip_up_down(distorted_image, seed=seed)
    distorted_dpt_image = tf.image.random_flip_up_down(distorted_dpt_image, seed=seed)
    # 5. Randomly britgtning & contrasting & hue & saturation
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=60)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
    distorted_image = tf.image.random_hue(distorted_image, max_delta=0.4)
    distorted_image = tf.image.random_saturation(distorted_image, lower=0.2, upper=1.8)
    
    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(distorted_image)
    #float_image = distorted_image
    # Set the shapes of tensors.
    float_image.set_shape([height, width, 3])    
    distorted_dpt_image.set_shape([height, width, 1])
    
    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
    print ("Filling queue with %d Images before starting to train." %min_queue_examples, "This will take a few minutes.")
    
    # Generate a batch of images and dpt_images by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, distorted_dpt_image,
                                          min_queue_examples, batch_size, shuffle=True)

def random_crop_raw(data_dir, batch_size):
    """Construct distorted(&crop)  input for Depth Estimation training using the Reader ops.
    
    Args:
        data_dir : Path to the tfrecord data directory.
        batch_size : Number of images per batch.
        
    Returns:
        images, dpt_images, depth_binary_images (from _generate_image_and_label_batch Function)
    """
    filenames =[os.path.join(data_dir, 'train_raw.tfrecords')]
    #filenames =[os.path.join(data_dir, 'train.tfrecords')]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file : ' + f)
            
    # Create a queue that produces the filenames to read.
    
    filename_queue = tf.train.string_input_producer(filenames)
    
    # Read example from files in the filename queue.
    single_ex = read_NYUdepthV2(filename_queue)
    reshaped_img = tf.cast(single_ex.image, tf.float32)
    reshaped_dpt_img = tf.cast(single_ex.dpt_image, tf.float32)
    
    height = 240
    width = 320
    seed = random.randint(0, 100)
    # Image processing for training the network. 

    # 2. Crop A image at random position.
    distorted_image = tf.random_crop(reshaped_img, [height, width, 3], seed=seed)
    distorted_dpt_image = tf.random_crop(reshaped_dpt_img, [height, width, 1], seed=seed)
    # 3. Randomly flip the image horizontally
    distorted_image = tf.image.random_flip_left_right(distorted_image, seed=seed)
    distorted_dpt_image = tf.image.random_flip_left_right(distorted_dpt_image, seed=seed)
    # 4. Randomly flip the image vertically
    distorted_image = tf.image.random_flip_up_down(distorted_image, seed=seed)
    distorted_dpt_image = tf.image.random_flip_up_down(distorted_dpt_image, seed=seed)
    # 5. Randomly britgtning & contrasting & hue & saturation
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=60)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
    distorted_image = tf.image.random_hue(distorted_image, max_delta=0.4)
    distorted_image = tf.image.random_saturation(distorted_image, lower=0.2, upper=1.8)
    
    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(distorted_image)
    #float_image = distorted_image
    # Set the shapes of tensors.
    float_image.set_shape([height, width, 3])    
    distorted_dpt_image.set_shape([height, width, 1])
    
    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
    print ("Filling queue with %d Images before starting to train." %min_queue_examples, "This will take a few minutes.")
    
    # Generate a batch of images and dpt_images by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, distorted_dpt_image,
                                          min_queue_examples, batch_size, shuffle=True)

def resized_raw(data_dir, batch_size):
    """Construct distorted(&crop)  input for Depth Estimation training using the Reader ops.
    
    Args:
        data_dir : Path to the tfrecord data directory.
        batch_size : Number of images per batch.
        
    Returns:
        images, dpt_images, depth_binary_images (from _generate_image_and_label_batch Function)
    """
    filenames =[os.path.join(data_dir, 'train_raw.tfrecords')]
    #filenames =[os.path.join(data_dir, 'train.tfrecords')]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file : ' + f)
            
    # Create a queue that produces the filenames to read.
    
    filename_queue = tf.train.string_input_producer(filenames)
    
    # Read example from files in the filename queue.
    single_ex = read_NYUdepthV2(filename_queue)
    reshaped_img = tf.cast(single_ex.image, tf.float32)
    reshaped_dpt_img = tf.cast(single_ex.dpt_image, tf.float32)
    
    height = 240
    width = 320
    seed = random.randint(0, 100)
    # Image processing for training the network. 
    # 1. Resize images.
    reshaped_img = tf.image.resize_images(reshaped_img, [height, width])
    reshaped_dpt_img = tf.image.resize_images(reshaped_dpt_img, [height, width])
    # 2. Crop A image at random position.
    distorted_image = tf.random_crop(reshaped_img, [height, width, 3], seed=seed)
    distorted_dpt_image = tf.random_crop(reshaped_dpt_img, [height, width, 1], seed=seed)
    # 3. Randomly flip the image horizontally
    distorted_image = tf.image.random_flip_left_right(distorted_image, seed=seed)
    distorted_dpt_image = tf.image.random_flip_left_right(distorted_dpt_image, seed=seed)
    # 4. Randomly flip the image vertically
    distorted_image = tf.image.random_flip_up_down(distorted_image, seed=seed)
    distorted_dpt_image = tf.image.random_flip_up_down(distorted_dpt_image, seed=seed)
    # 5. Randomly britgtning & contrasting & hue & saturation
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=60)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
    distorted_image = tf.image.random_hue(distorted_image, max_delta=0.2)
    distorted_image = tf.image.random_saturation(distorted_image, lower=0.2, upper=1.8)
    
    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(distorted_image)
    #float_image = distorted_image
    # Set the shapes of tensors.
    float_image.set_shape([height, width, 3])    
    distorted_dpt_image.set_shape([height, width, 1])
    
    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
    print ("Filling queue with %d Images before starting to train." %min_queue_examples, "This will take a few minutes.")
    
    # Generate a batch of images and dpt_images by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, distorted_dpt_image,
                                          min_queue_examples, batch_size, shuffle=True)

def test_inputs_resize(data_dir, batch_size):
    """Construct distorted input for Depth Estimation training using the Reader ops.
    
    Args:
        data_dir : Path to the tfrecord data directory.
        batch_size : Number of images per batch.
        
    Returns:
        images, dpt_images (from _generate_image_and_label_batch Function)
    """
    
    filenames =[os.path.join(data_dir, 'test.tfrecords')]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file : ' + f)
            
    # Create a queue that produces the filenames to read.
    
    filename_queue = tf.train.string_input_producer(filenames)
    
    # Read example from files in the filename queue.
    single_ex = read_NYUdepthV2(filename_queue)
    reshaped_img = tf.cast(single_ex.image, tf.float32)
    reshaped_dpt_img = tf.cast(single_ex.dpt_image, tf.float32)

    
    height = 240
    width = 320
    
    # resize
    reshaped_img = tf.image.resize_images(reshaped_img, [height, width])
    reshaped_dpt_img = tf.image.resize_images(reshaped_dpt_img, [height, width])
    
    # Subtract off the mean and divide by the variance of the pixels.
    #float_image = tf.image.per_image_standardization(reshaped_img)
    float_image = reshaped_img
    # Set the shapes of tensors.
    float_image.set_shape([height, width, 3])    
    reshaped_dpt_img.set_shape([height, width, 1])
    
    
    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
    print ("Filling queue with %d Images before starting to train." %min_queue_examples, "This will take a few minutes.")
    
    # Generate a batch of images and dpt_images by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, reshaped_dpt_img,
                                          min_queue_examples, batch_size, shuffle=False)

def test_inputs_cent_crop(data_dir, batch_size):
    """Construct distorted input for Depth Estimation training using the Reader ops.
    
    Args:
        data_dir : Path to the tfrecord data directory.
        batch_size : Number of images per batch.
        
    Returns:
        images, dpt_images (from _generate_image_and_label_batch Function)
    """
    
    filenames =[os.path.join(data_dir, 'test.tfrecords')]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file : ' + f)
            
    # Create a queue that produces the filenames to read.
    
    filename_queue = tf.train.string_input_producer(filenames)
    
    # Read example from files in the filename queue.
    single_ex = read_NYUdepthV2(filename_queue)
    reshaped_img = tf.cast(single_ex.image, tf.float32)
    reshaped_dpt_img = tf.cast(single_ex.dpt_image, tf.float32)

    
    height = 240
    width = 320
    
    # central crop
    reshaped_img = tf.image.central_crop(reshaped_img, 0.5)
    reshaped_dpt_img = tf.image.central_crop(reshaped_dpt_img, 0.5)
    
    
    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(reshaped_img)
    #float_image = reshaped_img
    # Set the shapes of tensors.
    float_image.set_shape([height, width, 3])    
    reshaped_dpt_img.set_shape([height, width, 1])
    
    
    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
    print ("Filling queue with %d Images before starting to train." %min_queue_examples, "This will take a few minutes.")
    
    # Generate a batch of images and dpt_images by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, reshaped_dpt_img,
                                          min_queue_examples, batch_size, shuffle=False)


"""
def distorted_resized_inputs(data_dir, batch_size):
    
    filenames =[os.path.join(data_dir, 'train.tfrecords')]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file : ' + f)
            
    # Create a queue that produces the filenames to read.
    
    filename_queue = tf.train.string_input_producer(filenames)
    
    # Read example from files in the filename queue.
    single_ex = read_NYUdepthV2(filename_queue)
    reshaped_img = tf.cast(single_ex.image, tf.float32)
    reshaped_dpt_img = tf.cast(single_ex.dpt_image, tf.float32)
    
    height = 240
    width = 320
    seed = random.randint(0, 100)
    # Image processing for training the network. 
    # 1. Randomly scailing a [HEIGHT, WIDTH] image into [240, 320] image using resize method(BILINEAR).
    reshaped_img = tf.image.resize_images(reshaped_img, [height, width])
    reshaped_dpt_img = tf.image.resize_images(reshaped_dpt_img, [height, width])
    # 2. Randomly crop a [height, width] section of the image.
    distorted_image = tf.random_crop(reshaped_img, [height, width, 3], seed=seed)
    distorted_dpt_image = tf.random_crop(reshaped_dpt_img, [height, width, 1], seed=seed)
    # 3. Randomly flip the image horizontally
    distorted_image = tf.image.random_flip_left_right(distorted_image, seed=seed)
    distorted_dpt_image = tf.image.random_flip_left_right(distorted_dpt_image, seed=seed)
    # 4. Randomly flip the image vertically
    distorted_image = tf.image.random_flip_up_down(distorted_image, seed=seed)
    distorted_dpt_image = tf.image.random_flip_up_down(distorted_dpt_image, seed=seed)
    # 5. Randomly britgtning & contrasting
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=20)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.6, upper=1.4)
    
    
    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(distorted_image)
    
    # Set the shapes of tensors.
    float_image.set_shape([height, width, 3])    
    distorted_dpt_image.set_shape([height, width, 1])
    
    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
    print ("Filling queue with %d Images before starting to train." %min_queue_examples, "This will take a few minutes.")
    
    # Generate a batch of images and dpt_images by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, distorted_dpt_image,
                                          min_queue_examples, batch_size, shuffle=True)

def read_NYUdepthV2_temp(filename_queue):
    
    class NYUdepthV2_Record(object):
        pass
    result = NYUdepthV2_Record()
    
    # Dimensions of the images in the NYUdepthV2 dataset.
    result.height = HEIGHT
    result.width = WIDTH
    result.image_ch = 3
    result.dpt_image_ch = 1
    image_bytes = result.height * result.width * result.image_ch # image
    dpt_image_bytes = result.height * result.width * result.dpt_image_ch # depth image
    #binary_bytes = 478*638*1
    
    reader = tf.TFRecordReader() # make reader
    _, serialized_example = reader.read(filename_queue)
    
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw' : tf.FixedLenFeature([], tf.string),
            'depth_raw' : tf.FixedLenFeature([], tf.string)
        })
    
    # Convert from a scalar string tensor to a uint8 tesor and a float tensor.
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    dpt_image = tf.decode_raw(features['depth_raw'], tf.float32)

    image.set_shape(image_bytes)
    dpt_image.set_shape(dpt_image_bytes)

    
    # reshpae from [channel * width * height] to [channel, width, height]
    temp_img = tf.reshape(
        tf.strided_slice(image, [0], [image_bytes]), [result.image_ch, result.width, result.height])
    temp_dpt_img = tf.reshape(
        tf.strided_slice(dpt_image, [0], [dpt_image_bytes]), 
                         [result.dpt_image_ch, result.width, result.height])
    
    # Convert from [channel, width, height] to [height, width, channel]
    result.image = tf.transpose(temp_img, [2, 1, 0])
    result.dpt_image = tf.transpose(temp_dpt_img, [2, 1, 0])
    
    return result    



def temp2(image, dpt_image, min_queue_examples, batch_size, shuffle):
    # Create a queue that shuffles that examples, and then read 'batch_size' images + dpt_image from the example queue.
    num_preprocess_threads = 8
    images, dpt_images = tf.train.batch(
        [image, dpt_image],
        batch_size = batch_size,
        num_threads = num_preprocess_threads,
        capacity = min_queue_examples + 3 *batch_size)
    
    return images, dpt_images

def temp(data_dir, batch_size):
    
    filenames =[os.path.join(data_dir, 'test.tfrecords')]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file : ' + f)
            
    # Create a queue that produces the filenames to read.
    
    filename_queue = tf.train.string_input_producer(filenames)
    
    # Read example from files in the filename queue.
    single_ex = read_NYUdepthV2_temp(filename_queue)
    reshaped_img = tf.cast(single_ex.image, tf.uint8)
    reshaped_dpt_img = tf.cast(single_ex.dpt_image, tf.float32)
    
    # Set the shapes of tensors.
    reshaped_img.set_shape([480, 640, 3])    
    reshaped_dpt_img.set_shape([480, 640, 1])
    
    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
    print ("Filling queue with %d Images before starting to train." %min_queue_examples, "This will take a few minutes.")
    
    # Generate a batch of images and dpt_images by building up a queue of examples.
    return temp2(reshaped_img, reshaped_dpt_img,
                                          min_queue_examples, batch_size, shuffle=False)
"""