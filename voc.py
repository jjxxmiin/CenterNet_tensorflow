# pascal voc dataset parsing
import tensorflow as tf
import os
import numpy as np
import xml.etree.ElementTree as Et
import matplotlib.pyplot as plt
import time

def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(bytes_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def float_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(bytes_list=tf.train.FloatList(value=values))


def PascalVOC(path,labels):
    dataset_dir = path
    result = []
    #writer = tf.python_io.TFRecordWriter('./pascal_voc.tfrecord')
    xml_dir = os.path.join(dataset_dir,'Annotations')
    img_dir = os.path.join(dataset_dir,'JPEGImages')

    xml_path_list = [os.path.join(xml_dir,xml_name) for xml_name in os.listdir(xml_dir)]

    start = time.time()

    print('PascalVOC XML parsing start')

    for xml_path in xml_path_list:
        ground_truth = []

        xml = open(xml_path,'r')
        tree = Et.parse(xml)
        root = tree.getroot()

        # =========== image name ===========

        img_name = root.find("filename").text
        #print("filename : {}".format(img_name))

        img_path = os.path.join(img_dir,img_name)
        img = plt.imread(img_path)
        #img = tf.gfile.GFile(img_path, 'rb').read()
        # =========== image size ===========

        size = root.find("size")

        width = int(size.find("width").text)
        height = int(size.find("height").text)
        channels = int(size.find("depth").text)

        #print("width : {} height : {} channels : {}".format(width, height, channels))
        shape = np.array([width,height,channels])

        # =========== image bounding box ===========

        objects = root.findall("object")

        for object in objects:
            name = labels.index(object.find("name").text)
            bndbox = object.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            ground_truth.append([name,xmin,ymin,xmax,ymax])

        ground_truth =np.array(ground_truth)

        features = {
            'image': img,
            'shape': shape,
            'ground_truth':ground_truth
        }

        result.append(features)

        end = time.time()

        print('PascalVOC XML parsing end\nPascalVOC XML  parsing time : ',end-start)

        return result

'''
result = PascalVOC('G:\\dataset\\VOC2007')
print(result[0].keys())
print(result[0]['shape'])



        # =========== TFRecord ===========

        features = {
            'image': bytes_feature(img),
            'shape': bytes_feature(shape.tobytes()),
            'ground_truth': bytes_feature(ground_truth.tobytes())
        }

        example = tf.train.Example(features=tf.train.Features(
            feature=features))


        writer.write(example.SerializeToString())

    writer.close()


filenames="./pascal_voc.tfrecord",

def decode(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'shape': tf.FixedLenFeature([], tf.string),
            'gt': tf.FixedLenFeature([], tf.string),
        })
    image = tf.decode_raw(features['image'], tf.float32)
    image = tf.reshape(tensor=image, shape=[28, 28, 1])
    shape = tf.decode_raw(features['shape'], tf.uint8)
    gt = tf.decode_raw(features['gt'], tf.uint8)

    # image = tf.cast(image, tf.int32)
    # image = tf.reshape(image, shape)
    return image, shape, gt


train_dataset = tf.data.TFRecordDataset(filenames)
train_dataset = train_dataset.map(decode)
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.batch(batch_size = 2)

val_iterator = train_dataset.make_initializable_iterator()
x, y,z = val_iterator.get_next()

with tf.Session() as sess:
    sess.run(val_iterator.initializer)
    X_data = sess.run(x)

print(X_data.shape)
'''