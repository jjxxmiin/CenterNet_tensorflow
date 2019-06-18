import tensorflow as tf
import numpy as np
import argparse
import voc
import centernet

PASCAL_LABELS = [
      '__background__', "aeroplane", "bicycle", "bird", "boat",
      "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
      "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
      "train", "tvmonitor"
                ]

COCO_LABELS = [
      '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
      'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
      'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
      'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
      'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
      'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
      'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
      'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
      'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
      'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
      'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
      'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
      'scissors', 'teddy bear', 'hair drier', 'toothbrush'
            ]

def opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir',
                        help="dataset directory please")

    args = parser.parse_args()

    return args


def main():
    p = {
        'input_size' : 512,
        'output_size' : 128,
        'batch_size' : 128,
        'lr' : 5e-4,
        'epoch' : 140
    }

    #args = opts()
    #datasets = voc.PascalVOC('G:\dataset\VOC2007', PASCAL_LABELS)

    #print('image : ',datasets[0]['image'])
    #print('gt : ',datasets[0]['ground_truth'])
    #print('shape : ',datasets[0]['shape'])

    model = centernet.ResNet18()

    X = tf.placeholder(tf.float32,[p['batch_size'],p[''],p[''],p['']])
    Y = tf.placeholder()

    with tf.Session() as sess:
        sess.init()





if __name__ == "__main__":
    main()