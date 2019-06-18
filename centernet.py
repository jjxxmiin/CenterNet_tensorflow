import tensorflow as tf
import numpy as np

def normalization(image):
    mean = [0.408, 0.447, 0.470]
    std = [0.289, 0.274, 0.278]

    return (np.float32(image / 255.) - mean) / std

class batch_norm(object):
    def __init__(self,name, epsilon=1e-5, momentum=0.9):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope = self.name # if tensorflow vesrion < 1.4, delete this line
                                            )

class ResNet18():
    def __init__(self):

        self.layer = [2,2,2,2]

        #self.parameter = parameter
        self.batch_size = 10
        #self.lr = parameter.lr

    def weight_init(self,filter,in_ch,out_ch):
        return tf.Variable(tf.truncated_normal([filter,filter,in_ch,out_ch], stddev=0.02))

    def residual_block(self,x,in_ch,out_ch,stride,name):
        with tf.variable_scope(name) as scope:
            print('Building residual : ',scope.name)

            if in_ch == out_ch:
                if stride == 1:
                    shortcut = tf.identity(x)
                else:
                    shortcut = tf.nn.max_pool(x,[1,stride,stride,1],[1,stride,stride,1],'VALID')
            else:
                shortcut = tf.nn.conv2d(x,self.weight_init(1,in_ch,out_ch),strides=[1,stride,stride,1],padding='SAME')

            x = tf.nn.conv2d(x, self.weight_init(3,in_ch,out_ch), strides=[1, stride, stride, 1], padding='SAME',name='conv_1')
            x = batch_norm(name='bn1')(x)
            x = tf.nn.relu(x,name='relu1')

            in_ch = x.get_shape().as_list()[-1]

            x = tf.nn.conv2d(x, self.weight_init(3,in_ch,out_ch), strides=[1, 1, 1, 1], padding='SAME',name='conv_2')
            x = batch_norm(name='bn2')(x)

            x = x + shortcut
            x = tf.nn.relu(x,name='relu2')

        return x

    def Architecture(self,input):

        ###############################
        # resnet-18 architecture      #
        # filter = [64,64,128,256,512]#
        # kernel = [7,3,3,3,3]        #
        # stride = [2,0,2,2,2]        #
        ###############################

        with tf.variable_scope('layer1'):
            layer1 = tf.nn.conv2d(input,self.weight_init(7,3,64),strides=[1,2,2,1],padding='SAME')
            layer1 = tf.nn.relu(batch_norm(name='bn1')(layer1))
            layer1 = tf.nn.max_pool(layer1,[1,3,3,1],[1,2,2,1],'SAME')


        layer2 = self.residual_block(layer1, 64, 64, 1, name='layer2_1')
        layer2 = self.residual_block(layer2, 64, 64, 1, name='layer2_2')


        layer3 = self.residual_block(layer2, 64, 128, 2, name='layer3_1')
        layer3 = self.residual_block(layer3, 128, 128, 1, name='layer3_2')

        layer4 = self.residual_block(layer3, 128, 256, 2, name='layer4_1')
        layer4 = self.residual_block(layer4, 256, 256, 1, name='layer4_2')

        layer5 = self.residual_block(layer4, 256, 512, 2, name='layer5_1')
        layer5 = self.residual_block(layer5, 512, 512, 1, name='layer5_2')

        return layer5

