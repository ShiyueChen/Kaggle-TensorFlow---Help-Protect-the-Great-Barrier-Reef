import sys
sys.path.append('../input/yolox-tf2')
import os
import  cv2

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model

from nets.yolo import yolo_body
from utils.utils import cvtColor, preprocess_input, resize_image
from utils.utils_bbox import DecodeBox
from PIL import Image

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

class YOLO(object):
    _defaults = {
        "model_path"        : '../input/yolox-tf2/ep083-loss3.039-val_loss3.097.h5',
        "classes_path"      : '../input/yolox-tf2/model_data/cots_classes.txt',
        "input_shape"       : [6400, 6400],
        "phi"               : 's',
        "confidence"        : 0.1,
        "nms_iou"           : 0.3,
        "max_boxes"         : 100,
        "letterbox_image"   : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        self.class_names, self.num_classes = ['cots'], 1

        self.generate()


    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        self.yolo_model = yolo_body([None, None, 3], num_classes = self.num_classes, phi = self.phi)
        self.yolo_model.load_weights(self.model_path)
        print('{} model, and classes loaded.'.format(model_path))

        self.input_image_shape = Input([2,],batch_size=1)
        inputs  = [*self.yolo_model.output, self.input_image_shape]
        outputs = Lambda(
            DecodeBox,
            output_shape = (1,),
            name = 'yolo_eval',
            arguments = {
                'num_classes'       : self.num_classes,
                'input_shape'       : self.input_shape,
                'confidence'        : self.confidence,
                'nms_iou'           : self.nms_iou,
                'max_boxes'         : self.max_boxes,
                'letterbox_image'   : self.letterbox_image
             }
        )(inputs)
        self.yolo_model = Model([self.yolo_model.input, self.input_image_shape], outputs)

    @tf.function
    def get_pred(self, image_data, input_image_shape):
        out_boxes, out_scores, out_classes = self.yolo_model([image_data, input_image_shape], training=False)
        return out_boxes, out_scores, out_classes

    def detect_image(self, image):
        image       = cvtColor(image)

        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)

        image_data  = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)

        input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
        out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape)
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        res = ''
        for i, c in list(enumerate(out_classes)):
            box             = out_boxes[i]
            score           = out_scores[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            res += '{:.2f}'.format(score) + ' ' + str(left) + ' ' + str(top) + ' ' + str(right-left) + ' ' + str(bottom-top) + ' '
            print(res)
        return res


import greatbarrierreef
env = greatbarrierreef.make_env()   # initialize the environment
iter_test = env.iter_test()

yolo = YOLO()
for (pixel_array, sample_prediction_df) in iter_test:
    image = Image.fromarray(cv2.cvtColor(pixel_array,cv2.COLOR_BGR2RGB))
    r_image = yolo.detect_image(image).strip()
    sample_prediction_df['annotations'] = r_image
    env.predict(sample_prediction_df)