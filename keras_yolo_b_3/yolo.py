import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np
from keras import backend
from keras.models import load_model
from keras_yolo_b_3.yolo3.utils import letterbox_image
from keras_yolo_b_3.yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from keras.layers import Input


class License_plate_Detector_b_3:
    def __init__(self, model_num=22500, image_input_size=480, score_value=0.1):
        self.yolo_model = None
        self.model_path = '/mnt/4fead833-5e60-4aaa-8d32-c3cb6f2ad6f3/home/ted/PycharmProjects/License_plate/weights/weights_b_3/yolov3_' + str(
            model_num) + '_weights.h5'
        self.anchors_path = '/mnt/4fead833-5e60-4aaa-8d32-c3cb6f2ad6f3/home/ted/PycharmProjects/License_plate/keras_yolo_b_3/model_data/yolo_anchors.txt'
        self.classes_path = '/mnt/4fead833-5e60-4aaa-8d32-c3cb6f2ad6f3/home/ted/PycharmProjects/License_plate/keras_yolo_b_3/model_data/coco_classes.txt'
        self.score = score_value
        self.iou = 0.2  # representing the threshold for deciding whether boxes overlap too much with respect to IOU.
        # > threshold = delete. High = too many boxes
        self.model_image_size = (image_input_size, image_input_size)

        self.anchors = self._get_anchors()
        self.class_names = self._get_class()
        self.sess = backend.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = self.classes_path
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]  # remove spaces at the beginning or the end of the string
        return class_names

    def _get_anchors(self):
        anchors_path = self.anchors_path
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == num_anchors / len(self.yolo_model.output) * (
                    num_classes + 5), 'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = backend.placeholder(shape=(2,))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors, len(self.class_names),
                                           self.input_image_shape, score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))  # PIL(w,h,c),np(h,w,c)
        else:
            new_image_size = (image.width - (image.width % 32), image.height - (image.height % 32))
            boxed_image = letterbox_image(image, tuple(reversed(new_image_size)))

        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.  # Normalize
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run([self.boxes, self.scores, self.classes],
                                                           feed_dict={
                                                               self.yolo_model.input: image_data,
                                                               self.input_image_shape: [image.size[1], image.size[0]],
                                                               backend.learning_phase(): 0
                                                           })
        # dict_results = {label: [] for label in self.class_names}
        dict_results = {}
        for i, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[c]
            ymin, xmin, ymax, xmax = out_boxes[i]

            ymin = max(0, np.floor(ymin + 0.5))
            xmin = max(0, np.floor(xmin + 0.5))
            ymax = min(image.size[1], np.floor(ymax + 0.5))
            xmax = min(image.size[0], np.floor(xmax + 0.5))
            # dict_results[predicted_class].append([xmin, ymin, xmax, ymax])
            dict_results.setdefault(predicted_class, []).append([xmin, ymin, xmax, ymax])

        for value in list(dict_results.values()):
            if len(value) != 0:
                return dict_results
        return None
