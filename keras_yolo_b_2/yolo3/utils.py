"""Miscellaneous utility functions."""

from functools import reduce
from PIL import Image
import numpy as np
import math
import cv2


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def letterbox_image(image, size):
    """
    resize image with unchanged aspect ratio using padding
    """
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image


def find_middle_point(xmin, ymin, xmax, ymax):
    x_middle = (xmin + xmax)/2
    y_middle = (ymin + ymax)/2
    return [x_middle, y_middle]


def angle2pt(a, b):
    angle = math.degrees(math.atan2(a[1] - b[1], a[0] - b[0]))
    return angle


def process_wrong_corners(image, dict_box):
    width, height = image.size

    for label in list(dict_box.keys()):  # use list here to avoid the del command below, if not use will yield the error: "RuntimeError: dictionary changed size during iteration"
        list_boxes = list(dict_box[label])
        num_boxes = len(list_boxes)
        # print(f'{label} has {num_boxes}')

        if num_boxes > 1:
            del dict_box[label]

            if label == 'top_left':
                for index, value in enumerate(list_boxes):  # list boxes cua nhung boxes trung
                    middle_point = find_middle_point(value[0], value[1], value[2], value[3])
                    if middle_point[0] < 1.1*width/2 and middle_point[1] < 1.1*height/2 and (value[0] > width/10 and value[1] > height/10):
                        dict_box[label] = [value]

            elif label == 'top_right':
                for index, value in enumerate(list_boxes):
                    middle_point = find_middle_point(value[0], value[1], value[2], value[3])
                    if middle_point[0] > 0.9*width/2 and middle_point[1] < 1.1*height/2 and (value[2] < 9*width/10 and value[1] > height/10):
                        dict_box[label] = [value]

            elif label == 'bottom_right':
                for index, value in enumerate(list_boxes):
                    middle_point = find_middle_point(value[0], value[1], value[2], value[3])
                    if middle_point[0] > 0.9*width/2 and middle_point[1] > 0.9*height/2 and (value[2] < 9*width/10 and value[3] < 9*height/10):
                        dict_box[label] = [value]

            elif label == 'bottom_left':
                for index, value in enumerate(list_boxes):
                    middle_point = find_middle_point(value[0], value[1], value[2], value[3])
                    if middle_point[0] < 1.1*width/2 and middle_point[1] > 0.9*height/2 and (value[0] > width/10 and value[3] < 9*height/10):
                        dict_box[label] = [value]

    return dict_box


def process_1_corner(image, dict_box):
    label = list(dict_box.keys())[0]
    xmin = list(dict_box[label])[0][0]
    ymin = list(dict_box[label])[0][1]
    xmax = list(dict_box[label])[0][2]
    ymax = list(dict_box[label])[0][3]
    middle_point = find_middle_point(xmin, ymin, xmax, ymax)

    width, height = image.size
    top_left_state = False
    top_right_state = False
    bottom_right_state = False
    bottom_left_state = False
    if middle_point[0] < width/2 and middle_point[1] < height/2:
        top_left_state = True
    elif middle_point[0] > width/2 and middle_point[1] < height/2:
        top_right_state = True
    elif middle_point[0] > width/2 and middle_point[1] > height/2:
        bottom_right_state = True
    elif middle_point[0] < width/2 and middle_point[1] > height/2:
        bottom_left_state = True

    if label == 'top_left':
        if top_right_state is True:
            rotated_angle = Image.ROTATE_90
            image = image.transpose(rotated_angle)

        elif bottom_right_state is True:
            rotated_angle = Image.ROTATE_180
            image = image.transpose(rotated_angle)

        elif bottom_left_state is True:
            rotated_angle = Image.ROTATE_270
            image = image.transpose(rotated_angle)

    elif label == 'top_right':
        if top_left_state is True:
            rotated_angle = Image.ROTATE_270
            image = image.transpose(rotated_angle)

        if bottom_right_state is True:
            rotated_angle = Image.ROTATE_90
            image = image.transpose(rotated_angle)

        if bottom_left_state is True:
            rotated_angle = Image.ROTATE_180
            image = image.transpose(rotated_angle)

    elif label == 'bottom_right':
        if top_left_state is True:
            rotated_angle = Image.ROTATE_180
            image = image.transpose(rotated_angle)

        if top_right_state is True:
            rotated_angle = Image.ROTATE_270
            image = image.transpose(rotated_angle)

        if bottom_left_state is True:
            rotated_angle = Image.ROTATE_90
            image = image.transpose(rotated_angle)

    elif label == 'bottom_left':
        if top_left_state is True:
            rotated_angle = Image.ROTATE_90
            image = image.transpose(rotated_angle)

        if top_right_state is True:
            rotated_angle = Image.ROTATE_180
            image = image.transpose(rotated_angle)

        if bottom_right_state is True:
            rotated_angle = Image.ROTATE_270
            image = image.transpose(rotated_angle)
    return image


def find_middle_point_label(dict_box):
    middle_point_dict = {}

    for label in list(dict_box.keys()):
        xmin = list(dict_box[label])[0][0]
        ymin = list(dict_box[label])[0][1]
        xmax = list(dict_box[label])[0][2]
        ymax = list(dict_box[label])[0][3]
        x_middle = (xmin + xmax) / 2
        y_middle = (ymin + ymax) / 2
        middle_point_dict.setdefault(label, []).append([x_middle, y_middle])
    return middle_point_dict


def process_2_corners(image, dict_box):
    middle_point_dict = find_middle_point_label(dict_box)
    labels = list(middle_point_dict.keys())
    if 'top_left' in labels and 'top_right' in labels:
        angle_to_rotate = angle2pt((middle_point_dict['top_right'][0][0], middle_point_dict['top_right'][0][1]),
                                   (middle_point_dict['top_left'][0][0], middle_point_dict['top_left'][0][1]))
        image = image.rotate(angle_to_rotate - 90)

    if 'top_right' in labels and 'bottom_right' in labels:
        angle_to_rotate = angle2pt(
            (middle_point_dict['bottom_right'][0][0], middle_point_dict['bottom_right'][0][1]),
            (middle_point_dict['top_right'][0][0], middle_point_dict['top_right'][0][1]))
        image = image.rotate(angle_to_rotate - 90)

    if 'bottom_right' in labels and 'bottom_left' in labels:
        angle_to_rotate = angle2pt(
            (middle_point_dict['bottom_right'][0][0], middle_point_dict['bottom_right'][0][1]),
            (middle_point_dict['bottom_left'][0][0], middle_point_dict['bottom_left'][0][1]))
        image = image.rotate(angle_to_rotate)

    if 'bottom_left' in labels and 'top_left' in labels:
        angle_to_rotate = angle2pt((middle_point_dict['bottom_left'][0][0], middle_point_dict['bottom_left'][0][1]),
                                   (middle_point_dict['top_left'][0][0], middle_point_dict['top_left'][0][1]))
        image = image.rotate(angle_to_rotate - 90)
    return image


def process_3_corners(dict_box):
    if 'top_left' not in dict_box.keys():
        midpoint = np.add(dict_box['top_right'], dict_box['bottom_left'])
        xmin = midpoint[0][0] - dict_box['bottom_right'][0][0]
        ymin = midpoint[0][1] - dict_box['bottom_right'][0][1]
        xmax = midpoint[0][2] - dict_box['bottom_right'][0][2]
        ymax = midpoint[0][3] - dict_box['bottom_right'][0][3]
        dict_box.setdefault('top_left', []).append([xmin, ymin, xmax, ymax])

    elif 'top_right' not in dict_box.keys():
        midpoint = np.add(dict_box['bottom_right'], dict_box['top_left'])
        xmin = midpoint[0][0] - dict_box['bottom_left'][0][0]
        ymin = midpoint[0][1] - dict_box['bottom_left'][0][1]
        xmax = midpoint[0][2] - dict_box['bottom_left'][0][2]
        ymax = midpoint[0][3] - dict_box['bottom_left'][0][3]
        dict_box.setdefault('top_right', []).append([xmin, ymin, xmax, ymax])

    elif 'bottom_right' not in dict_box.keys():
        midpoint = np.add(dict_box['bottom_left'], dict_box['top_right'])
        xmin = midpoint[0][0] - dict_box['top_left'][0][0]
        ymin = midpoint[0][1] - dict_box['top_left'][0][1]
        xmax = midpoint[0][2] - dict_box['top_left'][0][2]
        ymax = midpoint[0][3] - dict_box['top_left'][0][3]
        dict_box.setdefault('bottom_right', []).append([xmin, ymin, xmax, ymax])

    elif 'bottom_left' not in dict_box.keys():
        midpoint = np.add(dict_box['top_left'], dict_box['bottom_right'])
        xmin = midpoint[0][0] - dict_box['top_right'][0][0]
        ymin = midpoint[0][1] - dict_box['top_right'][0][1]
        xmax = midpoint[0][2] - dict_box['top_right'][0][2]
        ymax = midpoint[0][3] - dict_box['top_right'][0][3]
        dict_box.setdefault('bottom_left', []).append([xmin, ymin, xmax, ymax])
    return dict_box


def perspective_transform(image, dict_box):
    # PIL to openCV
    open_cv_image = np.array(image)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    # All points are in format [x, y]
    top_left = find_middle_point(dict_box['top_left'][0][0], dict_box['top_left'][0][1], dict_box['top_left'][0][2], dict_box['top_left'][0][3])  # tL
    top_right = find_middle_point(dict_box['top_right'][0][0], dict_box['top_right'][0][1], dict_box['top_right'][0][2], dict_box['top_right'][0][3])  # tR
    bottom_right = find_middle_point(dict_box['bottom_right'][0][0], dict_box['bottom_right'][0][1], dict_box['bottom_right'][0][2], dict_box['bottom_right'][0][3])  # bR
    bottom_left = find_middle_point(dict_box['bottom_left'][0][0], dict_box['bottom_left'][0][1], dict_box['bottom_left'][0][2], dict_box['bottom_left'][0][3])  # bL

    width_AB = np.sqrt(((top_left[0] - top_right[0]) ** 2) + ((top_left[1] - top_right[1]) ** 2))
    width_DC = np.sqrt(((bottom_left[0] - bottom_right[0]) ** 2) + ((bottom_left[1] - bottom_right[1]) ** 2))
    maxWidth = max(int(width_AB), int(width_DC))

    height_AD = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))
    height_BC = np.sqrt(((bottom_right[0] - top_right[0]) ** 2) + ((bottom_right[1] - top_right[1]) ** 2))
    maxHeight = max(int(height_AD), int(height_BC))

    input_pts = np.float32([top_left, bottom_left, bottom_right, top_right])
    output_pts = np.float32([[0, 0],
                            [0, maxHeight - 1],
                            [maxWidth - 1, maxHeight - 1],
                            [maxWidth - 1, 0]])

    # Compute the perspective transform matrix M
    M = cv2.getPerspectiveTransform(input_pts, output_pts)
    # print(M.astype(int))
    out = cv2.warpPerspective(open_cv_image, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)
    # out = cv2.resize(out, (416, 416))
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(out)
    return img_pil


def total_process(image, dict_box):
    duplicated_box = False
    box_numbers = 0
    # print('original dict box:', dict_box)
    if dict_box is None:
        final_image = image
        return final_image

    for key in dict_box:  # loop through all the keys
        value = dict_box[key]  # get value for the key
        value_num = len(value)
        box_numbers += value_num
        if value_num > 1:
            duplicated_box = True

    if box_numbers > 4 or duplicated_box is True:
        dict_box = process_wrong_corners(image, dict_box)
    # print('after process dict box:', dict_box)

    box_numbers = 0
    for key in dict_box:  # loop through all the keys
        value = dict_box[key]  # get value for the key
        value_num = len(value)
        box_numbers += value_num

    if box_numbers == 3 or box_numbers == 4:
        if box_numbers == 3:
            dict_box = process_3_corners(dict_box)
        final_image = perspective_transform(image, dict_box)

    elif box_numbers == 2:
        final_image = process_2_corners(image, dict_box)

    elif box_numbers == 1:
        final_image = process_1_corner(image, dict_box)
    else:
        final_image = None
    return final_image

def add_black_padding_b_2(image):
    width_box, height_box = image.size
    black_border = max(round(width_box * 1.2 - width_box), round(height_box * 1.2 - height_box))
    background = Image.new('RGB', (int(width_box + black_border), int(height_box + black_border)), 'black')
    bg_w, bg_h = background.size
    offset = ((bg_w - width_box) // 2, (bg_h - height_box) // 2)

    background.paste(image, offset)
    return background

def add_black_padding_and_list(image):
    list_image_final = []
    image_final = add_black_padding_b_2(image)
    return list_image_final.append(image_final)