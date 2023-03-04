"""Miscellaneous utility functions."""

from functools import reduce

from PIL import Image
import os


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
    """resize image with unchanged aspect ratio using padding"""
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def custom_sorted(dict_box_b_3, image_predict_b_2):
    width, height = image_predict_b_2.size
    print(f"Width: {width}, Height: {height}")
    print("dict_box_original:", dict_box_b_3)

    # Sort box
    boxes_list_sorted = sorted(dict_box_b_3, key=lambda box: (box[1], box[0]))
    print('boxes_list_sorted_1:', boxes_list_sorted)
    boxes_each_line_sorted = []
    temp_list = []
    for idx, item in enumerate(boxes_list_sorted):
        if boxes_list_sorted[idx][1] - boxes_list_sorted[idx - 1][1] < height / 10:
            temp_list.append(item)
        else:
            temp_list = sorted(temp_list, key=lambda box: box[0])
            boxes_each_line_sorted.extend(temp_list)
            temp_list = [item]

        if idx == len(boxes_list_sorted) - 1:
            temp_list = sorted(temp_list, key=lambda box: box[0])
            boxes_each_line_sorted.extend(temp_list)
    print('boxes_list_sorted_2:', boxes_each_line_sorted)
    return boxes_each_line_sorted
