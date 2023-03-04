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
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image


def add_black_padding_b_1(image):
    width_box, height_box = image.size
    black_border = max(round(width_box * 1.2 - width_box), round(height_box * 1.2 - height_box))
    background = Image.new('RGB', (int(width_box + black_border), int(height_box + black_border)), 'black')
    bg_w, bg_h = background.size
    offset = ((bg_w - width_box) // 2, (bg_h - height_box) // 2)

    background.paste(image, offset)
    return background


def crop_add_black_box(image, dict_box):
    image_width, image_height = image.width, image.height
    list_image_final = []
    ratio = 0.1  # ratio to add black box
    for key in list(dict_box.keys()):
        list_boxes = dict_box[key]

        for index, box in enumerate(list_boxes):
            xmin, ymin, xmax, ymax = box

            width_box, height_box = xmax - xmin, ymax - ymin

            if xmin - width_box * ratio >= 0:
                xmin -= width_box * ratio
            else:
                xmin = 0

            if ymin - height_box * ratio >= 0:
                ymin -= height_box * ratio
            else:
                ymin = 0

            if xmax + width_box * ratio <= image_width:
                xmax += width_box * ratio
            else:
                xmax = image_width

            if ymax + height_box * ratio <= image_height:
                ymax += height_box * ratio
            else:
                ymax = image_height

            image_1 = image.crop((xmin, ymin, xmax, ymax))
            background = add_black_padding_b_1(image_1)
            list_image_final.append(background)
    return list_image_final


def create_essential_folders(folder_path):

    output_b_1 = os.path.join(folder_path, 'output_b_1')
    if not os.path.exists(output_b_1):
        os.mkdir(output_b_1)

    wrong_b_1 = os.path.join(output_b_1, 'wrong_b_1')
    if not os.path.exists(wrong_b_1):
        os.mkdir(wrong_b_1)

    result_b_1 = os.path.join(output_b_1, 'result_b_1')
    if not os.path.exists(result_b_1):
        os.mkdir(result_b_1)

    output_b_2 = os.path.join(folder_path, 'output_b_2')
    if not os.path.exists(output_b_2):
        os.mkdir(output_b_2)

    wrong_b_2 = os.path.join(output_b_2, 'wrong_b_2')
    if not os.path.exists(wrong_b_2):
        os.mkdir(wrong_b_2)

    result_b_2 = os.path.join(output_b_2, 'result_b_2')
    if not os.path.exists(result_b_2):
        os.mkdir(result_b_2)

    output_b_3_4 = os.path.join(folder_path, 'output_b_3_4')
    if not os.path.exists(output_b_3_4):
        os.mkdir(output_b_3_4)

    wrong_b_3_4 = os.path.join(output_b_3_4, 'wrong_b_3_4')
    if not os.path.exists(wrong_b_3_4):
        os.mkdir(wrong_b_3_4)

    result_b_3_4 = os.path.join(output_b_3_4, 'result_b_3_4')
    if not os.path.exists(result_b_3_4):
        os.mkdir(result_b_3_4)

    processed = os.path.join(folder_path, 'processed')
    if not os.path.exists(processed):
        os.mkdir(processed)

    final_result = os.path.join(folder_path, 'final_result')
    if not os.path.exists(final_result):
        os.mkdir(final_result)

    return output_b_1, result_b_1, wrong_b_1, output_b_2, wrong_b_2, result_b_2, output_b_3_4, wrong_b_3_4, result_b_3_4, final_result, processed
