import glob
import os

from PIL import Image

from CRNN_b_4.main import CRNN
from keras_yolo_b_1.yolo import License_plate_Detector_b_1
from keras_yolo_b_1.yolo3.utils import crop_add_black_box, create_essential_folders
from keras_yolo_b_2.yolo import License_plate_Detector_b_2
from keras_yolo_b_2.yolo3.utils import total_process, add_black_padding_b_2
from keras_yolo_b_3.yolo import License_plate_Detector_b_3
from keras_yolo_b_3.yolo3.utils import custom_sorted


class Process_license_plate:
    def __init__(self, license_plate_detector_b_1, license_plate_detector_b_2, license_plate_detector_b_3):  # ham init
        self.yolo_b_1 = license_plate_detector_b_1()
        self.yolo_b_2 = license_plate_detector_b_2()
        self.yolo_b_3 = license_plate_detector_b_3()

    def predict_b1(self, image, image_name):  # dinh nghia buoc 1
        dict_box_b_1 = self.yolo_b_1.detect_image(image)  # nhan dang buoc 1
        list_image_predict_b_1 = []  # tao list chua anh rong
        if dict_box_b_1 is None:  # neu box chua ROI rong
            image.save(os.path.join(wrong_b_1, image_name))  # thi luu vao file du doan sai (wrong_b_1)
        else:  # con lai
            list_image_predict_b_1 = crop_add_black_box(image,
                                                        dict_box_b_1)  # cat vung ROI ra va them padding den, sau do them cac anh do vao 1 list
        return list_image_predict_b_1  # tra list anh ve

    def predict_b_2(self, image_predict_b_1, new_name_b_1):  # dinh nghia buoc 2

        image_predict_b_1.save(os.path.join(result_b_1, new_name_b_1))  # luu anh o buoc 1 va dat ten moi

        dict_box_b_2 = self.yolo_b_2.detect_image(image_predict_b_1)  # nhan dang buoc 2
        image_predict_b_2 = total_process(image_predict_b_1,
                                          dict_box_b_2)  # hau xu ly b2 (tien xu ly b3) - xu ly cac truong hop mat goc, trung goc
        if image_predict_b_2 is None:  # neu sau khi xu ly, box chua ROI rong
            image_predict_b_1.save(os.path.join(wrong_b_2, new_name_b_1))  # thi luu vao file du doan sai (wrong_b_2)
        else:
            image_predict_b_2 = add_black_padding_b_2(image_predict_b_2)  # them padding den
        return image_predict_b_2  # tra lai anh buoc 2

    def predict_b3_b4(self, image_predict_b_2, new_name_b_2): # dinh nghia buoc 3 va buoc 4
        image_predict_b_2.save(os.path.join(result_b_2, new_name_b_2))  # luu anh buoc 2
        # print('new_name_b_2', new_name_b_2)
        dict_box_b_3 = self.yolo_b_3.detect_image(image_predict_b_2)  # nhan dang anh buoc 3

        if dict_box_b_3 is None:  # neu box chua ROI rong
            image_predict_b_2.save(os.path.join(wrong_b_3_4, new_name_b_2))  # thi luu vao file du doan sai (wrong_b_3)
        else:
            # print('dict_box_b_3:', dict_box_b_3['text'])
            detector = CRNN()  # dinh nghia detector
            dict_box_b_3 = custom_sorted(dict_box_b_3, image_predict_b_2)
            final_predict = ''

            for index, coor in enumerate(
                    dict_box_b_3['text']):  # chay qua tung ROI tren anh sau khi da nhan dang duoc o buoc 3
                image_copy = image_predict_b_2.copy()  # copy anh sau buoc 2
                image_predict_b_3 = image_copy.crop((int(coor[0]), int(coor[1]), int(coor[2]),
                                                     int(coor[3])))  # thuc hien buwoc 3-cat vung ROI tren anh ra
                result = detector.predict_OCR_b_4(image_predict_b_3)  # nhan dang OCR buoc 4
                # print('result', result)
                final_predict = '-' + final_predict + result # them nhan vao ket qua cuoi cung tra ve
                new_name_b_3 = f'{new_name_b_2.split(".jpg")[0]}_{result}.jpg'  # tao 1 ten moi cho rung vung ROI voi ket qua buoc 4 ben trong
                save_path = os.path.join(result_b_3_4, new_name_b_3)  # luu anh lai voi ten moi
                image_predict_b_3.save(save_path)  # luu anh lai voi ten moi
            return final_predict

    def process(self, image, image_path): # ket noi cac buoc 1, 2 ,3 ,4
        image_name = os.path.basename(image_path) # lay ten anh
        list_image_predict_b_1 = self.predict_b1(image, image_name) #nhan ve ket qua buoc 1
        for index, image_predict_b_1 in enumerate(list_image_predict_b_1): # chay qua tung anh trong list anh buoc 1 tra ve
            new_name_b_1 = f'{image_name.split(".jpg")[0]}_{index + 1}.jpg' # tao ten moi
            image_predict_b_2 = self.predict_b_2(image_predict_b_1, new_name_b_1) # nhan ket qua tra ve buoc 2
            new_name_b_2 = f'{new_name_b_1.split(".jpg")[0]}_2.jpg' # tao ten moi
            final_predict = self.predict_b3_b4(image_predict_b_2, new_name_b_2) # nhan ket qua tra ve buoc 3, 4
            final_name = f'{image_name}_{final_predict}' # tao ten moi (ten anh goc + ten nhan)
            image.save(os.path.join(final_result, final_name))
        os.rename(image_path, os.path.join(processed, image_name))


if __name__ == '__main__':
    folder_path = 'test_area/input'  # path chua anh va luu anh
    image_paths = glob.glob(os.path.join(folder_path, '*.jpg'))  # path chua anh
    output_b_1, result_b_1, wrong_b_1, output_b_2, wrong_b_2, result_b_2, output_b_3_4, wrong_b_3_4, result_b_3_4, final_result, processed = create_essential_folders(
        folder_path)  # tao cac thu muc can thiet de luu ket qua
    process = Process_license_plate(License_plate_Detector_b_1, License_plate_Detector_b_2,
                                    License_plate_Detector_b_3)  # gan cac tham so vao class Process_license_plate da tao

    for image_path_ in image_paths[:1]:  # chay qua tung anh mot
        print(image_path_)  # in ra path anh
        image_ = Image.open(image_path_)  # mo anh
        process.process(image_, image_path_)  # thuc hien xu ly anh

    print('Done')
