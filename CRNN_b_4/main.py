from CRNN_b_4.predict import Predict

model_path = 'weights/weights_b_4/CRNN_140.pth'  # lua chon weight
detect = Predict(model_path)


class CRNN:
    def __int__(self, image_path):
        self.image_path = image_path

    def predict_OCR_b_4(self, image):
        # print('image_path:', image_path)
        image = image.convert('L')  # chuyen anh sang single-channel mode (grayscale)
        predict_result = detect.predict(image)  # du doan ky tu tren anh
        return predict_result
