import torch
import CRNN_b_4.utils
import CRNN_b_4.dataset_v2
import CRNN_b_4.models.crnn_v2
import CRNN_b_4.params


class Predict:
    def __init__(self, model_path):
        self.model_path = model_path
        self.nclass = len(CRNN_b_4.params.alphabet) + 1
        self.model = CRNN_b_4.models.crnn_v2.CRNN(CRNN_b_4.params.imgH, CRNN_b_4.params.nc, self.nclass, CRNN_b_4.params.nh)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.converter = CRNN_b_4.utils.strLabelConverter(CRNN_b_4.params.alphabet)
        self.transformer = CRNN_b_4.dataset_v2.resizeNormalize((128, 32))

    def predict(self, image):
        image = self.transformer(image)
        if torch.cuda.is_available():
            image = image.cuda()
        image = image.view(1, *image.size())  # flattening

        preds = self.model(image)

        _, preds = preds.max(2)
        # flatten, not required
        preds = preds.transpose(1, 0).view(-1)
        preds_size = torch.LongTensor([torch.numel(preds)])  # x.size(0) returns 1st dimension of the tensor (which is the batch size, this should remain the constant), =len(preds)
        # print('preds:',preds)
        # print('preds_size:',preds_size)
        raw_pred = self.converter.decode(preds, preds_size, raw=True)
        sim_pred = self.converter.decode(preds, preds_size, raw=False)
        # print('%-20s => %-20s' % (raw_pred, sim_pred))

        return sim_pred
