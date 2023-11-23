python

class ObjectDetector:
    def __init__(self, weights_path, data_path):
        self.weights_path = weights_path
        self.data_path = data_path
        self.model = None
        self.stride = None
        self.names = None
        self.pt = None
        self.jit = None
        self.onnx = None
        self.engine = None

    def load_model(self):
        device = self.select_device('')
        model = self.DetectMultiBackend(self.weights_path, device=device, dnn=False, data=self.data_path)
        self.model = model
        self.stride, self.names, self.pt, self.jit, self.onnx, self.engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine

    def run(self, img_path):
        img = cv2.imread(img_path)
        cal_detect = []

        device = self.select_device('')
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        im = self.letterbox(img, (640, 640), self.stride, self.pt)[0]
        im = im.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(device)
        im = im.half() if False else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]

        pred = self.model(im, augment=False)
        pred = self.non_max_suppression(pred, 0.35, 0.05, classes=None, agnostic_nms=False, max_det=1000)

        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = self.scale_coords(im.shape[2:], det[:, :4], img.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = f'{names[c]}'
                    lbl = names[int(cls)]
                    cal_detect.append([label, xyxy, float(conf)])
        return cal_detect

    def select_device(self, device=''):
        pass

    def DetectMultiBackend(self, weights, device='', dnn=False, data=None):
        pass

    def letterbox(self, img, new_shape=(640, 640), stride=32, auto=True, scaleFill=False, scaleup=True, color=(114, 114, 114)):
        pass

    def non_max_suppression(self, pred, conf_thres=0.35, iou_thres=0.05, classes=None, agnostic_nms=False, max_det=1000):
        pass

    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        pass


