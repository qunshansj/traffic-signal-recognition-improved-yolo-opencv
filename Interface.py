python

class YOLOv5Detector:
    def __init__(self, weights, data, device='', half=False, dnn=False):
        self.weights = weights
        self.data = data
        self.device = device
        self.half = half
        self.dnn = dnn

    def load_model(self):
        FILE = Path(__file__).resolve()
        ROOT = FILE.parents[0]  # YOLOv5 root directory
        if str(ROOT) not in sys.path:
            sys.path.append(str(ROOT))  # add ROOT to PATH
        ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

        # Load model
        device = self.select_device(self.device)
        model = self.DetectMultiBackend(self.weights, device=device, dnn=self.dnn, data=self.data)
        stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine

        # Half
        half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
        if pt or jit:
            model.model.half() if half else model.model.float()
        return model, stride, names, pt, jit, onnx, engine

    def run(self, model, img, stride, pt, imgsz=(640, 640), conf_thres=0.25, iou_thres=0.45, max_det=1000,
            device='', classes=None, agnostic_nms=False, augment=False, half=False):
        cal_detect = []

        device = self.select_device(device)
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names

        # Set Dataloader
        im = self.letterbox(img, imgsz, stride, pt)[0]

        # Convert
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)

        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        pred = model(im, augment=augment)

        pred = self.non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = self.scale_coords(im.shape[2:], det[:, :4], img.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = f'{names[c]}'
                    cal_detect.append([label, xyxy])
        return cal_detect

    def detect(self):
        model, stride, names, pt, jit, onnx, engine = self.load_model()   # 加载模型
        image = cv2.imread("./images/1.jpg")   # 读取识别对象
        results = self.run(model, image, stride, pt)   # 识别， 返回多个数组每个第一个为结果，第二个为坐标位置
        for i in results:
            box = i[1]
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            print(i[0])
            cv2.rectangle(image, p1, p2, (0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
        cv2.imshow('image', image)
        cv2.waitKey(0)

    @staticmethod
    def select_device(device):
        # ...
        pass

    @staticmethod
    def DetectMultiBackend(weights, device, dnn, data):
        # ...
        pass

    @staticmethod
    def letterbox(img, imgsz, stride, pt):
        # ...
        pass

    @staticmethod
    def non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det):
        # ...
        pass

    @staticmethod
    def scale_coords(shape, coords, img_shape):
        # ...
        pass
