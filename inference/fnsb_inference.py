import time
import numpy as np
import cv2
# from .utils import ( non_max_suppression, scale_coords, save_one_box)
from utils import ortInferenceSession

def letterbox(_image, dst_width, dst_height, border_value=(114, 114, 114),
              return_scale=False, interpolation=cv2.INTER_LINEAR):
    src_height, src_width = _image.shape[:2][0], _image.shape[:2][1]
    x_scale = dst_width / src_width
    y_scale = dst_height / src_height
    if y_scale > x_scale:
        resize_w = dst_width
        resize_h = int(x_scale * src_height)
        left = right = 0
        top = (dst_height - resize_h) // 2
        bottom = dst_height - resize_h - top
        scale = x_scale
    else:
        resize_w = int(y_scale * src_width)
        resize_h = dst_height
        left = (dst_width - resize_w) // 2
        right = dst_width - resize_w - left
        top = bottom = 0
        scale = y_scale
    resized_image = cv2.resize(_image, (int(resize_w), int(resize_h)), interpolation=interpolation)
    dst_image = cv2.copyMakeBorder(resized_image, int(top), int(bottom), int(left), int(right),
                                   cv2.BORDER_CONSTANT, value=border_value)
    if not return_scale:
        return dst_image
    else:
        return dst_image, scale, left, top

def load_image(img0, img_size=[640, 640]):

    img = letterbox(img0, img_size[0], img_size[1], return_scale=False)
    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    return  img, img0


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def nms(boxes, scores, iou_threshold):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = scores
    keep = []
    index = scores.argsort()[::-1]  # 从小到大——[::-1]反序

    while index.size > 0:
        i = index[0]  # every time the first is the biggst, and add it directly
        keep.append(i)
        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)  # the weights of overlap
        h = np.maximum(0, y22 - y11 + 1)  # the height of overlap

        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)

        idx = np.where(ious <= iou_threshold)[0]
        index = index[idx + 1]  # because index start from 1

    return keep


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300):
    """
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    x = np.squeeze(prediction)  # (1,nums,5+class) to (nums,5+class)
    nc = x.shape[1] - 5  # number of classes
    xc = x[..., 4] > conf_thres  # candidates   第一次筛选，obj_conf >conf_thres

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 7680  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()

    # Apply constraints
    x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height   第二次筛选，xywh不能超出限制
    x = x[xc]  # confidence
    # If none remain process next image
    if not x.shape[0]:
        return list()

    # Compute conf
    x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

    # Box (center x, center y, width, height) to (x1, y1, x2, y2)
    box = xywh2xyxy(x[:, :4])

    # Detections matrix nx6 (xyxy, conf, cls)
    conf = np.max(x[:, 5:], axis=1)
    cls = np.argmax(x[:, 5:], axis=1)
    conf = conf.reshape(-1, 1)
    cls = cls.reshape(-1, 1)
    x = np.concatenate((box, conf, cls), 1)[conf[:, -1] > conf_thres]  # 第三次筛选，obj_conf*cls  >conf_thres

    # Check shape
    n = x.shape[0]  # number of boxes
    if not n:  # no boxes
        print(f'can not find any objects with conf_thres!')
        return list()
    elif n > max_nms:  # excess boxes
        x = x[-x[:, 4].argsort()[:max_nms]]  # conf从大到小排序

    c = x[:, 5:] * max_wh

    # Batched NMS
    boxes, scores = x[:, :4], x[:, 4]  # boxes (offset by class), scores
    i = nms(boxes, scores, iou_thres)  # NMS

    if len(i) > max_det:  # limit detections
        i = i[:max_det]
    output = x[i]

    return output


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def save_one_box(box, im, gain=1.02, pad=10, square=False, BGR=False):
    # Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop
    box = np.reshape(box, (-1, 4))
    b = xyxy2xywh(box)  # boxes
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    box = np.floor(xywh2xyxy(b))
    clip_coords(box, im.shape)
    crop = im[int(box[0, 1]):int(box[0, 3]), int(box[0, 0]):int(box[0, 2]), ::(1 if BGR else -1)]

    return crop


class Detect():
    def __init__(self,
                 model_path='yolov5s.onnx',
                 image_size=[640, 640],
                 confidence_threshold=0.6,
                 iou_threshold=0.7,
                 inference = "onnx",
                 max_det=300,
                 ):
        self.weight_path = model_path
        self.conf_thres = confidence_threshold
        self.iou_thres = iou_threshold
        self.max_det = max_det
        self.inference = inference
        self.max_det = max_det

        # Load model
        self.model = ortInferenceSession(self.weight_path)
        # load_model(self.weight_path)
        self.imgsz = image_size


    def __call__(self, img, second_detect_classes):
        OUT = []
        im, im0s = load_image(img, img_size=self.imgsz)
        im = np.float32(np.divide(im, 255))  # 0 - 255 to 0.0 - 1.0
        im = im[np.newaxis, :]  # squeeze
        # Inference
        pred = self.model.run([self.model.get_outputs()[0].name], {self.model.get_inputs()[0].name: im})[0]
        # NMS
        det = non_max_suppression(pred, self.conf_thres, self.iou_thres, max_det=self.max_det)
        # Process predictions
    
        if len(det):  # (n,6)
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0s.shape).round()  # 原始图片坐标xyxy
            # crop image
            # cls: idx,需要通过 name 转换
            for xmin, ymin, xmax, ymax, conf, cls in det:
                label = second_detect_classes[int(cls)]
                box = [xmin, ymin, xmax, ymax]
                result = {"label":label, "location":{"xmin":xmin, "ymin":ymin, "xmax":xmax, "ymax":ymax}}
                OUT.append(result)
        return OUT


if __name__ == '__main__':
    from utils import imread_ex, plot_boxes
    import os 
    weights = "/home/qw/shudian/code/fangbiaosheshi_v1.onnx"
    imgz = [1280, 1280]
    Detect = Detect(model_path=weights,image_size=imgz)
    detect_classes= ["fc", "zc1", "zc2"]
    image_dir = "/home/qw/shudian/data/niaoci"
    save_txt_dir = "/home/qw/shudian/data/niaoci_txt"
    os.makedirs(save_txt_dir, exist_ok=True)
    for filename in os.listdir(image_dir):
        results_txt = os.path.join(save_txt_dir, filename.split(".")[0] +".txt")
        result_f = open(results_txt, "w", encoding="utf-8")
        txt_result = []
        image = imread_ex(os.path.join(image_dir, filename))
        results = Detect(image, detect_classes)
        image = plot_boxes(image, results)
        cv2.imencode('.JPG', image)[1].tofile(os.path.join(save_txt_dir, filename))
        for result in results:
            result_f.write(result["label"] +" " +str(result["location"]["xmin"]) +" " +
                            str(result["location"]["ymin"]) +" " + str(result["location"]["xmax"]) +" " +str(result["location"]["ymax"])  +"\n")
        result_f.close()



