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

def softmax( f ):
    return np.exp(f) / np.sum(np.exp(f))

class FirstClassification():
    def __init__(self,
                 model_path='yolov5s.onnx',
                 image_size=[640, 640],
                 inference = "onnx",
                 ):
        self.weight_path = model_path

        # Load model
        self.model = ortInferenceSession(self.weight_path)
        # load_model(self.weight_path)
        self.imgsz = image_size


    def __call__(self, img,first_classificaton_classes):
        OUT = []
        im, im0s = load_image(img, img_size=self.imgsz)
        im = np.float32(np.divide(im, 255))  # 0 - 255 to 0.0 - 1.0
        im = im[np.newaxis, :]  # squeeze
        # Inference
        pred = self.model.run([self.model.get_outputs()[0].name], {self.model.get_inputs()[0].name: im})[0][0]
        pred = softmax(pred)
        top_index = np.argsort(-pred)[0]# top 5 indices
        print(top_index)
        result = False if top_index ==(1 or 3) else True
        return result


if __name__ == '__main__':
    from utils import imread_ex
    import os 
    class_weights = "/home/qw/shudian/code/model/Recognization-Sankua_v1.0.0-20230420.onnx"
    class_imgz = [1280, 1280]
    FirstClassification = FirstClassification(model_path=class_weights,image_size=class_imgz)
    first_classificaton_classes = ["kgl","ksd", "ktl", "nk"]
    image_dir = "/home/qw/shudian/data/sankua/fenlei/pred"
    save_txt = "/home/qw/shudian/data/sankua/fenlei/pred.txt"
    result_f = open(save_txt, "w", encoding="utf-8")
    for filename in os.listdir(image_dir):
        image = imread_ex(os.path.join(image_dir, filename))
        result = FirstClassification(image, first_classificaton_classes)
        result_post = "nk" if result==False else "kl"
        result_f.write(filename +" " + result_post +"\n")
    result_f.close()

