import cv2 
import numpy as np 
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm # to create font

def imread_ex(filename, flags=-1):
    """cv2.imread 的扩展, 使支持中文路径.
    """
    try:
        return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), flags) #bgr
    except Exception as e:
        return None


def plot_boxes(img, results, color=None, line_thickness=10):
    """
    box和label显示
    """
    import random
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = tuple([random.randint(0, 255) for _ in range(4)])
    for result in results:
        x_min =result["location"]["xmin"]
        y_min = result["location"]["ymin"]
        x_max =result["location"]["xmax"]
        y_max = result["location"]["ymax"]
        c1, c2 = (int(x_min), int(y_min)), (int(x_max), int(y_max))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        label = result["label"]
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            #font = cv2.FONT_HERSHEY_SIMPLEX
            # imgPIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # drawPIL = ImageDraw.Draw(imgPIL)
            # fontText = ImageFont.truetype(font = fm.findfont(fm.FontProperties(family="simsun.ttc")), size=36)
            # # fontText = ImageFont.truetype("font/simsun.ttc", 400, encoding="utf-8")
            # drawPIL.text(c1, label, color, font=fontText)
            # imgPutText = cv2.cvtColor(np.asarray(imgPIL), cv2.COLOR_RGB2BGR)
            # cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0,  tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def ortInferenceSession(model_path, providers=['CUDAExecutionProvider']):
    """
    load onnx model 
    """
    import onnx
    import onnxruntime as ort
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 2
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        # cipher_key = mdsjg.cfg.cipher_key
        # cipher = Fernet(cipher_key)
        # with open(model_path, 'rb') as f:
        #     file_content = cipher.decrypt(f.read())

        # buffer = BytesIO()
        # buffer.write(file_content)
        # buffer.seek(0)
        onnx_model = onnx.load_model(model_path)
        return ort.InferenceSession(
            onnx_model.SerializeToString(), sess_options=sess_options, providers=providers
        )
    except:
        return ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)