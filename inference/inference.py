import os,sys 
from classification import FirstClassification
from second_detect import SecondDetect
from third_detect import ThirdDetect
import numpy as np 
import cv2
from utils import plot_boxes, imread_ex

## -------------data cfg ----------------------------------------------------
image_dir = "/home/qw/shudian/data/test/test_ori/±800kV雁淮线1937号/"
save_dir = "/home/qw/shudian/data/test/test_result/±800kV雁淮线1937号/"

## -------------model cfg ----------------------------------------------------
# first part
class_weights = "/home/qw/shudian/code/model/Recognization-Sankua_v1.0.0-20230420.onnx"
class_imgz = [1280,1280]

# second part
second_weights = "/home/qw/shudian/code/model/Detection-Jueyuanzichuan_v1.0.0-20230420.onnx"
second_imgz = [1280, 1280]

# third part
third_weights = "/home/qw/shudian/code/model/Detection-JYZGuadian_v1.0.0-20230420.onnx"
third_imgz = [1280, 1280]
device = "cpu"


# ----------label cfg --------------------------------------------

first_classificaton_classes = ["kgl", "ksd","ktl", "nk"]
second_detect_classes= ["xcjyz_1_1", "xcjyz_2_2", "xcjyz_2_1", "nzjyz_1_1", 'nzjyz_2_1']
second_fault_classes = ["xcjyz_1_1","nzjyz_1_1","xcjyz_2_1"]
second_fault_classes_chinese = ["悬垂绝缘子单串单挂点","耐张绝缘子单串单挂点","悬垂绝缘子双串单挂点"]

third_detect_classes = ["jueyuanzi_danguadian","jueyuanzi_shuangguadian"]
third_fault_classes = ["jueyuanzi_danguadian"]
third_fault_classes_chinese = ["单挂点"]

# 
def write_result_to_txt(result_f, txt_results) :
    # head
    result_f.write("三跨异常图片名".ljust(20,chr(12288)))
    result_f.write("异常绝缘子类型".ljust(20,chr(12288)))
    result_f.write("\n")
    result_f.write("***************************************************************************\n")  
    # table 
    for key, values in txt_results.items():
        name = key.split("号塔")[-1].split(".")[0]      
        result_f.write(name.ljust(20,chr(12288)))
        for index, value in enumerate(values):
            fault = second_fault_classes_chinese[second_fault_classes.index(value)]
            if index == 0:
                result_f.write(fault.ljust(20,chr(12288)))
                result_f.write("\n")
            else:
                result_f.write("".ljust(20,chr(12288)))
                result_f.write(fault.ljust(20,chr(12288)))
                result_f.write("\n")
        result_f.write("----------------------------------------------------------------------------\n")    



# # loade classification model 
Classifcation = FirstClassification(model_path=class_weights,image_size=class_imgz)

# loade second_detect model 
SecondDetect = SecondDetect(model_path=second_weights,image_size=second_imgz)

# load thire_detect model
ThirdDetect = ThirdDetect(model_path=third_weights,image_size=third_imgz)

# sankua
class_result = []
for image in os.listdir(image_dir):
    if "通道" in image:
        print(image)
        image_path = os.path.join(image_dir, image)
        image = imread_ex(image_path)
        result = Classifcation(image, first_classificaton_classes,flag)
        class_result.append(result)
sankua_result = True if True in class_result else False 
# if sankua go on 
if sankua_result:  
    txt_results = {}
    for imagename in os.listdir(image_dir):
        print(imagename)
        # detect the 绝缘子 picture 
        if "绝缘子" in imagename:
            final_results, txt_result = [], []
            image_path = os.path.join(image_dir, imagename)
            image = imread_ex(image_path)
            height, width = image.shape[0], image.shape[1]
            second_detect_results = SecondDetect(image, second_detect_classes) # [{label:xxx, location:{xmin:int,xmax:int,ymin:int, ymax:int}},{}]
            for second_detect_result in second_detect_results:
                # write fault to txt 
                if  second_detect_result["label"] in second_fault_classes: 
                    txt_result.append(second_detect_result["label"])
                if second_detect_result["label"] in second_fault_classes[2]:
                    xmin,ymin,xmax,ymax = second_detect_result["location"]['xmin'],  second_detect_result["location"]['ymin'],  second_detect_result["location"]['xmax'],  second_detect_result["location"]['ymax']
                    xmin = max(0, int(xmin - 0.05*(xmax-xmin)))
                    ymin = max(0, int(ymin - 0.05*(ymax-ymin)))
                    xmax = min(width, int(xmax + 0.05*(xmax-xmin)))
                    ymax = min(height, int(ymax + 0.05*(ymax-ymin)))
                    image_crop = image[ymin:ymax,xmin:xmax]
                    third_detect_results = ThirdDetect(image_crop, third_detect_classes)
                    for third_detect_result in third_detect_results:
                        third_detect_result["location"]["xmin"] = third_detect_result["location"]["xmin"]  + xmin 
                        third_detect_result["location"]["ymin"] = third_detect_result["location"]["ymin"]  + ymin 
                        third_detect_result["location"]["xmax"] = third_detect_result["location"]["xmax"]  + xmin 
                        third_detect_result["location"]["ymax"] = third_detect_result["location"]["ymax"]  + ymin 
                        if third_detect_result["label"] in third_fault_classes:
                            # third_detect_result["label"] = third_fault_classes_chinese[0]
                            final_results.append(third_detect_result)
                elif second_detect_result["label"] in second_fault_classes[:2]:
                    # # 转中文
                    # second_detect_result["label"] =second_fault_classes_chinese[0] if second_detect_result["label"] == second_fault_classes[0] else second_fault_classes_chinese[1]
                    final_results.append(second_detect_result)
                else:
                    continue  
            if len(txt_result):
                    txt_results[imagename] = txt_result 
            if len(final_results):
                image = plot_boxes(image, final_results,)
                # 保存站点路径
                # save_dir = os.path.dirname(image_path.replace("test_ori", "test_result"))
                # 输出异常图片结果
                os.makedirs(save_dir, exist_ok=True)
                cv2.imencode('.JPG', image)[1].tofile(os.path.join(save_dir, imagename))
    print(txt_results)
    if len(txt_results):
        # save result to txt 
        result_f = open(os.path.join(save_dir, "result.txt"), "w", encoding="utf-8") 
        write_result_to_txt(result_f, txt_results)      
            
else:
    print("不是三跨站点")
