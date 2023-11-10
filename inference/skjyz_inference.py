import os,sys 
from first_detect import FirstDetect
from second_detect import SecondDetect
from third_detect import ThirdDetect
import numpy as np 
import cv2
from utils import plot_boxes, imread_ex
from PIL  import Image, ImageDraw, ImageFont  



dir_list = []
def traverse_directory(path):
    # list all files and directories in the path
    files_list = os.listdir(path)
    for file in files_list:
        # create the full path to the file or directory
        full_path = os.path.join(path, file)
        if os.path.isdir(full_path):
            # if the full path is a directory, call the function recursively
            traverse_directory(full_path)
            # do something with the directory here
            print("Directory: ", full_path)
        else:
            # do something with the file here
            dir_list.append(path)
            continue
 # call the function with the root directory path


## -------------data cfg ----------------------------------------------------
# image_dir = "/home/qw/shudian/data/test/test_ori/500kV洛庄5306线067号/"
# save_dir = "/home/qw/shudian/data/test/test_result/500kV洛庄5306线067号/"

## -------------model cfg ----------------------------------------------------
# first part
class_weights = "/home/qw/shudian/code/model/Sankua-v3.onnx"
class_imgz = [1280,1280]

# second part
second_weights = "/home/qw/shudian/code/model/SecondDetection.onnx"
second_imgz = [1280, 1280]

# third part
third_weights = "/home/qw/shudian/code/model/ThirdDetection.onnx"
third_imgz = [1280, 1280]
device = "cpu"


# ----------label cfg --------------------------------------------

first_classificaton_classes = ["kgl","ktl"]
second_detect_classes= ["xcjyz_1_1", "xcjyz_2_2", "xcjyz_2_1", "nzjyz_1_1", 'nzjyz_2_1',"dx_1_1","dx_2_2","dx_2_1","vc"]
second_nz_fault_classes = ["nzjyz_1_1"]
second_xc_fault_classes = ["xcjyz_1_1","dx_1_1","xcjyz_2_1","dx_2_1"]
second_fault_classes = ["xcjyz_1_1","dx_1_1","xcjyz_2_1","dx_2_1", "nzjyz_1_1"]
second_fault_classes_chinese = ["悬垂绝缘子单串单挂点","地线绝缘子单串单挂点","悬垂绝缘子双串单挂点","地线绝缘子双串单挂点","耐张绝缘子单串单挂点"]

third_detect_classes = ["jueyuanzi_danguadian","jueyuanzi_shuangguadian"]
third_fault_classes = ["jueyuanzi_danguadian"]
#third_fault_classes_chinese = ["单挂点"]



# # loade first detect model 
FirstDetect = FirstDetect(model_path=class_weights,image_size=class_imgz)

# loade second_detect model 
SecondDetect = SecondDetect(model_path=second_weights,image_size=second_imgz)

# load thire_detect model
ThirdDetect = ThirdDetect(model_path=third_weights,image_size=third_imgz)


def write_result_to_image(open_path,save_path,result_f): #dir_path+'/'+all_tower   save_path+'/'+all_tower
    img=Image.open(open_path)
    w,h=img.size
    font_size=int(h/60)  # 字体大小
    font_space=int(h/40) # 字体行间隔
    setfont = ImageFont.truetype('Arial.Unicode.ttf', font_size)
    draw = ImageDraw.Draw(img)
    lines = result_f.readlines()
    y_loc=font_size+font_space
    for line in lines :
        draw.text(((100,y_loc)),' %s\n'%(line),font=setfont, fill=(255, 255, 0))
        y_loc=y_loc+font_space
    img.save(save_path, 'JPEG')

# 
def write_result_to_txt(result_f, txt_results) :
    # head
    result_f.write("是否三跨：是\n")
    result_f.write("\n")
    result_f.write("图片名称".center(20,chr(12288)))
    #result_f.write("".ljust(20,chr(12288)))
    result_f.write("绝缘子类型".center(20,chr(12288)))
    result_f.write("异常类型".center(20,chr(12288)))
    result_f.write("异常数量".center(20,chr(12288)))
    result_f.write("\n")
    result_f.write("***"*80 +"\n")  
    # table 
    
    num_nzjyz_1_1, num_xcjyz_2_1, num_xcjyz_1_1 , num_dx_2_1, num_dx_1_1 = 0, 0, 0, 0, 0
    for key, values in txt_results.items():
        name = key.split(".")[0]      
        result_f.write(name.center(20,chr(12288)))
        #result_f.write("".ljust(20,chr(12288)))
        type_juz = []
        values= Counter(values)
        i = 0 
        for index, value in values.items():
            fault = second_fault_classes_chinese[second_fault_classes.index(index)]
            if index == "xcjyz_1_1" :
                num_xcjyz_1_1 +=1 
            elif index == "nzjyz_1_1" :
                num_nzjyz_1_1 +=1
            elif index == "xcjyz_2_1":
                num_xcjyz_2_1 +=1 
            elif index == "dx_2_1":
                num_dx_2_1 +=1               
            else:
                num_dx_1_1 +=1
            if "悬垂绝缘子" in fault :
                type_jyz = "悬垂绝缘子"
            elif "耐张绝缘子" in fault:
                type_jyz = "耐张绝缘子"
            else:
                type_jyz = "地线绝缘子"

            type_fault = "单串"if "单串" in fault else "双串单挂点"
            if i == 0:
                result_f.write(type_jyz.center(20,chr(12288)))
                result_f.write(type_fault.center(20,chr(12288)))
                result_f.write(str(value).center(20,chr(12288)))
                result_f.write("\n")
            else:
                result_f.write("".center(20,chr(12288)))
                result_f.write(type_jyz.center(20,chr(12288)))
                result_f.write(type_fault.center(20,chr(12288)))
                result_f.write(str(value).center(20,chr(12288)))
                result_f.write("\n")
            i += 1 
    result_f.write("---"*80 +"\n") 
    result_f.write("\n")   
    result_f.write("【异常结果统计】\n")
    result_f.write("耐张绝缘子单串：{num_nzjyz_1_1}    悬垂绝缘子单串：{num_xcjyz_1_1}    悬垂绝缘子双串单挂点：{num_xcjyz_2_1}    地线绝缘子双串单挂点：{num_dx_2_1}    地线绝缘子单串单挂点：{num_dx_1_1}".format(num_nzjyz_1_1=num_nzjyz_1_1, num_xcjyz_1_1=num_xcjyz_1_1, num_xcjyz_2_1=num_xcjyz_2_1, num_dx_2_1=num_dx_2_1, num_dx_1_1=num_dx_1_1))



"""
文件夹格式如下：
|--dir
  `|--dir1(塔1)
   |--dir2(塔2)
   |--dir3(塔3)
   .....
    `|---image1
     |---image2
     |---image3
     .....

"""
def inference(image_dir, save_dir, flag) :
    # 遍历文件夹
    traverse_directory(image_dir)
    dir_list = set(dir_list)
    for dir_file in dir_list: # 所有文件夹
        class_result = []
        flag = 0
        for image_name in os.listdir(dir_file): # 对应文件夹下的文件
            file_path = os.path.join(dir_file, image_name)
            if (not os.path.isdir(file_path)) and (not "红外" in dir_file) : # 文件夹下面没有文件夹,且不是红外
                # sankua
                if "通道" in image_name:
                    print(image_name)
                    flag =1 
                    image = imread_ex(file_path)
                    result = FirstDetect(image, first_classificaton_classes)
                    sankua_show_name = image_name if len(result)
                    class_result.extend(result)
        if flag == 0:
            print("文件夹下一级无通道数据，无法判断")
            continue
        print(class_result)
        sankua_result = True if len(class_result) else False 
        # if sankua go on 
        if sankua_result:  
            save_result_dir = dir_file.replace(image_dir, save_dir)
            os.makedirs(save_result_dir, exist_ok=True)
            for imagename in os.listdir(image_dir):
                print(image)
                # detect the 绝缘子 picture 
                if "绝缘子" in imagename:
                    final_results, txt_result, txt_results = [], [],{}
                    image_path = os.path.join(image_dir, imagename)
                    image = imread_ex(image_path)
                    height, width = image.shape[0], image.shape[1]
                    second_detect_results = SecondDetect(image, second_detect_classes) # [{label:xxx, location:{xmin:int,xmax:int,ymin:int, ymax:int}},{}]
                    
                    # 判断是否是耐张塔
                    second_detect_result_labels = []
                    for second_detect_result in second_detect_results:
                        second_detect_result_labels.append(second_detect_result["label"])
                    inter = list(set(second_detect_result_labels).intersection(set(second_detect_classes[3:5]))) # 判断是否存在耐张绝缘子
                    is_nz = True if len(inter) else False 
                    second_fault_classes = second_nz_fault_classes if len(inter) else  second_xc_fault_classes 
                                
                    # 将缺陷绝缘子写进txt中保存
                    for second_detect_result in second_detect_results:
                        # write fault to txt 
                        if  second_detect_result["label"] in second_fault_classes: 
                            txt_result.append(second_detect_result["label"])

                    #绝缘子挂点检测
                    for second_detect_result in second_detect_results:
                        if is_nz:
                            if second_detect_result["label"] in second_nz_fault_classes:
                                final_results.append(second_detect_result)
                            else:
                                print("no fault")

                        else:
                            if second_detect_result["label"] in second_xc_fault_classes[2:]:
                                xmin,ymin,xmax,ymax = second_detect_result["location"]['xmin'],  second_detect_result["location"]['ymin'],  second_detect_result["location"]['xmax'],  second_detect_result["location"]['ymax']
                                xmin = max(0, int(xmin - 0.05*(xmax-xmin)))
                                ymin = max(0, int(ymin - 0.05*(ymax-ymin)))
                                xmax = min(width, int(xmax + 0.05*(xmax-xmin)))
                                ymax = min(height, int(ymax + 0.05*(ymax-ymin)))
                                image_crop = image[ymin:ymax,xmin:xmax]
                                third_detect_results = ThirdDetect(image_crop, third_detect_classes)
                                # 对应位置，同时只标注横担点
                                if len(third_detect_results)==2:
                                    third_result_filter = []
                                    ymin_ = int(ymax - ymin )
                                    for third_detect_result in third_detect_results:
                                        ymin_ = min(ymin_, third_detect_result["location"]["ymin"])
                                    third_result_filter = [third_detect_result if ymin_ == third_detect_result["location"]["ymin"] for third_detect_result in third_detect_results]
                                elif len(third_detect_results)==1: # 如果只有一个挂点出现
                                    if third_detect_results["location"]["ymin"] < int(ymax - ymin )/2: # TODO 根据挂点的上下位置来判断
                                        third_result_filter = third_detect_results

                                for third_detect_result in third_result_filter:
                                    third_detect_result["location"]["xmin"] = third_detect_result["location"]["xmin"]  + xmin 
                                    third_detect_result["location"]["ymin"] = third_detect_result["location"]["ymin"]  + ymin 
                                    third_detect_result["location"]["xmax"] = third_detect_result["location"]["xmax"]  + xmin 
                                    third_detect_result["location"]["ymax"] = third_detect_result["location"]["ymax"]  + ymin 

                                    if third_detect_result["label"] in third_fault_classes:
                                        # third_detect_result["label"] = third_fault_classes_chinese[0]
                                        final_results.append(third_detect_result)
                            elif second_detect_result["label"] in second_xc_fault_classes[:2]:
                                # # 转中文
                                # second_detect_result["label"] =second_fault_classes_chinese[0] if second_detect_result["label"] == second_fault_classes[0] else second_fault_classes_chinese[1]
                                final_results.append(second_detect_result)
                            else:
                                print("no fault")
                    if len(txt_result):
                            txt_results[imagename] = txt_result

                    if len(final_results):
                        image = plot_boxes(image, final_results,)
                        # 保存站点路径
                        # save_dir = os.path.dirname(image_path.replace("test_ori", "test_result"))
                        # 输出异常图片结果
                        cv2.imencode('.JPG', image)[1].tofile(os.path.join(save_result_dir, imagename))
            if len(txt_results):
                # save result to txt 
                result_f = open(os.path.join(save_result_dir, "result.txt"), "w", encoding="utf-8") 
                write_result_to_txt(result_f, txt_results)  
                result_f.close()
            else:
                result_f = open(os.path.join(save_result_dir, "result.txt"), "w", encoding="utf-8") 
                result_f.write("是否三跨：是\n") 
                result_f.write("---"*80 +"\n") 
                result_f.write("\n")   
                result_f.write("【异常结果统计】\n")
                result_f.write("无异常结果")          
                result_f.close()
            result_f = open(os.path.join(save_result_dir, "result.txt"), "r", encoding="utf-8")
            # shutil.copy(os.path.join(image_dir, image_name), os.path.join(save_dir, image_name))
            write_result_to_image(os.path.join(image_dir, sankua_show_name),os.path.join(save_result_dir, sankua_show_name), result_f) 
            result_f.close()

        else:
            result_f = open(os.path.join(save_result_dir, "result.txt"), "w", encoding="utf-8") 
            result_f.write("是否三跨：否")  
            result_f.close()
            print("不是三跨站点")



def main(opt):
    image_dir, save_dir = opt.image_dir, opt.save_dir 
    inference( image_dir, save_dir)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default=r'D:\开发代码\project\test\三跨测试\独立双串\220kV鹭贺4V80线021#', help='weights path')
    parser.add_argument('--save_dir', type=str, default=r'D:\开发代码\project\data_result\独立双串\220kV鹭贺4V80线021#', help='weights path')

    opt= parser.parse_args()
    return opt

if __name__=='__main__':
    opt=parse_opt()
    main(opt)





