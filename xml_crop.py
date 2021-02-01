import xml.etree.cElementTree as et   # 读取xml文件的包
import os
import cv2
def GetFileList(dir, fileList):
    newDir = dir
    if os.path.isfile(dir):
        fileList.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            # if s == "pts":
            #     continue
            newDir=os.path.join(dir,s)
            GetFileList(newDir, fileList)
    return fileList

label=0

data_dir='./label'+str(label)
save_dir='crops/'+str(label)


xml_list=[]
GetFileList(data_dir,xml_list)
xml_list=[x for x in xml_list if 'xml' in x]



cnt=0
for xml_name in xml_list:
    try:
        tree = et.parse(xml_name)
    except:
        print(xml_name,'err')
        continue
    root = tree.getroot()  # 使用getroot()获取根节点，得到的是一个Element对象

    img_name=root.find('filename').text

    image_path=os.path.join(data_dir,img_name)

    image=cv2.imread(image_path,-1)


    obj=root.find('object')

    if obj is  None:
        continue
    label=obj.find('name').text

    if label=='0':

        xml_box = obj.find('bndbox')
        xmin = int(float(xml_box.find('xmin').text))
        ymin = int(float(xml_box.find('ymin').text))
        xmax = int(float(xml_box.find('xmax').text))
        ymax = int(float(xml_box.find('ymax').text))

        # print(xmin)
        cur_patch=image[ymin:ymax,xmin:xmax,:]

        save_path=os.path.join(save_dir,str(cnt)+'.jpg')
        cv2.imwrite(save_path,cur_patch)
        # cv2.imshow('ss',cur_patch)
        # cv2.waitKey(0)
        cnt+=1


