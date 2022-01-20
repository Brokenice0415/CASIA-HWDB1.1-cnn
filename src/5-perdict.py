from keras.models import model_from_json
import json
import cv2
import h5py
import numpy as np

import utils

img_path = r'C:\Users\lap\Desktop\XRT\img\split\ha\img45\img212.png'
img_path = r'img.png'
model_filepath = 'model-1612617459.json'
weights_filepath = 'weights-1612617459-0.921989.hdf5'


label_list = ['谈','般','盏','坤','膀','脂','型','骏','童','挟','损','恋','婴','读','账','服',
              '任','茸','张','亢','耀','涉','个','随','挂','抗','贞','瞥','瘤','作','河','欲',
              '侵','吸','眺','线','捂','倾','牌','筒','渊','拥','话','赞','知','除','巩','惫',
              '揭','扬','驼','绿','渔','榆','辊','应','儡','假','崩','抬','是','讲','刷','鸿',
              '契','寒','录','教','也','艾','囤','秦','峨','括','诲','滴','凶','须','孽','巾',
              '沉','餐','暂','蒙','攘','键','厄','的','芭','岳','惜','椰','足','伴','离','笼',
              '临','胁','泉','晚','迟','汞','级','跳','轴','偶','啸','移','贾','老','节','蜗',
              '堑','帕','肖','伟','渝','撮','臀','吉','汉','反','双','坏','翔','胖','绪','固',
              '舀','再','咏','堂','尔','沟','符','涵','水','误','岿','所','摄','广','结','学',
              '苫','臭','恬','诱','递','烷','硼','茁','标','越','吏','笑','馒','耗','氟','加',
              '砧','稻','晃','臂','其','配','城','筑','痹','揖','江','连','卡','狠','瓤','乳',
              '赵','仿','睹','相','好','屿','争','袭','王','吃','疏','粕','涟','垣','逢','锤',
              '覆','薯','贴','冷','霸','聂','糕','占']

def make_tagcode_list():
    dataset_filepath = 'HWDB1.1.hdf5'
    with h5py.File(dataset_filepath, 'r') as f1, open(r'label_list.txt', 'a') as f2:
        allowed = list(set(utils.unicode_to_tagcode(c) for c in label_list))
        for i in range(897758):
            tagcode = f1['trn/tagcode'][i][0]
            if tagcode in allowed:
                f2.write(str(tagcode) + '\n')


if __name__ == "__main__":
    with open(model_filepath) as f:
        d = json.load(f)
        model = model_from_json(json.dumps(d))

    model.load_weights(weights_filepath)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # img = cv2.dilate(img, None)
    img = cv2.resize(img, (64, 64))

    cv2.imwrite(r'img.png', img)
    x = np.zeros([1, 1, 64, 64])
    x[0, 0] = img

    label_i = model.predict_classes(x)[0]
    print(label_i)

    with open(r'label_list.txt', 'r') as f:
        i = 0
        for tagcode in f:
            if i == label_i:
                label = utils.tagcode_to_unicode(int(tagcode))
                print(label)
                break
            else:
                i += 1