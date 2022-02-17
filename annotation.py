# create train.txt, test.txt and convert coco to voc
import pandas as pd
import os
import re
import random

'''
COCO Bounding box: (x-top left, y-top left, width, height)
Pascal VOC Bounding box :(x-top left, y-top left,x-bottom right, y-bottom right)
'''

classes_path = './model_data/cots_classes.txt'
train_path = './tensorflow-great-barrier-reef/train.csv'
test_path = './tensorflow-great-barrier-reef/test.csv'
img_path  = 'tensorflow-great-barrier-reef/train_images'

classes, _ = ['cots'], 1
trainval_percent = 0.9

if __name__ == '__main__':
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    train_data = train_data[train_data.annotations!='[]']

    n = len(train_data)
    train_list = random.sample(range(n), int(n * trainval_percent))

    f = open('train.txt', 'w')
    fv = open('val.txt', 'w')

    for i in range(n):
        a = train_data.annotations.iloc[i]
        l = re.findall(r'\d+', a)
        if i in train_list:
            f.write('%s/video_%s/%s.jpg' % (os.path.abspath(img_path), train_data.video_id.iloc[i], train_data.video_frame.iloc[i]))
            for j in range(len(l) // 4):
                x, y, w, h = list(map(int, l[4*j: 4*j+4]))
                f.write(' %s,%s,%s,%s,0' % (x, y, x + w, y + h))
            f.write('\n')
        else:
            fv.write('%s/video_%s/%s.jpg' % (os.path.abspath(img_path), train_data.video_id.iloc[i], train_data.video_frame.iloc[i]))
            for j in range(len(l) // 4):
                x, y, w, h = list(map(int, l[4 * j: 4 * j + 4]))
                fv.write(' %s,%s,%s,%s,0' % (x, y, x + w, y + h))
            fv.write('\n')
    f.close()
    fv.close()
    print("Generate train.txt and val.txt for train done.")