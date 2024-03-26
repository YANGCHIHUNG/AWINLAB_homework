# i. Deep Learning
## 1. Import需要的套件
```
import numpy as np 
import pandas as pd 
import cv2
import os
import requests
from io import BytesIO

import matplotlib.pyplot as plt
#import plotly.graph_objs as go
#import seaborn as sns  

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, array_to_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D, Input , BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.applications import InceptionResNetV2,InceptionV3
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn import model_selection, metrics, preprocessing
from tensorflow.python.keras.utils import np_utils
```
## 2. 設定資料集的路徑
```
TRAIN_DIR = r'C:\Users\young\Documents\AWINLAB\i.Deep Learning\dataset\train'
VAL_DIR = r'C:\Users\young\Documents\AWINLAB\i.Deep Learning\dataset\valid'
TEST_DIR = r'C:\Users\young\Documents\AWINLAB\i.Deep Learning\dataset\test'
```
## 3. 依照作業的指示，留下需要的指定類別
```
data_path = '/Users/young/Documents/AWINLAB新生作業/i.深度學習(Deep Learning)/dataset/train'
class_names = sorted(os.listdir(data_path))
num_classes = len(class_names)

for name in sorted(os.listdir(data_path)):
    flag = 0
    if(name == '.DS_Store'):
        continue
    for target_name in target_data:
        if(name == target_name):
            flag = 1
    if(flag == 0):
        shutil.rmtree('/Users/young/Documents/AWINLAB新生作業/i.深度學習(Deep Learning)/dataset/valid/' + name)
```
## 4. 處理圖片
```
data_gen = ImageDataGenerator(rescale=1/255)
```
將圖片控制在0~1之間
## 5. 訓練模型
設定資料集
```
BATCH_SIZE = 32
DIMS = (224,224)
IMG_SIZE = 224

train_data = data_gen.flow_from_directory(batch_size=BATCH_SIZE, directory=TRAIN_DIR, shuffle=True,
                                            target_size=DIMS, class_mode='categorical')

val_data = data_gen.flow_from_directory(batch_size=BATCH_SIZE, directory=VAL_DIR, shuffle=True,
                                            target_size=DIMS, class_mode='categorical')

test_data = data_gen.flow_from_directory(batch_size=BATCH_SIZE, directory=TEST_DIR, shuffle=True,
                                            target_size=DIMS, class_mode='categorical')
```
---
建立InceptionV3模型
```
IMG_SHAPE = (224,224,3)
base_model = InceptionV3(input_shape=IMG_SHAPE, include_top=False, weights='imagenet', classes=70)

base_model.trainable = False

model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(100,activation='relu'))
model.add(Dense(15,activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='loss', patience=5, verbose=2, factor=0.001)
es = EarlyStopping(monitor='loss', verbose=2, patience=10, min_delta=0.001)
```
---
開始訓練模型
```
logs = model.fit(train_data, validation_data=val_data,
                steps_per_epoch = train_data.samples//BATCH_SIZE,
                validation_steps = val_data.samples//BATCH_SIZE,
                epochs=80, verbose=1, callbacks=[reduce_lr,es])

model.summary()
```
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential_2"</span>

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ inception_v3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Functional</span>)       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">5</span>, <span style="color: #00af00; text-decoration-color: #00af00">5</span>, <span style="color: #00af00; text-decoration-color: #00af00">2048</span>)     │    <span style="color: #00af00; text-decoration-color: #00af00">21,802,784</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ global_average_pooling2d_1      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2048</span>)           │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">GlobalAveragePooling2D</span>)        │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">100</span>)            │       <span style="color: #00af00; text-decoration-color: #00af00">204,900</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">15</span>)             │         <span style="color: #00af00; text-decoration-color: #00af00">1,515</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘

<span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">22,422,031</span> (85.53 MB)

<span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">206,415</span> (806.31 KB)

<span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">21,802,784</span> (83.17 MB)

<span style="font-weight: bold"> Optimizer params: </span><span style="color: #00af00; text-decoration-color: #00af00">412,832</span> (1.57 MB)
</pre>
## 6. 評估模型
計算出test_data的accuracy、loss
```
loss_cnn, accuracy_cnn = model.evaluate(test_data)
loss_cnn = loss_cnn*100
accuracy_cnn = accuracy_cnn*100
print('Test loss:', loss_cnn)
print('Test accuracy:', accuracy_cnn)
```
Test loss: 2.018200419843197
Test accuracy: 99.33333396911621

---

計算出valid_data的accuracy、loss
```
loss_cnn, accuracy_cnn = model.evaluate(val_data)
loss_cnn = loss_cnn*100
accuracy_cnn = accuracy_cnn*100
print('Test loss:', loss_cnn)
print('Test accuracy:', accuracy_cnn)
```
Test loss: 2.0647358149290085
Test accuracy: 99.33333396911621

---
轉換成折線圖
```
train_acc = logs.history['accuracy']
val_acc = logs.history['val_accuracy']

plt.plot(range(1, len(train_acc) + 1), train_acc, label='Training Accuracy')
plt.plot(range(1, len(val_acc) + 1), val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```
![image](https://github.com/yangchihung/AWINLAB_homework/blob/master/img/output_1.png)


```
train_loss = logs.history['loss']
val_loss = logs.history['val_loss']

plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
plt.plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```
![image](https://github.com/yangchihung/AWINLAB_homework/blob/master/img/output_2.png)

---

```
#train loss
plt.plot(logs.history['loss'])
#test loss
plt.plot(logs.history['val_loss'])
#標題
plt.title('Model loss')
#y軸標籤
plt.ylabel('Loss')
#x軸標籤
plt.xlabel('Epoch')
#顯示折線的名稱
plt.legend(['Train', 'Test'], loc='upper left')
#顯示折線圖
plt.show()
```
![image](https://github.com/yangchihung/AWINLAB_homework/blob/master/img/output_3.png)


---
在網路上隨機找圖片測試
```
# 印出測試集隨機十筆預測後的結果和正解進行比較
url = [
    'https://www.google.com/url?sa=i&url=https%3A%2F%2Fencrypted-tbn3.gstatic.com%2Flicensed-image%3Fq%3Dtbn%3AANd9GcRv7Ev1T8O6as52YDwz3YDa9ya3-xv5SpMw3Lk_mZHqxwWvGDCw47ZaixFiTefWHF_dHHqDFFSYk2ZRKfU&psig=AOvVaw2brgj8mLWs5S3os1vdjdgh&ust=1690324655029000&source=images&cd=vfe&opi=89978449&ved=0CBEQjRxqFwoTCMDy4pa0qIADFQAAAAAdAAAAABAE',
    'https://images.chinatimes.com/newsphoto/2019-01-31/656/20190131001004.jpg',
    'https://cdn2.ettoday.net/images/7229/d7229507.jpg',
    'https://cdn.britannica.com/42/233842-050-E64243D7/Pomeranian-dog.jpg',
    'https://www.akc.org/wp-content/uploads/2017/11/Pekingese-standing-in-the-grass.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Wy%C5%BCe%C5%82_w%C4%99gierski_g%C5%82adkow%C5%82osy_500.jpg/1200px-Wy%C5%BCe%C5%82_w%C4%99gierski_g%C5%82adkow%C5%82osy_500.jpg',
    'https://cdn.britannica.com/85/232785-050-0EE871BE/Belgian-Malinois-dog.jpg',
    'https://dogsbestlife.com/wp-content/uploads/2019/11/Belgian-malinois-scaled.jpeg',
    'https://cdn.britannica.com/92/171292-050-AA6ABC3A/species-authorities-dingos-wolf-subspecies.jpg',
    'https://people.com/thmb/ZgAGcuEb7dooKyT9DygAo7XO7ZQ=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc():focal(749x0:751x2)/dingo-040423-dec34a251fb444feb2001bfc23aa3542.jpg'
]
count=0
row=5 
col=2
fig,ax=plt.subplots(row,col) # row=2 col=5
fig.set_size_inches(20,10)
for i in range (row):
    for j in range (col):
        response = requests.get(url[count])
        img = load_img(BytesIO(response.content), target_size=DIMS)
        
        arr = img_to_array(img)
        ax[i,j].imshow(img)
        arr = arr/255.0
        arr = np.expand_dims(arr,0)
        res = model.predict(arr)
        idx = res.argmax()
        ax[i,j].set_title("Predicted: "+label_mapper[idx])
        plt.tight_layout()
        count+=1
```
![image](https://github.com/yangchihung/AWINLAB_homework/blob/master/img/output_4.png)
在網路上隨機搜尋的十張圖片中，全部都預測正確。

---
## 7.對測試集(Testing set.zip)進行測試，並將結果依格式輸出「test_data.xlsx」(10分)
```
import openpyxl

wb = openpyxl.load_workbook(r'C:\Users\young\Documents\AWINLAB\i.Deep Learning\test_data.xlsx') 
s1 = wb['Sheet1']

test_data_path = r'C:\Users\young\Documents\AWINLAB\i.Deep Learning\Testing set'

img_names = sorted(os.listdir(test_data_path))

row = 2

for i in img_names:
    if(i == ".DS_Store"):
        continue
    test_img = test_data_path + "/" + i
    img = image.load_img(test_img, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img/255 # rescale

    pred = model.predict(img)[0]
    #print(pred)
    index = np.argmax(pred)
    prediction = target_data[index]
    print(row-1,", prep:", prediction,)

    s1.cell(row, 1).value = i
    s1.cell(row, 2).value = prediction
    row = row + 1

wb.save(r'C:\Users\young\Documents\AWINLAB\i.Deep Learning\test_data.xlsx')
```
![image](https://github.com/yangchihung/AWINLAB_homework/blob/master/img/output_5.png)
