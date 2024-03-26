# AWINLAB 新生作業
※作業繳交形式 (請符合要求，否則以0分計算)  
(1) 請使用python做為作業的程式語言，限制繳交ipynb檔。  
(2) 請將作業上傳至Github，不同題請用資料夾做區分。  
(3) 每題都要附上README.md檔案說明程式的流程、方法與自己的想法。  
(4) 程式的部分需有註解。  
(5) 繳交時直接附上Github連結即可。  
## 深度學習（Deep Learning）－ (40分)
請至Kaggle下載70 Dog資料集，分出指定類別，並用深度學習的方法進行「分類」。  
https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set  
指定類別: ["Airedale", "Beagle", "Bloodhound", "Bluetick", "Chihuahua", "Collie", "Dingo", " French Bulldog", " German Sheperd", " Malinois", " Newfoundland", " Pekinese", " Pomeranian", "Pug", "Vizsla"]
* 請使用Keras的方法建立模型(10分)
* 模型(共20分)
    * 訓練及測試深度學習的模型請使用深度神經網路(Deep Neural Network, DNN)或者卷積神經網路(Convolutional Neural Network, CNN)，擇一即可。(15分)
    * 訓練集(Training Set)訓練完後，請針對驗證集(Valid set)透過一些性能指標衡量模型的好壞，並印出Valid set 的Accuracy。(5分)
    * 請使用下方雲端連結下載測試集(Testing set.zip)，進行測試，並將結果依格式輸出「test_data.xlsx」(10分)  
https://drive.google.com/drive/folders/18-CPmuGVojlaFHuxmz3piv64aC0dQMyd?usp=sharing


## 超啟發式學習（Meta-heuristic Algorithm）－ (60分)
https://people.sc.fsu.edu/~jburkardt/datasets/knapsack_01/knapsack_01.html  
請至上方連結下載P06的測試資料，並使用以下的方法來解決問題。
* 爬山演算法(Hill climbing, HC)。(20分)
* 基因演算法(Genetic algorithm, GA)。(40分) 

圖1、x軸為迭代次數，y軸為收斂的值(當前迭代所找到的最佳解)

爬山與基因演算法的部分，初始解請自己隨機取，並做100次迭代，畫出收斂圖(如圖1)。
