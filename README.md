## Step1: To extract feature images from the upstream model(by default, ResNet-50), execute the following instructions.
```shell
python feature_extraction.py
```
The default input dataset is the cifar10 dataset.</br>
The output includes:
1. `data/random/{0,10,20,30,40,50}.csv` (indices and labels of different portion of data)
2. `data/cifar_10/train/{0-49999}.npy` (50000 training feature images)
3. `data/cifar_10/test/{0-9999}.npy` (10000 testing feature images)
4. `data/test.csv` (indices and labels of test data)
5. `data/cifar_10/train_mix_{nd,uni}_{0.1,0.2,0.5,0.7,1.0}/{0-49999}.npy`</br>
   (training feature images mixed with noise of various types and intensities)
6. `data/cifar_10/test_mix_{nd,uni}_{0.1,0.2,0.5,0.7,1.0}/{0-9999}.npy` </br>
   (testing feature images mixed with noise of various types and intensities)

## Step2: Train the downstream model(a single-layer fully-connected network)
```shell
CUDA_VISIBLE_DEVICES=0 python -u train.py --SEED=42 --SAVE_CHECKPOINT=True
```
The input includes:
1. `data/cifar_10/train,train_mix_{nd,uni}_{0.1,0.2,0.5,0.7,1.0}/{0-49999}.npy`</br>
(training feature images mixed with noise of various types and intensities)
2. `data/random/0.csv` (indices and labels of orginal data) </br>

The output includes：
1. `saved/random/0/42/checkpoint{0-9}.pth` (model weights)
2. `saved/random/0/42/report/{0-9}_train_dev.csv`（train accuracy）
3. `saved/random/0/42/report/9_test.csv` (test accuracy)

## Step3 Compute memorization scores and memorization matrix
Using multiple GPUs for computations (for example: using 4 GPUs)
```shell
bash ./script/compute_mem_0.sh; bash ./script/compute_mem_1.sh; bash ./script/compute_mem_2.sh; bash ./script/compute_mem_3.sh
```
The input includes:
1. `data/cifar_10/train_mix_{nd,uni}_{0.1,0.2,0.5,0.7,1.0}/{0-49999}.npy`
2. `saved/random/0/42/checkpoint/9.pth` (儲存兩種雜訊完整訓練資料所訓練出來的權重)
3. `data/random_uni/0.csv` (完整訓練資料的index與label)</br>

The output is:
1. `saved/score_cifar10_mix_{uni,nd}_{0.1,0.2,0.5,0.7,1.0}/{0-9000}.pkl` (memorization scores and memorization matrices, indices and labels)

## Step4 Visualize the images projected by memorization matrices
```shell
python visualization.py
```
The input includes:
1. `saved/score_cifar10_mix_{uni,nd}_{0.1,0.2,0.5,0.7,1.0}/{0-9000}.pkl`
2. `data/random/0.csv`
The output is:
1. `saved/vis/{uni,nd}_{0.1,0.2,0.5,0.7,1.0}.pdf` (輸出由mem-matrix投影出來的記憶圖片)

## Step5 將記憶由高到低移除百分之10-50,輸出其index與label
```shell
python exclude_top-k.py
```
The input is:
1. `"saved/score_cifar10_mix_{uni,nd}_0/{0-9000}.pkl`
The output includes
1. `data/{mem_uni,mem_nd}/{0,10,20,30,40,50}.pkl` (remove top-k highest influence score)
2. `data/random/{0,10,20,30,40,50}.pkl`

## Step6 消融測試依照記憶高低移除以及依照隨機移除資料的訓練
```shell
bash ./script/train_exc_mem_{0,10,20,30,40,50}.sh
```
The input include:
1. `data/cifar_10/train`
2. `data/{mem_uni,mem_nd}/{0,10,20,30,40,50}.csv`
3. `data/random/0.csv` </br>

The output includes:
1. `saved/{mem_uni,mem_nd}/{0,10,20,30,40,50}/{0,1,2,3,42}/checkpoint/`  (checkpoint of the single layer)
2. `saved/{mem_uni,mem_nd}/{0,10,20,30,40,50}/{0,1,2,3,42}/report/`  (report accuracy)

```shell
bash ./script/train_exc_rand_{0,10,20,30,40,50}.sh
```
The input includes:
1. `data/cifar_10/train`
2. `data/random/{0,10,20,30,40,50}.csv` </br>

The output includes:
1. `saved/{random_uni,random_nd}/{0,10,20,30,40,50}/{0,1,2,3,42}/checkpoint/`  (checkpoint of the single layer)
2. `saved/{random_uni,random_nd}/{0,10,20,30,40,50}/{0,1,2,3,42}/report/`  (report training data accuracy)

## Step7 將測試圖片取特徵後,對其進行辨識得到正確率
```shell
bash ./script/test_{uni,nd}_x.sh (x= 0,10,20,30,40,50)
```
The input includes:
1. `data/cifar_10/test_{uni,nd}_{0,0.1,0.2,0.5,0.7,1.0}`
2. `saved/{mem_uni,random_uni,mem_nd,random_nd}/{0,10,20,30,40,50}/{0,1,2,3,42}/checkpoint/9.th` </br>

The output includes:
1. `saved/{mem_uni,random_uni,mem_nd,random_nd}/{0,10,20,30,40,50}/{0,1,2,3,42}/report/9_test_{uni,nd}_{0.1,0.2,0.5,0.7,1.0}`  #report test data accuracy

## Step8 兩種雜訊及不同雜訊強度的消融測試折線圖
```shell
python abtest_visual.py
```
The input includes:
1. `saved/score_cifar10_mix_{uni,nd}_{0,0.1,0.2,0.5,0.7,1.0}/{0-9000}.pkl`
2. `saved/{mem_uni,mem_nd}/{0,10,20,30,40,50}/{0,1,2,3,42}/report/9_test_{uni,nd}_{0.1,0.2,0.5,0.7,1.0}`  (report test data accuracy by the checkpoint trained with removing top-k memorization score data)
3. `saved/{random_uni,random_nd}/{0,10,20,30,40,50}/{0,1,2,3,42}/report/9_test_{uni,nd}_{0.1,0.2,0.5,0.7,1.0}` (report test data accuracy by the checkpoint trained with removing random-k data) </br>

The output includes:
1. `saved/vis/abtest_CIFAR_{uni,nd}_{0.1,0.2,0.5,0.7,1.0}.pdf`
