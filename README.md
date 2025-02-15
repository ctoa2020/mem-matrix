# An Explanable  Sheme for Memorization of Noisy Instances by Downstream Evaluation

<img src="https://github.com/user-attachments/assets/4d186a50-cbb2-474a-bbb0-52bfa5a41c7d" alt="Alt Text" style="width:200%; height:auto;">

## Installation
```shell
pip install -r requirement.txt
```

## Step1: To extract feature images from the upstream model(by default, ResNet-50), execute the following instructions.
```shell
python feature_extraction.py
```
The default input dataset is the cifar10 dataset.</br>
The output includes:
1. `data/random/{0,10,20,30,40,50}.csv` (indices and labels of different portion of data)
2. `data/cifar_10/train/{0-49999}.npy` (50000 training feature images)
3. `data/cifar_10/test/{0-9999}.npy` (10000 testing feature images)
4. `data/test.csv` (indices and labels of testing data)
5. `data/cifar_10/train_mix_{nd,uni}_{0.1,0.2,0.5,0.7,1.0}/{0-49999}.npy`</br>
   (training feature images mixed with noise of various types and intensities)
6. `data/cifar_10/test_mix_{nd,uni}_{0.1,0.2,0.5,0.7,1.0}/{0-9999}.npy` </br>
   (testing feature images mixed with noise of various types and intensities)

## Step2: Train the downstream model(a single-layer fully-connected network).
```shell
CUDA_VISIBLE_DEVICES=0 python -u train.py --SEED=42 --SAVE_CHECKPOINT=True
```
The input includes:
1. `data/cifar_10/train,train_mix_{nd,uni}_{0.1,0.2,0.5,0.7,1.0}/{0-49999}.npy`</br>
(training feature images mixed with noise of various types and intensities)
2. `data/random/0.csv` (indices and labels of training data) </br>

The output includes：
1. `saved/random/0/42/checkpoint{0-9}.pth` (model weights of different rounds)
2. `saved/random/0/42/report/{0-9}_train_dev.csv`（train accuracy）
3. `saved/random/0/42/report/9_test.csv` (test accuracy)

## Step3 Compute memorization scores and memorization matrix.
Using multiple GPUs for computations (for example: using 4 GPUs)
```shell
bash ./script/compute_mem_0.sh; bash ./script/compute_mem_1.sh; bash ./script/compute_mem_2.sh; bash ./script/compute_mem_3.sh
```
The input includes:
1. `data/cifar_10/train_mix_{nd,uni}_{0.1,0.2,0.5,0.7,1.0}/{0-49999}.npy`
2. `saved/random/0/42/checkpoint/9.pth` (the latest modell weight)
3. `data/random_uni/0.csv` (indices and labels of training data))</br>

The output is:
1. `saved/score_cifar10_mix_{uni,nd}_{0.1,0.2,0.5,0.7,1.0}/{0-9000}.pkl` (memorization scores and memorization matrices, indices and labels)

## Step4 Visualize the images projected by memorization matrices
```shell
python visualization.py
```
The input includes:
1. `saved/score_cifar10_mix_{uni,nd}_{0.1,0.2,0.5,0.7,1.0}/{0-9000}.pkl`
2. `data/random/0.csv` (indices and labels of training data) </br>

The output is:
1. `saved/vis/{uni,nd}_{0.1,0.2,0.5,0.7,1.0}.pdf` (the image visualizations projected by memorization matrices)

## Step5 Exclude the training data with top-k high memorization scores
```shell
python exclude_top-k.py
```
The input is:
1. `"saved/score_cifar10_mix_{uni,nd}_0/{0-9000}.pkl`
The output includes
1. `data/{mem_uni,mem_nd}/{0,10,20,30,40,50}.pkl` (excluding training data with top-k highest memorization scores)
2. `data/random/{0,10,20,30,40,50}.pkl` (excluding random-k portions of training data)

## Step6.1 Ablation test: Train the downstream model with the remaining training data after excluding top-k highest memorization scores
```shell
bash ./script/train_exc_mem_{0,10,20,30,40,50}.sh
```
The input include:
1. `data/cifar_10/train`
2. `data/{mem_uni,mem_nd}/{0,10,20,30,40,50}.csv`
3. `data/random/0.csv` </br>

The output includes:
1. `saved/{mem_uni,mem_nd}/{0,10,20,30,40,50}/{0,1,2,3,42}/checkpoint/`  (various checkpoints of the downstream model)
2. `saved/{mem_uni,mem_nd}/{0,10,20,30,40,50}/{0,1,2,3,42}/report/`  (report training accuracies)

## Step6.2 Ablation test: Train the downstream model with the remaining training data after excluding the random-k portions of data
```shell
bash ./script/train_exc_rand_{0,10,20,30,40,50}.sh
```
The input includes:
1. `data/cifar_10/train`
2. `data/random/{0,10,20,30,40,50}.csv`
3. `data/random/0.csv` </br>

The output includes:
1. `saved/{random_uni,random_nd}/{0,10,20,30,40,50}/{0,1,2,3,42}/checkpoint/`  (various checkpoints of the downstream model)
2. `saved/{random_uni,random_nd}/{0,10,20,30,40,50}/{0,1,2,3,42}/report/`  (report train accuracies)

## Step7 Use the model checkpoints in Step 6.1 and 6.2 to classify testing data.
```shell
bash ./script/test_{uni,nd}_x.sh (x= 0,10,20,30,40,50)
```
The input includes:
1. `data/cifar_10/test_{uni,nd}_{0,0.1,0.2,0.5,0.7,1.0}`
2. `saved/{mem_uni,random_uni,mem_nd,random_nd}/{0,10,20,30,40,50}/{0,1,2,3,42}/checkpoint/9.th` </br>

The output includes:
1. `saved/{mem_uni,random_uni,mem_nd,random_nd}/{0,10,20,30,40,50}/{0,1,2,3,42}/report/9_test_{uni,nd}_{0.1,0.2,0.5,0.7,1.0}`  (report test accuracy)

## Step8 Draw and save the performance diagrams of ablation test for two noise types and various noise intensities.
```shell
python abtest_visual.py
```
The input includes:
1. `saved/score_cifar10_mix_{uni,nd}_{0,0.1,0.2,0.5,0.7,1.0}/{0-9000}.pkl`
2. `saved/{mem_uni,mem_nd}/{0,10,20,30,40,50}/{0,1,2,3,42}/report/9_test_{uni,nd}_{0.1,0.2,0.5,0.7,1.0}`</br>
   (report test data accuracy by the checkpoint trained with removing top-k memorization score data)
4. `saved/{random_uni,random_nd}/{0,10,20,30,40,50}/{0,1,2,3,42}/report/9_test_{uni,nd}_{0.1,0.2,0.5,0.7,1.0}` </br>
   (report test data accuracy by the checkpoint trained with removing random-k data)

The output includes:
1. `saved/vis/abtest_CIFAR_{uni,nd}_{0.1,0.2,0.5,0.7,1.0}.pdf`</br>
(the performance diagrams of ablation test for two noise types and various noise intensities)
