<!--
 * @Author: yanxinhao
 * @Email: 1914607611xh@i.shu.edu.cn
 * @LastEditTime: 2020-02-25 15:12:34
 * @LastEditors: yanxinhao
 * @Description: 
 -->
# FaceSwap

## Multi-decoder implement
### Training demo
Three ways to train
```
/train_comp_demo.py
```
Use all images of all persons to compute the loss. Same as the way described in Disney paper. Large GPU memory needed.
```
/train_comp_demo_lite.py
```
Use one batch of images of one person to compute the loss. Fast and low GPU memory used. This way of training may be wrong (not proved)

```
/train_comp_demo_batch.py
```
Use one batch of images of each person to compute the loss. Chosen to perform the demo now.

### Training data path
```
/data1/Dataset
```
Extracted face images of person X are stored in
```
/data1/Dataset/studentX
```
Extracted mask images (0-1 binary images) of person X are stored in
```
/data1/Dataset/maskX
```

### Source codes path
```
/deepfakes_pytorch
```

### Saved model path
```
/Training_Results/models
```
