<!--
 * @Author: yanxinhao
 * @Email: 1914607611xh@i.shu.edu.cn
 * @LastEditTime: 2020-04-30 18:05:44
 * @LastEditors: yanxinhao
 * @Description: 
 -->
# RealtimeDF

## 1.Introduction

It is a project to implement Realtime system of [deepfakes](https://github.com/iperov/DeepFaceLab). 
    

## 2.Install RealtimeDF

> conda create -n deepfacelab -c main python=3.6.8 cudnn=7.6.5 cudatoolkit=10.0.130<br/>
conda activate deepfacelab<br/>
git clone https://github.com/DGeneffects/RealtimeDF.git<br/>
cd RealtimeDF/<br/>
python -m pip install -r ./requirements-cuda.txt<br/>

### config:
**One thing before :** put your own model into `./checkpoint/model`.

All of the relative parameters are saved in `config.Config` class.You should edit the file first before you run your own model.

## 3.Debug

The filefolder "debug/" contains some files each of which is used to test different modules.

## 4.Release
If you want to try it , you can just run main.py

## 5.Finally
Have Fun!