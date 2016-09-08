# Deep Learning 開発に必要な環境

## OS

* Ubuntuがおすすめ。
ほとんどのフレームワークはUbuntuをサポートしている。
* Macはハードウェアに制限がかかるのでおすすめしない。
* Windowsは対応していないフレームワークがあるので、様々なフレームワークを試したい人にはおすすめしない。  
* Windowsで動作を(MNISTデータセットを学習できるレベルで)確認したフレームワーク:
    * [Chainer](http://chainer.org/)
    * [mxnet](http://mxnet.io/)
    * [Theano](http://deeplearning.net/software/theano/index.html)
    * [Keras(バックグラウンドとしてTheano使用)](https://keras.io/ja/)
* Windowsで動作するはずのフレームワーク
    * [CNTK](https://github.com/Microsoft/CNTK)
    * [Caffe](http://caffe.berkeleyvision.org/)  
    [Windows版Caffe](https://github.com/niuzhiheng/caffe)があるが古い

## NVIDIA製チップセットを搭載したグラフィックボード(Highly recommended)

ニューラルネットワークはGPUを使うことで数倍から数十倍高速に動作する。
GPUのチップセットメーカーは複数あるが、ほとんどのフレームワークはCUDAにしか対応していないのでNVIDIAしか選択肢がない状態である。
