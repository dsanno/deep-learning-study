# Chainerのインストール方法

## Linux, MacOS

[公式のInstall Guide](http://docs.chainer.org/en/stable/install.html) を参照してください

## Windows7以降

公式にはWindowsをサポートしていませんが動作することを確認しています。
少し古いですが以下の手順でインストールできると思います。

[Windows上にChainer v1.5+CUDA+cuDNNを一番簡単に入れれる方法](http://qiita.com/okuta/items/f985b9da6de33a016a75)

`pip install chainer` でインストールできない場合

0. `pip uninstall chainer`で古いChainerを削除
0. `pip list` でパッケージ一覧を表示してChainerが表示されないことを確認
0. [Chainerソースコード](https://github.com/pfnet/chainer)のClone & `python setup.py install` でビルドする
0. 依存パッケージのnumpyのビルドに失敗する場合は`pip install numpy`でnumpyを個別にインストールした後、setup.pyの`    'numpy>=1.9.0',`の1行をコメントアウトして`python setup.py install`を行う
