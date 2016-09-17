# Numpy

[公式ドキュメント](http://www.numpy.org/)

## Numpyとは

Pythonの数値計算用パッケージで、以下の機能をもつ。

* 多次元配列操作
* 線形代数

内部はC/C++とFortranで実装されており、高速に動作する。また配列の中身とインデックスとを別々に管理しており、効率的なメモリ管理を行っている。

Numpyを効率よく使用する方法については以下が参考になる。

http://ipython-books.github.io/featured-01/

## 使い方

まずnumpyをimportする。`np`をエイリアスとすることが多い。

```
>>> import numpy as np
```

### numpy.xxxとnumpy.ndarray.xxxについて

Numpyのドキュメントを読むと`numpy.xxx`と`numpy.ndarray.xxx`とがあることがわかる。
以降では`numpy.xxx`について説明するが、通常同じ機能を持つメソッドがnumpy配列のインスタンスメソッド`numpy.ndarray.xxx`として定義されている。

```
>>> x = np.array([1,2,3], np.float32)
>>> x
array([ 1.,  2.,  3.], dtype=float32)
>>> np.sum(x)
6.0
>>> x.sum()
6.0
```

## 配列の生成

### numpy.array

listまたはnumpy配列を元に配列を生成する。

```
>>> x = np.array([[1,2],[3,4]], dtype=np.float32)
>>> x
array([[ 1.,  2.],
       [ 3.,  4.]], dtype=float32)
```

`dtype`で要素の型を指定する。Chainerを使用する時には通常整数は`np.int32`、浮動小数は`np.float32`を指定する。

`dtype`を指定しない場合は、元のnumpy配列またはlistの型になる。

### numpy.asarray

`numpy.asarray`とほとんど同じだが、生成する配列が入力した配列と同じ場合にはコピーを生成しない。

```
>>> x = np.array([1,2,3], dtype=np.float32)
>>> y = np.asarray(x)
>>> x is y
True
```

### numpy.copy

配列のコピーには`numpy.copy`を使う

```
>>> x = np.array([1,2,3], dtype=np.float32)
>>> x
array([ 1.,  2.,  3.], dtype=float32)
>>> y = np.copy(x)
>>> y
array([ 1.,  2.,  3.], dtype=float32)
>>> x is y
False
```

### numpy.zeros, numpy.ones, numpy.full

`numpy.zeros`を使うと要素を0で埋めた配列を生成できる。

```
>>> x = np.zeros((2,3), dtype=np.float32)
>>> x
array([[ 0.,  0.,  0.],
       [ 0.,  0.,  0.]], dtype=float32)
```

`numpy.ones`を使うと要素を1で埋めた配列を生成できる。

```
>>> x = np.ones((2,3), dtype=np.float32)
>>> x
array([[ 1.,  1.,  1.],
       [ 1.,  1.,  1.]], dtype=float32)
```

`numpy.full`を使うと要素を指定した値で埋めた配列を生成できる。

```
>>> x = np.full((2, 3), 10, dtype=np.float32)
>>> x
array([[ 10.,  10.,  10.],
       [ 10.,  10.,  10.]], dtype=float32)
```

### numpy.zeros_like, numpy.ones_like, numpy.full_like

別の配列と同じ形状で中身を0, 1, または指定した値の配列を生成できる。

```
>>> a = np.asarray([[1,2],[3,4]], dtype=np.float32)
>>> a
array([[ 1.,  2.],
       [ 3.,  4.]], dtype=float32)
>>> x = np.zeros_like(a)
>>> x
array([[ 0.,  0.],
       [ 0.,  0.]], dtype=float32)
>>> x = np.ones_like(a)
>>> x
array([[ 1.,  1.],
       [ 1.,  1.]], dtype=float32)
>>> x = np.full_like(a, 10)
>>> x
array([[ 10.,  10.],
       [ 10.,  10.]], dtype=float32)
```

### numpy.arange

等差数列を生成する。開始、終了、間隔を指定することができる。開始位置は省略すると0、間隔は省略すると1になる。

```
>>> x = np.arange(5, dtype=np.int32)
>>> x
array([0, 1, 2, 3, 4])
>>>
>>> x = np.arange(3, 9, 2, dtype=np.float32)
>>> x
array([ 3.,  5.,  7.], dtype=float32)
```
