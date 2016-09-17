# Numpy

## 配列の演算

### 要素ごとの演算

`+`,`-`,`*`,`/`などの演算子を使うと配列とスカラ、または配列の要素同士の演算を行うことができる。

```

>>> x = np.array([1, 2, 3, 4], dtype=np.float32)
>>> x
array([ 1.,  2.,  3.,  4.], dtype=float32)
>>>
>>> x + 2
array([ 3.,  4.,  5.,  6.], dtype=float32)
>>> x - 2
array([-1.,  0.,  1.,  2.], dtype=float32)
>>> x * 2
array([ 2.,  4.,  6.,  8.], dtype=float32)
>>> x / 2
array([ 0.5,  1. ,  1.5,  2. ], dtype=float32)
>>>
>>> y = np.array([2, 3, 4, 5], dtype=np.float32)
>>> y
array([ 2.,  3.,  4.,  5.], dtype=float32)
>>>
>>> x + y
array([ 3.,  5.,  7.,  9.], dtype=float32)
>>> x - y
array([-1., -1., -1., -1.], dtype=float32)
>>> x * y
array([  2.,   6.,  12.,  20.], dtype=float32)
>>> x / y
array([ 0.5       ,  0.66666669,  0.75      ,  0.80000001], dtype=float32)
```

`**`, `numpy.exp`, `numpy.log`, `numpy.sin`なども使える。

```
>>> x ** 2
array([  1.,   4.,   9.,  16.], dtype=float32)
>>> np.exp(x)
array([  2.71828175,   7.38905621,  20.08553696,  54.59814835], dtype=float32)
>>> np.log(x)
array([ 0.        ,  0.69314718,  1.09861231,  1.38629436], dtype=float32)
>>> np.sin(x)
array([ 0.84147096,  0.90929741,  0.14112   , -0.7568025 ], dtype=float32)
```

### Broadcasting

2つの配列の次元、要素数が異なっている場合でも配列の演算を行うことができる。
全ての次元について、以下が成立すればよい。

* 要素数の大きさが同じ
* どちらかの要素数が1

要素数が少ない側は要素数を多い側と同じになるようにbroadcastする。

詳しくは[Numpy ManualのBroadcastingの項](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)を参照すること。

```
>>> x = np.zeros((2,2,2), dtype=np.float32)
>>> y = np.array([[1, 2]], dtype=np.float32)
>>> x
array([[[ 0.,  0.],
        [ 0.,  0.]],

       [[ 0.,  0.],
        [ 0.,  0.]]], dtype=float32)
>>> y
array([[ 1.,  2.]], dtype=float32)
>>> y.shape
(1L, 2L)
>>> x + y
array([[[ 1.,  2.],
        [ 1.,  2.]],

       [[ 1.,  2.],
        [ 1.,  2.]]], dtype=float32)
```

### numpy.dot

`numpy.dot`を使うとベクトル同士の積、行列とベクトルの積、行列同士の積を計算できる。

引数が両方とも1次元の場合は内積を計算する。

```
>>> x = np.asarray([1, 2, 3], dtype=np.float32)
>>> y = np.asarray([4, 5, 6], dtype=np.float32)
>>> np.dot(x, y)
```

一方が行列の場合は行列の乗算となる。

```
>>> x = np.array([[1, 2], [3, 4]], dtype=np.float32)
>>> y = np.array([[2, 3], [4, 5]], dtype=np.float32)
>>> z = np.asarray([10, 20], dtype=np.float32)
>>>
>>> np.dot(x, y)
array([[ 10.,  13.],
       [ 22.,  29.]], dtype=float32)
>>> np.dot(x, z)
array([  50.,  110.], dtype=float32)
```

配列が3次元以上の場合は、1個目の配列の最後の軸の要素と、2個目の配列の最後から2番目の軸の要素をそれぞれ掛けた値の輪をとる。
例えば3次元同士の配列を使って`numpy.dot`を呼ぶと戻り値は4次元配列となるが、各要素は以下のようになる。

```
dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])
```

```
>>> x = np.arange(8, dtype=np.float32).reshape((2, 2, 2))
>>> x
array([[[ 0.,  1.],
        [ 2.,  3.]],

       [[ 4.,  5.],
        [ 6.,  7.]]], dtype=float32)
>>> y = np.arange(12, dtype=np.float32).reshape((3, 2, 2))
>>> y
array([[[  0.,   1.],
        [  2.,   3.]],

       [[  4.,   5.],
        [  6.,   7.]],

       [[  8.,   9.],
        [ 10.,  11.]]], dtype=float32)
>>>
>>> z = np.dot(x, y)
>>> z
array([[[[   2.,    3.],
         [   6.,    7.],
         [  10.,   11.]],

        [[   6.,   11.],
         [  26.,   31.],
         [  46.,   51.]]],


       [[[  10.,   19.],
         [  46.,   55.],
         [  82.,   91.]],

        [[  14.,   27.],
         [  66.,   79.],
         [ 118.,  131.]]]], dtype=float32)
>>> z.shape
(2L, 2L, 3L, 2L)
```
