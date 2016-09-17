# Numpy

## 配列の変形と連結

### shapeとnumpy.reshape

配列の形状はshapeプロパティを参照することで得られる

```
>>> x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
>>> x
array([[ 1.,  2.,  3.],
       [ 4.,  5.,  6.]], dtype=float32)
>>> x.shape
(2L, 3L)
```

`numpy.reshape`を使うとshapeを変更できる。変更後の総要素数が一致しない場合にはエラーとなる。

```
>>> np.reshape(x, (3, 2))
array([[ 1.,  2.],
       [ 3.,  4.],
       [ 5.,  6.]], dtype=float32)
```

shapeの1つに負の値を指定すると自動的に計算される。

```
>>> np.reshape(x, (-1, 2))
array([[ 1.,  2.],
       [ 3.,  4.],
       [ 5.,  6.]], dtype=float32)
```

### numpy.transpose

`numpy.transpose`を使うと軸を入れ替えることができる。

```
>>> x = np.arange(12, dtype=np.float32).reshape((2, 2, 3))
>>> x
array([[[  0.,   1.,   2.],
        [  3.,   4.,   5.]],

       [[  6.,   7.,   8.],
        [  9.,  10.,  11.]]], dtype=float32)
>>> x.shape
(2L, 2L, 3L)
>>> y = np.transpose(x, (2, 1, 0))
>>> y
array([[[  0.,   6.],
        [  3.,   9.]],

       [[  1.,   7.],
        [  4.,  10.]],

       [[  2.,   8.],
        [  5.,  11.]]], dtype=float32)
>>> y.shape
(3L, 2L, 2L)
```

### numpy.swapaxes

`numpy.swapaxes`を使うと2つの軸を入れ替えることができる。

```
>>> x = np.arange(12, dtype=np.float32).reshape((2, 2, 3))
>>> x.shape
(2L, 2L, 3L)
>>> y = np.swapaxes(x, 1, 2)
>>> y
array([[[  0.,   3.],
        [  1.,   4.],
        [  2.,   5.]],

       [[  6.,   9.],
        [  7.,  10.],
        [  8.,  11.]]], dtype=float32)
>>> y.shape
(2L, 3L, 2L)
```

### numpy.concatenate

`numpy.concatenate`を使うと複数の配列を特定の軸に沿って連結することができる。

```
>>> x = np.array([[1, 2], [3, 4]], dtype=np.float32)
>>> y = np.array([[5, 6], [7, 8]], dtype=np.float32)
>>>
>>> z = np.concatenate((x, y), axis=1)
>>>
>>> x
array([[ 1.,  2.],
       [ 3.,  4.]], dtype=float32)
>>> y
array([[ 5.,  6.],
       [ 7.,  8.]], dtype=float32)
>>>
>>> z
array([[ 1.,  2.,  5.,  6.],
       [ 3.,  4.,  7.,  8.]], dtype=float32)
```
