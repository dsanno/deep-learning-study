# Numpy

## 統計処理と線形代数

### 統計処理

numpyを使って統計処理を行うことができる。例えば以下のメソッドを使える。

* `numpy.sum`: 合計
* `numpy.mean`: 平均
* `numpy.var`: 分散
* `numpy.std`: 標準偏差

```
>>> x = np.arange(10, dtype=np.float32)
>>>
>>> x.sum()
45.0
>>> np.sum(x)
45.0
>>> np.mean(x)
4.5
>>> np.var(x)
8.25
>>> np.std(x)
2.8722813
```

`axis`を指定することで特定の次元に沿った計算を行うこともできる。

```
>>> x = np.arange(9, dtype=np.float32).reshape((3,3))
>>> x
array([[ 0.,  1.,  2.],
       [ 3.,  4.,  5.],
       [ 6.,  7.,  8.]], dtype=float32)
>>> x.sum(axis=1)
array([  3.,  12.,  21.], dtype=float32)
```

### 線形代数

`numpy.linalg`パッケージを使うと線形代数の処理を行うことができる。詳しくは[Numpy ManualのLinear algebraの項](http://docs.scipy.org/doc/numpy/reference/routines.linalg.html)を参照すること。

`numpy.linalg.inv`を使って逆行列を求める。

```
>>> x = np.array([[1, 2], [3, 4]], dtype=np.float32)
>>> x
array([[ 1.,  2.],
       [ 3.,  4.]], dtype=float32)
>>> y = np.linalg.inv(x)
>>> y
array([[-2. ,  1. ],
       [ 1.5, -0.5]], dtype=float32)
>>> np.dot(x, y)
array([[ 1.,  0.],
       [ 0.,  1.]], dtype=float32)
```

`np.linalg.eig`を使って固有値、固有ベクトルを求める。

```
>>> e_value, e_vector = np.linalg.eig(x)
>>> e_value
array([-0.37228131,  5.37228155], dtype=float32)
>>> e_vector
array([[-0.82456481, -0.41597354],
       [ 0.56576747, -0.90937668]], dtype=float32)
>>>
>>> np.dot(x, e_vector[:,0])
array([ 0.30697012, -0.21062446], dtype=float32)
>>> e_vector[:, 0] * e_value[0]
array([ 0.30697006, -0.21062465], dtype=float32)
```
