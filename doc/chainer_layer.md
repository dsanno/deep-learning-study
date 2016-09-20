# Layerについて

Deep Learningでは様々なLayerを組み合わせる必要がある。ここでは使用頻度の高いLayerの特徴を説明する。

## chainer.functionsとchainer.linksの違い。

`chainer.functions`はパラメータを持たないLayerを集めたパッケージ、`chainer.links`はパラメータを持つLayerを集めたパッケージである。`chainer.links`にある機能のパラメータを学習対象とせずに自分で指定することもできる。その場合は`chainer.functions`にある同等の機能を使用する。例えば`chainer.links.Liner`のパラメータを指定したい場合には`chainer.functions.linear`を使う。

## Connection

線形変換を行うLayerについて説明する。

### Fully Connected (links.Linear)

入力と出力が全結合したLayerである。
パラメータとして行列`W`とバイアスベクトル`b`を持ち、入力`x`に対して`Wx + b`が出力となる。

### 2 Dimension Convolution (links.Convolution2D)

2次元Convolutional Layerである。

### N Dimension Convolution (links.ConvolutionND)

N次元Convolutional Layerである。

### Embed ID (links.EmbedID)

整数値をベクトルに変換するのに使用する。例えば自然言語処理で単語IDをベクトルに変換する時に使用する。IDとベクトル値との対応が学習対象となる。

## Activation Function

Activation Function (活性化関数)となるLayerについて説明する。

### Rectified Linear Unit (ReLU) (functions.relu)

Rectified Linear Unit (ReLU)を実行する。`x`を入力として`max(0, x)`で定義される。

### Tanh (functions.tanh)

tanhを実行する。形状はSigmoidと同じだが、出力の範囲が-1～1である点が異なる。使用頻度はReLUに比べると少ないが、出力の範囲を限定したい場合に使われる。

### Sigmoid (functions.sigmoid)

Sigmoid関数を実行する。最近はあまり使われない。

### Softmax (functions.softmax)

Softmax値を計算する。

### Log Softmax (functions.log_softmax)

Softmax値のlogをとる。単純に`functions.softmax`の出力のlogをとるのでは、`softmax`の出力が0に丸められることがあるので支障がある。softmaxのlogをとる場合は`log_softmax`を使う方が安全である。

### Exponential Linear Unit (ELU) (functions.elu)

ReLUは入力が負の時に勾配が消失するというデメリットがあり、それを改善するために導入された。出力は入力を`x`、`alpha`を定数として`x >= 0`のとき`x`、`x < 0`のとき`alpha * (exp(x) - 1)`となる。使用頻度は低い。

### Leaky ReLU (functions.leaky_relu)

ELUと同じくReLUの勾配消失問題を改善するために導入された。`x`を入力、`alpha`を0～1の範囲にある定数として、出力は`max(alpha * x, x)`となる。使用頻度は低い。

## Normalization

データにノイズを加えるLayerについて説明する。ノイズを加えることで過学習を防ぎ汎化性を高めることができる。

### Dropout (functions.dropout)

学習時に出力の一部を確率的に0にする。0にしない箇所の出力は確率に応じて大きくする。0にする確率を`p`とする場合、出力は`1 / (1 - p)`倍にする。予測時には出力の操作は行わない。
使用頻度は高く、Fully Connected LayerまたはConvolutional Layerの直後で使われることが多い。

## Normalization

Normalizationを行うことで学習速度や精度を向上させることができる。

### Batch Normalization (links.BatchNormalization)

出力をミニバッチ内で平均0, 分散1にすることでその後のLayerの学習を行いやすくする。
Convolutional Layerの直後に挿入することが多い。
使用頻度は非常に高く、CNNを使う場合導入はほぼ必須といえる。

## Spatial Pooling

2 x 2, 3 x 3など、空間的に近い領域の出力の平均または最大値をとって出力の数を減らすことをPoolingと呼ぶ。

### Max Pooling (functions.max_pooling_2d)

局所領域の最大値を出力する。画像認識では、複数のCNNの後にMax Poolingを挿入するというのを何回か繰り返すことが多い。

### Average Pooling (functions.average_pooling_2d)

局所領域の平均値を出力する。画像認識では最終出力の数段前にAverage Poolingを挿入して、幅・高さ方向の全平均をとる手法がある。

## 損失関数

使用頻度の高い損失関数が用意されている。

### Softmax Cross Entropy (functions.softmax_cross_entropy)

Softmax Cross Entropyを計算する。多値分類でよく使われる。出力は求めたSoftmax Cross Entropyの平均となる。

### 平均二乗誤差 (functions.mean_squared_error)

平均二乗誤差を計算する。回帰でよく使われる。

## 配列操作

numpyに備わっている配列操作の一部を`chainer.functions`でも行うことができる。

### 連結 (functions.concat)

2つの配列を連結する。

### reshape (functions.reshape)

配列のreshapeを行う。

### 行列積 (functions.matmul)

2つの行列の積を求める。

### バッチごとの行列積 (functions.matmul)

バッチごとに2つの行列の積を求める。配列`a`と配列`b`とがあるとき、`c = functions.batch_matmul(a, b)`とすると、i = 0, 1, 2, ...について`c[i]`は`a[i]`と`b[i]`の積となる。

## 数値計算

### 演算子

numpy配列の計算のように`a + b - c`、`a ** b`といった記述で演算子を使用できる。以下の演算子を使用できる。

* 正負の反転: `-a`
* 加算: `a + b`
* 減算: `a - b`
* 乗算: `a * b`
* 除算: `a / b`
* べき乗: `a ** b`
* 絶対値: `abs(a)`
* 行列積(Python 3.5以降): `a @ b`  
`chainer.functions.matmul`でも実行可能なのであえて互換性のない`@`を使用する必要はない。

### 合計 (functions.sum)

合計値を計算する。特定の軸に沿った合計値を出力することもできる。
