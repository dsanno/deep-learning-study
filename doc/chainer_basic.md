# Chainerの基本的な使い方

Chainerに限らずニューラルネットワークのフレームワークは機能ごとにモジュール化されており、必要な機能を組み合わせて目的に沿ったニューラルネットワークを構築したり、学習を行ったりすることが容易になっている。

注意: ここの説明はTrainerを使用していないためChainer公式のMNISTデータセットの学習とは異なる。Trainerを使用しない理由は[ChainerのTrainerについて](chainer_trainer.md)を参照すること。

## 学習とは

ニューラルネットワークのパラメータを最適な値に調整することを指す。

## ニューラルネットワークを使った教師あり学習の流れ

### データセットを用意する

通常以下の3つを用意する。それぞれ入力データと教師データとがある。

* 学習データ (train data)  
パラメータ学習に使用するデータ
* validationデータ (validation data)  
どのiteration/epochでのモデルが最適であるか比較したり、ハイパーパラメータのどの値が適切かを比較するのに使用するデータ
* テストデータ (test data)  
学習後のモデルを評価するのに使用するデータ

### ニューラルネットワークとoptimizerを構築する

* ニューラルネットワークを構築する  
ニューラルネットワークの各層はモジュール化されており、モジュールを選択して結合することで容易に構築できるようになっている。必要ならパラメータをファイルから読み込む
* optimizerを構築する  
どのoptimizerを使用するか決め、ニューラルネットワークと関連付ける

### 学習

以下の処理を繰り返して学習を行う。以下の処理1回を1 iteration と呼ぶ。これに対して学習データを一通り使用するまでiterationを繰り返すことを1 epochと呼ぶ

* 学習データをニューラルネットワークに入力する
* ニューラルネットワークの出力と正解データから損失を計算する
* 損失から各パラメータの勾配を求める
* 勾配を基にパラメータを更新する

### 学習後のパラメータをファイルに保存する

学習後のパラメータを保存しておいて予測時に使用する

## 教師あり学習の詳細

### データセットを用意する

データセットごとに異なる。以下にMNISTデータの場合について述べる。

#### 読み込み

Chainerでは`chainer.datasets.get_mnist`を使うことでMNISTデータセットを取得することができる。
内部ではデータセットのダウンロードを行っているので初回呼び出し時にはインターネット環境に接続している必要がある。

以下のようにするとtrain_dataに学習データが、test_dataにテストデータが格納される。

```
> import chainer
> train_data, test_data = chainer.datasets.get_mnist()
```

学習データとvalidationデータを分離する場合には`chainer.datasets.split_dataset_random`を使う。

```
> train_data, valid_data = chainer.datasets.split_dataset_random(train_data, len(train_data) - 5000)
```

#### データの取得

`Iterator`を使うことで学習データ、テストデータを順に取得することができる。
`Iterator`については[Chainer Referance ManualのIterator examples](http://docs.chainer.org/en/stable/reference/iterators.html)を参照のこと。

```
> batch_size = 100
> train_iterator = chainer.iterators.SerialIterator(train_data, batch_size)
> batch = train_iterator.next()
```

`batch`はlistで、各要素は1枚の画像を表すnumpy配列と正解ラベルを表す整数とのtupleである。
これを複数画像のnumpy配列と正解ラベルのnumpy配列に変換するには`chainer.dataset.convert.concat_examples`を使う。

```
> from chainer.dataset import convert
> x, t = convert.concat_examples(batch)
```

### ニューラルネットワークを構築する

以下にFully Connected Layer 3層からならニューラルネットワークの例を挙げる。

```
class MLP(chainer.Chain):

    def __init__(self, n_in, n_out, n_hidden):
        # Networkの持つLinkを定義する
        # superクラスの__init__を使う方法と
        # add_link()メソッドで追加する方法とがある
        # Chainer1.12からL.Linear()の第1引数はNoneでよく、
        # その場合入力サイズは実際のデータから自動的に決まる
        super(MLP, self).__init__(
            l1=L.Linear(n_in, n_hidden),
            l2=L.Linear(n_hidden, n_hidden),
            l3=L.Linear(n_hidden, n_out),
        )

    def __call__(self, x, train=True):
        # Linkはcallableとなっており、関数として呼び出すとLinkの処理
        # L.LinearはWx + bを計算する(W, bはL.Linearが持つ重み行列とバイアス項)
        h = self.l1(x)
        # Dropoutを実行する
        # Dropoutは学習時と予測時とで挙動が異なるので、trainでどちらであるかを指定する必要がある
        h = F.dropout(h, 0.5, train=train)
        # ReLUを実行する
        h = F.relu(h)
        h = self.l2(h)
        h = F.dropout(h, 0.5, train=train)
        h = F.relu(h)
        h = self.l3(h)
        return h
```

クラスの定義は以下のようになる。

* `chainer.Chain`を継承したクラスを定義する
* `__init__`で必要な`link`を追加する。`link`はパラメータつきの層として機能し、ここで追加した`link`の持つパラメータが学習対象のパラメータとなる。
* `__call__`で各層をどう結合するかを定義する。

`__init__`と`__call__`の引数は自由に決めることができる。例えば`__init__`の引数に中間層の大きさを渡したり、`__call__`が複数の入力データを受け付けるようにすることができる。

以下のようにニューラルネットワークのインスタンスを生成する。

```
net = MLP(28 * 28, 10, 100)
```

### optimizerを構築する

optimizerの構築は、optimizerのインスタンスを生成して`setup`メソッドでニューラルネットワークと関連付けるだけである。
`chainer.optimizers`以下に複数の最適化アルゴリズムのモジュールがあり、使用したいものを選ぶことができる。

```
from chainer import optimizers

optimizer = optimizers.Adam()
optimizer.setup(net)
```

### 学習を行う

#### 学習データをニューラルネットワークに入力する

ニューラルネットワークに入力するには
* 入力のnumpy配列(またはcupy配列)から`chainer.Variable`インスタンスを生成する。(省略可)
* ニューラルネットワークインスタンスを関数として呼び出す。このとき`chainer.Variable`インスタンスをの引数とする。

```
y = net(chainer.Variable(x), train=True)
```

Chainer 1.10から以下のように`chainer.Variable`インスタンス生成を省略できるようになった。numpy配列を直接渡した場合には内部で`Variable`インスタンスを生成する。

```
y = net(x, train=True)
```

#### ニューラルネットワークの出力と正解データから損失を計算する

ここでは画像分類でよく使われる損失関数(loss function, cost function)を計算する。

```
from chainer import functions as F

loss = F.softmax_cross_entropy(y, chainer.Variable(t))
```

以下のように`chainer.Variable`の省略が可能である。

```
loss = F.softmax_cross_entropy(y, t)
```

#### 損失から各パラメータの勾配を求める

以下のように`cleargrads`メソッドでパラメータの勾配を初期化し、`backward`メソッドで勾配を計算する。

```
net.cleargrads()
loss.backward()
```

Chainer 1.14までは勾配初期化に`zerograds`メソッドを使用していたが1.15からdeprecatedとなった。

#### 勾配を基にパラメータを更新する

optimizerの`update`メソッドを呼ぶだけである。

```
optimizer.update()
```

### 学習後のパラメータをファイルに保存する

パラメータをファイルに保存するには`chainer.serializers`を使う。
ここでは`save_npz`メソッドを使って保存する例を挙げる。

```
from chainer import serializers

serializers.save_npz('mnist.model', net)
```

`save_npz`メソッドで保存したファイルは`load_npz`メソッドで読み込むことができる。

```
serializers.load_npz('mnist.model', net)
```

`serializers`には他にも保存・読み込みを行うためのメソッドがある。
詳細は[Chainer Reference ManualのSerializersの項](http://docs.chainer.org/en/stable/reference/serializers.html)を参照すること。

## ニューラルネットワークを使った予測

### ニューラルネットワークを構築する

学習時と同様にニューラルネットワークのインスタンスを生成した後、ファイルからパラメータを読み込む。

```
from chainer import serializers

net = MLP(28 * 28, 10, 100)
serializers.load_npz('mnist.model', net)
```

### 入力データを生成する

画像を読み込み、numpy配列(またはcupy配列)に変換する。ここではPillowを使って画像を読み込んでいる。注意する点として、学習データの値の範囲が0～1で、背景が黒(0)なので、予測時の入力データも同様になるように変換する必要がある。

複数の画像を一度に入力することも可能である。その場合入力となる配列のshapeは(画像枚数, 画像のピクセル数)となる

```
import numpy as np
from PIL import Image

image = Image.open('sample.png').convert('L').reseize((28, 28), Image.BILINEAR)
image = 1 - np.asarray(image).astype(np.float32) / 255
image = image.reshape((1, -1))
```

### ニューラルネットワークに入力する

ニューラルネットワークインスタンスに入力となる配列を渡す。今回使用する`MLP`クラスは、引数として学習時かどうかを判別する`train`をとるが、今は予測時なので`False`を渡す。

`chainer.Variable`の`volatile=True`は出力が呼び出された`function`への参照を持たないことを意味する。`volatile`を有効にすると`function`の呼び出しを逆順にたどれなくなるのでバックプロパゲーションができなくなるが、消費メモリ量は減る。

```
y = net(chainer.Variable(image, volatile=True), train=False)
```

以下のように`chainer.Variable`を省略することが可能だが、無駄なメモリ消費を抑えるために`Variable`を使用して`volatile`を有効にしたほうが良い。

```
y = net(image, train=False)
```

### Softmax値を求める

出力結果としてどのラベルが選ばれたかを知るためにはニューラルネットワークのどの出力が最大であるかわかればよいが、出力の意味をわかりやすくするためにSoftmax値を求める。
Softmax値を求めるには`chainer.functions.softmax`を使用する。
出力`Variable`で、`data`プロパティを参照することでnumpy配列(またはcupy配列)が得られる。

```
from chainer import functions as F

result = F.softmax(y).data
```
