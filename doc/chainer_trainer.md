# ChainerのTrainerについて

Chainerは1.11.0からTrainerを使って学習ループを抽象化しているが、以下のような問題があるので、ここでは使用しない。

## 機能が不足している

以下のようにTrainerの機能が不足している。
TriggerやExtensionの実装を行えば対応可能だが、Trainerを使わずに自前でループを実装したほうが実装コストが低くなり、Trainerを使うメリットがなくなる。

### 学習率のスケジューリングが困難

Deep Learningの学習ではよく「100epochごとに学習率を0.1倍する」などといった学習率のスケジューリングを行うのだが、
[ExponentialShift](http://docs.chainer.org/en/stable/reference/extensions.html#exponentialshift)、[LinearShift](http://docs.chainer.org/en/stable/reference/extensions.html#linearshift)といったクラスだと1iterationごとにしか学習率を変更できないので不便である。より柔軟なスケジューリングを行うためにはこれらのクラスをTriggerで制御することが必要となる。

また「100, 150, 200epochで学習率を0.1倍する」、「しばらくvalidation dataの精度が上がらなかったら学習率を下げる」などのように複雑な条件もあるので新たなTriggerの実装も必要となる。

### Extensionからmetricsを簡単に取得できない

Trainerでは1epochごとのloss、accuracyなどのmetricsの計算を[LogReport](http://docs.chainer.org/en/stable/reference/extensions.html#logreport)が行っているが、他のExtensionから参照できないようになっている。
Extensionで1epochごとのaccuracyが必要になる場合には、Extension内でLogReportを持ったり、LogReportと同じ処理をする必要があり、LogReportが重複する構造になってしまう。

### metricsのファイル保存、グラフ化

metricsをファイルに保存したり、グラフとして表示したりしたい。

## Classifierという謎の役割を持ったクラスがある

Chainerのmnist、ptbのexampleを見ると`chainer.links.Classifier`というクラスを使っているが、以下のようにどのような役割を持っているのか不明である。

* Classifierの要件が不明。Classifierという名前から分類タスクで使用するものと思われるが、他のタスクを行うためにどのような実装をすればよいのかまとまった説明がない。
* `chainer.links`の下にある低レイヤーモジュールなのに、比較的高レイヤーの`chainer.reporter`を参照している。
