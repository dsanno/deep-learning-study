# 強化学習

OpenAI Gymが提供する環境で強化学習を行う。
以下の機能をサポートしている。

* Q学習
* "CartPole-v0"と"MountainCar-v0"の実行
* Experience Replay

## 参考にした実装

https://github.com/jaara/ai_examples

## 必要な環境

* [OpenAI Gym](https://gym.openai.com/docs)
`pip install gym`でインストールできる。

## 実行方法

### ソースディレクトリ

(rood dir)/src/open_ai_gym

### 学習

```
cd src/open_ai_gym
python train.py
```

オプション:
* `-e <environmant>` : 実行するEnvironment("cart_pole"または"mountain_car")(default: "cart_pole")
* `-s <skip size>` : 描画をスキップするエピソード数。描画をスキップすることで高速化する(default: 0)
* `-b <batch size>` : 学習時ミニバッチサイズ(default: 32)
* `-p <pool size>` : Experience Replay用に保持するiteration数(default: 2000)
* `-t <train iteration>` : 一度に学習する回数(default: 10)
* `--episode <episode num>` : 実行するエピソード数(default 1000)
* `--use-double-q` : 指定するとDoule Q-learningを使用する
