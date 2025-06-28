# empty_space_estimation

## 概要

`empty_space_estimation` は、棚の情報を Gemini API に送信して棚の中の空き場所を得るための ROS パッケージです。Flask ベースの API サーバーを通して、画像・物体情報と Gemini の生成モデルを連携します。

---

# ライブラリの使い方

## 環境
Docker環境（geminiAPI用）
- **python3.10** (GeminiAPIがpython3.9以上対応のため)
- Docker & docker-compose
- Gemini API キー（`.env` 経由で設定）
- api, ioディレクトリをマウント
想定ROS環境（singularity）
- Python 3.8+
- ROS (catkin ワークスペース)

## セットアップ手順
### 1. インストール
```bash
git clone 
```

### 2. ROS パッケージのビルド
このときにcartographerでエラーが出ることがあるがその場合は一度rm -rf devel/ build/ logs/
```bash
cd ~/<your workspace>
catkin build empty_space_estimation
source devel/setup.bash
```
### 3. Docker環境のビルド
```bash
cd <your_path>/empty_space_estimation/env_docker
docker compose build
```
---
## 起動手順
### 基準画像取得の場合


[terminal 1]
```bash
$ cd ~/ros_ws
$ . 0_env.sh
$ source /entrypoint.sh
$ source 1_hsrb_settings.sh
$ source devel/setup.bash
$ roslaunch tam_hsr_utils bringup.launch joy_device_id:=0
```


[Terminal 2]
```bash
$ cd ~/ros_ws
$ . 0_env.sh
$ source /entrypoint.sh
$ source 1_hsrb_settings.sh
$ source devel/setup.bash
$  roslaunch navigation_start navigation.launch map_name:=<>
```

[Terminal 3]
```bash
$ cd ~/ros_ws/src/5_skills/space_estimation_server/env_docker
$ docker compose up
```

[Terminal 4]
```bash
$ cd ~/ros_ws
$ . 0_env.sh
$ source /entrypoint.sh
$ source 1_hsrb_settings.sh
$ source devel/setup.bash
$ rosrun space_estimation.launch
```


```
起動後、Flask API は http://localhost:5001/empty_space_estimation でリクエストを受け付けます．HSR接続時に使用する（Singularity など外部コンテナからアクセスする）場合は、localhost の代わりにホストのIPアドレス（例：192.168.0.10）でリクエストを受け付けます．

---

## .envファイルの設定

プロジェクトルートまたは `env_docker` ディレクトリに `.env` ファイルを作成し、以下のように記述してください：

```env
GEMINI_API_KEY=your_google_gemini_api_key
```

---


## 入出力まとめ
出力として
```
float32 x　#棚の中の空き場所のx座標
float32 y　#棚の中の空き場所のy座標
string frame_id　#どのフレームから見たIDか（基本はHSRの頭のカメラ）
```

### 備考


- 多くのパスやラベル名は `config/config.yaml` からカスタマイズ可能．

