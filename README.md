# dragonspeedxl-app

SDXL（LCM、Base 2 step ＋ Hires 4 step）を、Android 端末の NPU（Hexagon HTP v81）と GPU（Adreno）だけで動かすアプリです。Kotlin / Jetpack Compose（Material 3、壁紙の動的カラー）と、C++ の QNN エンジン（JNI）でできています。

- 対応端末: Snapdragon 8 Elite Gen 5（SM8850、HTP v81）。arm64-v8a のみ。minSdk 31 / targetSdk 34。
- モデル: [Hugging Face の `realryo1/dragonspeedxl-models`](https://huggingface.co/realryo1/dragonspeedxl-models)（public、CC BY-NC-SA 4.0。配布条件は同じページに書きます）。変換スクリプトは同じ作者の `dragonspeedxl-convert`（予定）。
- 変換の再現に使う道具と、モデルの配布条件は、それぞれのリポジトリにあります。

## 状態

- モデルのダウンロード機能は**まだ入っていません**（モデルは手動で `models/` に置くか、PC の変換で作ります）。HF からの取得（進捗、再開、sha256 の検証）は次の作業です。
- 生成の画面、LoRA の取り込み（Civitai の URL かファイル）、プレビュー、領域の Hires（refine）は動きます。

## ビルドの準備（公開リポジトリには含みません）

QNN（Qualcomm AI Runtime）の SDK 由来のものは、このリポジトリに入れていません。配布の条件は、SDK の LICENSE.pdf に従います（object code を、アプリに組み込んだ形でだけ配布できる）。自分で SDK を入手して、次の場所に置いてください。

1. **ヘッダー**: `third_party/qnn/include/`（QNN のヘッダー。配布は許されていないので、リポジトリには入れない）。
2. **ランタイム**: `app/src/main/jniLibs/arm64-v8a/` に次の 5 つ。
   - `libQnnHtp.so`、`libQnnHtpV81Stub.so`、`libQnnSystem.so`、`libQnnGpu.so`（`lib/aarch64-android/`）
   - `libQnnHtpV81Skel.so`（`lib/hexagon-v81/unsigned/`）

開発に使ったバージョンは次のとおりです。

- NDK 30.0.16248370、CMake 4.1.2、Gradle 8.9（wrapper）、AGP 8.5.2、JDK 21。

```
./gradlew assembleDebug           # Windows は gradlew.bat
./gradlew :app:installDebug       # 端末を USB 接続して入れ替え
```

マニフェストの `uses-native-library`（`libcdsprpc.so`、`libadsprpc.so`、`libOpenCL.so`）は、QNN の HTP と GPU の初期化に要ります。外すと `QnnDevice_create` が失敗します。

## 構成

```
app/src/main/
  java/com/dragonspeedxl/app/   Kotlin（MainViewModel、ui/、engine/、data/、lora/、prompt/、security/）
  cpp/                          C++ エンジン（engine/pipeline、engine/qnn_model、engine/preview、engine/refine、jni_bridge）
  assets/ res/                  リソース
```

## ライセンス

ソースのライセンスは未定です（公開前に決めます）。モデルと、QNN の SDK は、それぞれの条件に従ってください。
