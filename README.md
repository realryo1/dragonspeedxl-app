# dragonspeedxl-app

SDXL（LCM、Base 2 step ＋ Hires 4 step）を、Android 端末の NPU（Hexagon HTP v81）と GPU（Adreno）だけで動かすアプリです。生成は端末の中で完結し、オフラインで動きます。Kotlin / Jetpack Compose（Material 3、壁紙の動的カラー）と、C++ の QNN エンジン（JNI）でできています。

- 対応端末: Snapdragon 8 Elite Gen 5（SM8850、HTP v81）。arm64-v8a のみ。minSdk 31 / targetSdk 34。
- モデル: [Hugging Face の `realryo1/dragonspeedxl-models`](https://huggingface.co/realryo1/dragonspeedxl-models)（public、CC BY-NC-SA 4.0。配布条件は同じページに書いてあります）。
- 変換の再現に使う道具は [`realryo1/dragonspeedxl-convert`](https://huggingface.co/realryo1/dragonspeedxl-convert) にあります。

## はじめに（初回）

初回に起動すると、生成に必要なモデルを取る画面が出ます。取り終えるまで、本体の画面は出ません。

- 取るもの: 約 7.6GB（UNet、CLIP、VAE、GPU 用の DLC など）。初回だけです。
- 途中で止めても、次回は続きから取ります（`.part` ファイルから再開）。
- 取った各ファイルは sha256 を確かめてから置きます。壊れたものは使いません。
- 進み具合は通知に出ます。アプリを閉じても、取得は続きます（Android 13 以降は通知の許可を求めます。断っても取得は続きますが、通知は出ません）。
- 取り終えたら「ロード」を押して、モデルを読み込みます。

手動で置く場合は、端末のアプリのフォルダ `models/` に同じ名前のファイルを置けば、取得の画面は出ません（sha256 の一覧は HF の `manifest.json` にあります）。

## 機能

- **生成**: 縦（720×1280 → 1080×1920）と横の両方。Base（2 step）と Hires（タイル、4 step）。生成中のプレビュー（TAESD）。速度優先のモード（TAESD で最終画像を出す）。
- **LoRA**: 端末の中で取り込みます（Civitai の URL かファイル）。取り込みは裏で進み、通知に出ます。
- **領域の Hires（refine）**: 生成した画像の一部を、細部を明瞭にして作り直します。
- **履歴**: 生成結果は端末の中に残ります（画像と一覧）。
  - **書き出し**: 設定の「履歴」から、zip（`index.json` と `images/<id>.png`）に書き出せます。画像の PNG には生成情報（プロンプト、seed など）が入っています。
  - **読み込み**: 書き出した zip を読み込み、足りない項目を履歴に足せます。同じ id の項目は飛ばします（上書きしません）。
- **ギャラリーに保存**: 画像を Pictures/dragonspeedXL に保存できます（生成情報つき）。
- **セキュリティ**: アプリのロック（パスワードか指紋認証）。ロック中は、アプリスイッチャーの縮小画像に中身を残しません。

## 通知

- モデルの取得、LoRA の取り込み、履歴の書き出し・読み込みは、前景サービスで裏で続きます。進み具合は「ダウンロード」の通知に出ます（1 秒に 1 回、または ％が変わったときだけ更新）。
- 完了や失敗は、通知の表示を完了のメッセージに変えて残します（タップでアプリを開きます）。

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
./gradlew :app:installDltest      # 別アプリ（「dragonspeedXL テスト」、applicationId は .dltest）として入れる。本体とは別のデータ
```

マニフェストの `uses-native-library`（`libcdsprpc.so`、`libadsprpc.so`、`libOpenCL.so`）は、QNN の HTP と GPU の初期化に要ります。外すと `QnnDevice_create` が失敗します。

## CI（GitHub Actions）

`Actions` の「build-apk」を手動で実行します（`workflow_dispatch`）。

- QNN のヘッダーと `.so` は、非公開の補助リポジトリ `realryo1/dragonspeedxl-qnn-deps` から取ります。読むには、secret `DEPS_TOKEN`（そのリポジトリだけを読める fine-grained PAT）が要ります。
- `tag` を入れると、リリース鍵で署名した APK（`app-release.apk`）を GitHub Release に上げます。空なら artifact だけ。
- 署名の鍵は secrets（`KEYSTORE_BASE64`、`KEYSTORE_PASSWORD`、`KEY_ALIAS`、`KEY_PASSWORD`）から復元します。鍵そのものはリポジトリには入れません。
- 手元で同じ署名のビルドをするときは、環境変数 `DSXL_KEYSTORE`（鍵ファイルのパス）、`DSXL_KEYSTORE_PASSWORD`、`DSXL_KEY_ALIAS`、`DSXL_KEY_PASSWORD` を設定して `./gradlew :app:assembleRelease` を実行します。

## 構成

```
app/src/main/
  java/com/dragonspeedxl/app/   Kotlin
    MainViewModel.kt            画面の状態と操作（生成、LoRA、履歴、モデルの取得）
    download/                   モデルの取得（ModelManifest、ModelDownloader）と通知（DownloadService、DownloadStatus）
    data/                       履歴（HistoryStore。画像と index.json、書き出し・読み込み）
    lora/                       LoRA の取り込み（端末の中で変換）
    ui/                         画面（MainScreen、SettingsDialog、ModelSetupScreen など）
    engine/ prompt/ security/   生成のエンジン、プロンプトの補完（Danbooru）、ロック
  cpp/                          C++ エンジン（engine/pipeline、engine/qnn_model、engine/preview、engine/refine、jni_bridge）
  assets/                       model_manifest.json（取得するファイルの一覧と sha256）、ほか
  res/                          リソース（ic_logo など）
```

## ライセンス

ソースのライセンスは未定です（公開前に決めます）。モデルと、QNN の SDK は、それぞれの条件に従ってください。
