plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "com.dragonspeedxl.app"
    compileSdk = 34
    ndkVersion = "30.0.16248370"

    defaultConfig {
        applicationId = "com.dragonspeedxl.app"
        minSdk = 31
        targetSdk = 34
        versionCode = 1
        versionName = "0.1.0"

        ndk { abiFilters += "arm64-v8a" }
        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++17"
                // C++ を使う .so は libdsxl_native.so だけ（QNN の .so は libc++ に依存しない）なので、静的に入れる。
                // libc++_shared.so（約 1.4MB）を APK に入れず、使う部分だけがリンクされる
                arguments += listOf("-DANDROID_STL=c++_static")
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "4.1.2"
        }
    }

    // リリースの署名。鍵は環境変数で渡す（DSXL_KEYSTORE: 鍵ファイルのパス、DSXL_KEYSTORE_PASSWORD、DSXL_KEY_ALIAS、DSXL_KEY_PASSWORD）。
    // 渡されていないときは debug 署名のまま（手元の assembleRelease が動くように）。CI では secrets から渡す。
    signingConfigs {
        create("release") {
            System.getenv("DSXL_KEYSTORE")?.let { path ->
                storeFile = file(path)
                storePassword = System.getenv("DSXL_KEYSTORE_PASSWORD")
                keyAlias = System.getenv("DSXL_KEY_ALIAS")
                keyPassword = System.getenv("DSXL_KEY_PASSWORD")
            }
        }
    }

    buildTypes {
        release {
            // R8 で使っていないクラスとリソースを除く。名前は変えない（JNI・エラー文のクラス名・履歴の enum 名がそのまま残る。proguard-rules.pro）
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            // 本体（デバッグ版の com.dragonspeedxl.app）と別アプリとして入る。データは別々（モデルも履歴も別）。
            applicationIdSuffix = ".release"
            signingConfig = if (System.getenv("DSXL_KEYSTORE") != null) signingConfigs.getByName("release") else signingConfigs.getByName("debug")
        }
        // 別アプリとして入れる試験用（applicationId が別なので、本体のデータ・モデルには触れない）。installDltest で入れる。
        create("dltest") {
            initWith(getByName("debug"))
            applicationIdSuffix = ".dltest"
            resValue("string", "app_name", "dragonspeedXL テスト")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    buildFeatures { compose = true }

    // LoRA の変換（lora/LoraConverter.kt）が読む layout.json と attach_map.json は、PC 版（loraconverter/data）と同じものを使う（写しを持たない）
    sourceSets.getByName("main").assets.srcDirs("src/main/assets", "../../loraconverter/data")

    packaging {
        jniLibs {
            // Skel（libQnnHtpV8xSkel.so）を実ファイルとして展開する（plan Phase 5）
            useLegacyPackaging = true
        }
    }
}

dependencies {
    val composeBom = platform("androidx.compose:compose-bom:2024.09.00")
    implementation(composeBom)
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.activity:activity-compose:1.9.2")
    implementation("androidx.biometric:biometric:1.1.0")   // 指紋認証ロック（BiometricPrompt は FragmentActivity が必要）
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.6")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.6")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.material3:material3")
    // アイコンは core の分だけ使う。extended（dex で約 30MB）にしか無い 2 つは ui/AppIcons.kt に写した
    implementation("androidx.compose.material:material-icons-core")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
    implementation("com.google.zxing:core:3.5.3")   // QR の復号（LoRA の Civitai URL の読み取り）。純 Java で軽い。カメラは Camera2 で直接使う
    debugImplementation("androidx.compose.ui:ui-tooling")
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")   // JVM のテストでは Android の org.json が使えないので、本物を入れる
}
