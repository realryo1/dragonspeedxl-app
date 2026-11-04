#include <jni.h>

#include <android/log.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "engine/pipeline.h"
#include "engine/util.h"

namespace {
std::unique_ptr<dsxl::Pipeline> g_pipe;
// 直近のエラー文。Begin と Await が別スレッドで重なるので、スレッドごとに持つ（Kotlin は同じスレッドで続けて読む）
std::string& g_err() { thread_local std::string e; return e; }
std::mutex g_mu;

std::string ToStd(JNIEnv* env, jstring s) {
    const char* c = env->GetStringUTFChars(s, nullptr);
    std::string out(c ? c : "");
    if (c) env->ReleaseStringUTFChars(s, c);
    return out;
}
// head 個の値のあとに、w×h 画素の ARGB を並べた Java の int 配列を作る。一時バッファを作らず、Java の配列に直接書く。
// argb（VAE が作ったもの）があればコピーだけ、無ければ rgb から変換する
jintArray ArgbArray(JNIEnv* env, std::initializer_list<jint> head, int w, int h, const std::vector<int32_t>* argb, const uint8_t* rgb) {
    const size_t n = static_cast<size_t>(w) * h, off = head.size();
    jintArray arr = env->NewIntArray(static_cast<jsize>(off + n));
    if (!arr) return nullptr;
    jint* buf = static_cast<jint*>(env->GetPrimitiveArrayCritical(arr, nullptr));
    if (!buf) return nullptr;
    std::copy(head.begin(), head.end(), buf);
    if (argb && argb->size() == n) std::memcpy(buf + off, argb->data(), n * sizeof(jint));
    else dsxl::RgbToArgb(rgb, buf + off, n);
    env->ReleasePrimitiveArrayCritical(arr, buf, 0);
    return arr;
}

// [w, h, ARGB...]
jintArray ToJava(JNIEnv* env, const dsxl::Image& img) { return ArgbArray(env, {img.w, img.h}, img.w, img.h, &img.argb, img.rgb.data()); }

// 生成中のプレビュー用リスナー（Kotlin の NativeEngine.PreviewListener.onPreview(w, h, ARGB[])）。呼ぶのはネイティブのワーカースレッドなので、そのたびに JVM に接続する。
JavaVM* g_vm = nullptr;
jobject g_preview_listener = nullptr;   // グローバル参照
jmethodID g_preview_mid = nullptr;
std::mutex g_preview_mu;

void CallPreview(int w, int h, const std::vector<uint8_t>& rgb) {
    std::lock_guard<std::mutex> lock(g_preview_mu);
    if (!g_vm || !g_preview_listener) return;
    JNIEnv* env = nullptr;
    bool attached = false;
    if (g_vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) == JNI_EDETACHED) {
        if (g_vm->AttachCurrentThread(&env, nullptr) != JNI_OK) return;
        attached = true;
    }
    if (jintArray arr = ArgbArray(env, {}, w, h, nullptr, rgb.data())) {
        env->CallVoidMethod(g_preview_listener, g_preview_mid, static_cast<jint>(w), static_cast<jint>(h), arr);
        env->DeleteLocalRef(arr);
    }
    if (env->ExceptionCheck()) env->ExceptionClear();
    if (attached) g_vm->DetachCurrentThread();
}

dsxl::ProgressFn MakeProgress(JNIEnv* env, jobject listener) {
    jclass cls = env->GetObjectClass(listener);
    jmethodID mid = env->GetMethodID(cls, "onProgress", "(III)V");
    return [env, listener, mid](dsxl::PStage s, int step, int total) { env->CallVoidMethod(listener, mid, static_cast<jint>(s), step, total); };
}
}  // namespace

extern "C" {

// 配置状況の表示用（必須モデル / QNN ライブラリ 4 つ / GPU 用 VAE）
JNIEXPORT jstring JNICALL
Java_com_dragonspeedxl_app_engine_NativeEngine_nativeBackendInfo(JNIEnv* env, jobject, jstring model_dir, jstring lib_dir) {
    std::string md = ToStd(env, model_dir), ld = ToStd(env, lib_dir);
    std::string miss = dsxl::Pipeline::MissingFiles(md);
    int libs = 0;
    for (const char* f : {"libQnnHtp.so", "libQnnSystem.so", "libQnnHtpV81Stub.so", "libQnnHtpV81Skel.so"}) libs += dsxl::FileExists(ld + "/" + f) ? 1 : 0;
    const bool gpu = dsxl::Pipeline::HasGpuVae(md, ld);
    std::string s = "QNN lib: " + std::to_string(libs) + "/4 / モデル: " + (miss.empty() ? "OK（" + std::string(dsxl::Pipeline::UnetFile()) + "）" : "不足 " + miss) +
                    " / GPU-VAE: " + (gpu ? "OK" : "なし") + "\n" + md;
    return env->NewStringUTF(s.c_str());
}

// UNet（縦横 4 グラフの共有 bin）と default の adapter が配置されているか
JNIEXPORT jboolean JNICALL
Java_com_dragonspeedxl_app_engine_NativeEngine_nativeHasUnet(JNIEnv* env, jobject, jstring model_dir) {
    return dsxl::Pipeline::HasUnet(ToStd(env, model_dir)) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jstring JNICALL
Java_com_dragonspeedxl_app_engine_NativeEngine_nativeInit(JNIEnv* env, jobject, jstring model_dir, jstring lib_dir) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_pipe) return env->NewStringUTF("");
    auto p = std::make_unique<dsxl::Pipeline>();
    std::string err;
    if (!p->Init(ToStd(env, model_dir), ToStd(env, lib_dir), &err)) return env->NewStringUTF(err.c_str());
    g_pipe = std::move(p);
    return env->NewStringUTF("");
}

// CLIP → Base UNet →（hires=true なら）Hires UNet まで実行し、VAE は裏で始める。戻り値は ticket（失敗は 0。理由は nativeLastError）。
// listener.onProgress(stage, step, total) を随時呼ぶ。portrait で UNet のグラフ（縦 / 横）を選ぶ。
// 返ったあと NPU は空くので、次の nativeBegin を続けて呼べる。画像は nativeAwait で受け取る。
JNIEXPORT jlong JNICALL
Java_com_dragonspeedxl_app_engine_NativeEngine_nativeBegin(JNIEnv* env, jobject, jstring prompt, jlong seed, jint steps, jboolean hires, jint hires_steps, jdouble hires_denoise, jboolean portrait, jboolean draft, jstring lora, jfloat lora_alpha, jfloat lora_te_alpha, jobject listener) {
    g_err().clear();
    if (!g_pipe) { g_err() = "エンジンが初期化されていません"; return 0; }
    dsxl::GenParams gp;
    gp.draft = draft == JNI_TRUE;   // 速度優先（TAESD を最終画像にする）
    gp.lora_adapter = ToStd(env, lora);   // 空なら LoRA なし（default の adapter）。名前なら loras/<名前>.4g.bin を適用する
    gp.lora_alpha = lora_alpha;
    gp.lora_te_alpha = lora_te_alpha;   // CLIP の LoRA の強さ w（clip_*_lora.bin と loras/<名前>.clip_*.bin があるときだけ使われる）
    gp.portrait = portrait == JNI_TRUE;
    gp.prompt = ToStd(env, prompt); gp.seed = seed; gp.steps = steps; gp.hires = hires == JNI_TRUE;
    gp.hires_steps = hires_steps; gp.hires_denoise = hires_denoise;
    const uint64_t ticket = g_pipe->Begin(gp, MakeProgress(env, listener), &g_err());
    if (ticket == 0) g_pipe->ReleaseVotes();   // 失敗・キャンセル: VAE まで行かないので、ここで電力の投票を解除する
    return static_cast<jlong>(ticket);
}

// base_ticket（Hires を掛けていない Base の ticket）の latent に Hires（hires_steps 段、denoise）を掛け、VAE を裏で始める。戻り値は ticket（失敗は 0）。
JNIEXPORT jlong JNICALL
Java_com_dragonspeedxl_app_engine_NativeEngine_nativeBeginRefine(JNIEnv* env, jobject, jlong base_ticket, jint hires_steps, jdouble hires_denoise, jboolean draft, jobject listener) {
    g_err().clear();
    if (!g_pipe) { g_err() = "エンジンが初期化されていません"; return 0; }
    g_pipe->SetDraft(draft == JNI_TRUE);   // Hires 化の最終デコードの方式（Base の作成時とは別に、今の設定を渡す）
    const uint64_t ticket = g_pipe->BeginRefine(static_cast<uint64_t>(base_ticket), hires_steps, hires_denoise, MakeProgress(env, listener), &g_err());
    if (ticket == 0) g_pipe->ReleaseVotes();
    return static_cast<jlong>(ticket);
}

// ticket の VAE が終わるまで待って、[w, h, ARGB...] を返す。失敗時は null（理由は nativeLastError）。待っている間 listener に VAE の進捗を流す。
JNIEXPORT jintArray JNICALL
Java_com_dragonspeedxl_app_engine_NativeEngine_nativeAwait(JNIEnv* env, jobject, jlong ticket, jobject listener) {
    g_err().clear();
    if (!g_pipe) { g_err() = "エンジンが初期化されていません"; return nullptr; }
    dsxl::Image img;
    if (!g_pipe->Await(static_cast<uint64_t>(ticket), MakeProgress(env, listener), &img, &g_err())) return nullptr;
    const auto t0 = std::chrono::steady_clock::now();
    jintArray arr = ToJava(env, img);
    __android_log_print(ANDROID_LOG_INFO, "dsxl-time", "to java array %.0f ms", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    return arr;
}

// base_ticket の Base latent に、まだ Hires を掛けられるか
JNIEXPORT jboolean JNICALL Java_com_dragonspeedxl_app_engine_NativeEngine_nativeCanRefine(JNIEnv*, jobject, jlong base_ticket) {
    return g_pipe && g_pipe->CanRefine(static_cast<uint64_t>(base_ticket)) ? JNI_TRUE : JNI_FALSE;
}

// モデルをすべて読み込んで常駐させる（listener には stage=LOAD で 済み/全体 を流す）。
// 戻り値: 空なら成功、"!" で始まれば失敗、それ以外は注記。
JNIEXPORT jstring JNICALL
Java_com_dragonspeedxl_app_engine_NativeEngine_nativeLoadModels(JNIEnv* env, jobject, jboolean portrait, jobject listener) {
    if (!g_pipe) return env->NewStringUTF("!エンジンが初期化されていません");
    return env->NewStringUTF(g_pipe->LoadModels(portrait == JNI_TRUE, MakeProgress(env, listener)).c_str());
}

// ジョブの完了後の先読み。足りないモデルの読み込みを始めるだけで、待たない。
JNIEXPORT void JNICALL
Java_com_dragonspeedxl_app_engine_NativeEngine_nativePrefetch(JNIEnv*, jobject, jboolean portrait) {
    if (g_pipe) g_pipe->PrefetchNext(portrait == JNI_TRUE);
}

// 常駐モデルの状態（ModelBit の論理和。kBitLoading はロード中）。エンジンの初期化前は 0。待たない。
JNIEXPORT jint JNICALL Java_com_dragonspeedxl_app_engine_NativeEngine_nativeModelState(JNIEnv*, jobject, jboolean portrait) {
    return g_pipe ? g_pipe->ModelState(portrait == JNI_TRUE) : 0;
}

// 常駐モデルをすべて解放する（裏の VAE が終わるまで待つ）。
JNIEXPORT void JNICALL Java_com_dragonspeedxl_app_engine_NativeEngine_nativeUnload(JNIEnv*, jobject) {
    if (g_pipe) g_pipe->DropModels();
}

// [NPU の累積実行 ns, GPU の累積実行 ns, 現在時刻 ns]。まだ初期化前なら null。
JNIEXPORT jlongArray JNICALL Java_com_dragonspeedxl_app_engine_NativeEngine_nativeBusy(JNIEnv* env, jobject) {
    if (!g_pipe) return nullptr;
    uint64_t v[3];
    g_pipe->BusyNs(&v[0], &v[1], &v[2]);
    jlongArray arr = env->NewLongArray(3);
    jlong buf[3] = {static_cast<jlong>(v[0]), static_cast<jlong>(v[1]), static_cast<jlong>(v[2])};
    env->SetLongArrayRegion(arr, 0, 3, buf);
    return arr;
}

// GPU の使用率の上限（%）。エンジンの初期化前でも、生成中でも呼べる。
JNIEXPORT void JNICALL Java_com_dragonspeedxl_app_engine_NativeEngine_nativeSetGpuCap(JNIEnv*, jobject, jint percent) { dsxl::Pipeline::SetGpuCap(percent); }

// メモリ節約モード（UNet を今の向きの 2 グラフだけ読む）。エンジンの初期化前でも呼べる。
JNIEXPORT void JNICALL Java_com_dragonspeedxl_app_engine_NativeEngine_nativeSetMemorySaver(JNIEnv*, jobject, jboolean on) { dsxl::Pipeline::SetMemorySaver(on == JNI_TRUE); }

// 生成中のプレビューのリスナーを登録する（null で無効）。新しいジョブの nativeBegin / nativeBeginRefine の前に呼ぶ。
// エンジンの初期化後に呼ぶこと（g_pipe が無いと、登録は覚えるだけで、有効にならない）。
JNIEXPORT void JNICALL
Java_com_dragonspeedxl_app_engine_NativeEngine_nativeSetPreviewListener(JNIEnv* env, jobject, jobject listener) {
    {
        std::lock_guard<std::mutex> lock(g_preview_mu);
        env->GetJavaVM(&g_vm);
        if (g_preview_listener) { env->DeleteGlobalRef(g_preview_listener); g_preview_listener = nullptr; }
        if (listener) {
            g_preview_listener = env->NewGlobalRef(listener);
            g_preview_mid = env->GetMethodID(env->GetObjectClass(listener), "onPreview", "(II[I)V");
        }
    }
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_pipe) g_pipe->SetPreviewFn(listener ? dsxl::Previewer::Fn(CallPreview) : dsxl::Previewer::Fn());
}

JNIEXPORT void JNICALL Java_com_dragonspeedxl_app_engine_NativeEngine_nativeCancel(JNIEnv*, jobject) { if (g_pipe) g_pipe->Cancel(); }

JNIEXPORT jstring JNICALL Java_com_dragonspeedxl_app_engine_NativeEngine_nativeLastError(JNIEnv* env, jobject) { return env->NewStringUTF(g_err().c_str()); }

// :clip プロセス（ClipService）から呼ぶ。CLIP のサーバーを別スレッドで始める（すぐ戻る）
JNIEXPORT void JNICALL
Java_com_dragonspeedxl_app_engine_ClipHost_nativeServe(JNIEnv* env, jobject, jstring model_dir, jstring lib_dir) {
    dsxl::ClipServeAsync(ToStd(env, model_dir), ToStd(env, lib_dir));
}

JNIEXPORT void JNICALL Java_com_dragonspeedxl_app_engine_NativeEngine_nativeRelease(JNIEnv*, jobject) {
    std::lock_guard<std::mutex> lock(g_mu);
    g_pipe.reset();
}

}  // extern "C"
