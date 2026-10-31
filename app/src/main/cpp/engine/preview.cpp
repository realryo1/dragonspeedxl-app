#include "engine/preview.h"

#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>

#include <android/log.h>

namespace dsxl {
namespace {
bool Exists(const std::string& p) { struct stat st{}; return stat(p.c_str(), &st) == 0; }
const char* kTag = "dsxl-preview";
std::string StemOf(const std::string& dir, int lh, int lw) { return dir + "/gpu/taesd_" + std::to_string(lh) + "x" + std::to_string(lw); }
}  // namespace

Previewer::Previewer(QnnBackendLib* gpu_lib, std::string model_dir, std::function<int()> cap)
    : lib_(gpu_lib), dir_(std::move(model_dir)), cap_(std::move(cap)), th_([this] { Loop(); }) {}

Previewer::~Previewer() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    if (th_.joinable()) th_.join();
}

void Previewer::SetCallback(Fn fn) {
    std::lock_guard<std::mutex> lock(mu_);
    fn_ = std::move(fn);
    enabled_ = static_cast<bool>(fn_);
    if (!fn_) has_ = false;
}

void Previewer::Submit(const std::vector<float>& latent, int lh, int lw) {
    if (!enabled_) return;
    {
        std::lock_guard<std::mutex> lock(mu_);
        pending_ = latent;   // 最新で置き換える
        plh_ = lh; plw_ = lw;
        has_ = true;
    }
    cv_.notify_one();
}

void Previewer::Reset() {
    std::lock_guard<std::mutex> lock(mu_);
    has_ = false;
}

void Previewer::DropModels() {
    drop_ = true;
    cv_.notify_one();   // ワーカーが次に起きたときに手放す（待機中でも起こして手放させる）
}

bool Previewer::HasFiles(int lh, int lw) const {
    const std::string stem = StemOf(dir_, lh, lw);
    return Exists(stem + "_gpu.bin") || Exists(stem + "_fp32.dlc");
}

std::shared_ptr<QnnModel> Previewer::Model(int lh, int lw, std::string* err) {
    const int key = lh * 10000 + lw;
    if (failed_.count(key)) { if (err) *err = "この寸法のモデルは使えません"; return nullptr; }
    auto it = models_.find(key);
    if (it != models_.end()) return it->second;
    const std::string stem = StemOf(dir_, lh, lw);
    const std::string dlc = stem + "_fp32.dlc", bin = stem + "_gpu.bin";
    std::shared_ptr<QnnModel> m;
    std::string e;
    if (Exists(bin)) {
        auto t = std::make_shared<QnnModel>();
        if (t->Load(lib_, bin, &e)) m = std::move(t);
        else std::remove(bin.c_str());   // 壊れた・古いキャッシュは作り直す
    }
    if (!m && Exists(dlc)) {
        const auto t0 = std::chrono::steady_clock::now();
        auto t = std::make_shared<QnnModel>();
        if (lib_->BuildGpuContextBinary(dlc, bin, &e) && t->Load(lib_, bin, &e)) {
            m = std::move(t);
            __android_log_print(ANDROID_LOG_INFO, kTag, "built GPU TAESD %dx%d in %.0f ms", lh, lw,
                                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
    }
    const TensorDesc* in = m ? m->Input("latent") : nullptr;
    const TensorDesc* out = m ? m->Output("image") : nullptr;
    if (!m || !in || !out || in->dims.size() != 4 || out->dims.size() != 4) {
        const std::string why = m ? "想定と違うテンソルです" : (e.empty() ? "モデルのファイルがありません" : e);
        __android_log_print(ANDROID_LOG_WARN, kTag, "TAESD %dx%d unavailable: %s", lh, lw, why.c_str());
        if (err) *err = why;
        failed_.insert(key);
        return nullptr;
    }
    models_[key] = m;
    return m;
}

bool Previewer::DecodeLocked(const std::vector<float>& latent, int lh, int lw, int f, std::vector<uint8_t>* rgb, int* w_out, int* h_out, double* ms, std::string* err) {
    if (latent.size() != static_cast<size_t>(4) * lh * lw) { if (err) *err = "latent の寸法が合いません"; return false; }
    auto m = Model(lh, lw, err);
    if (!m) return false;
    const auto t0 = std::chrono::steady_clock::now();
    const int H = lh * 8, W = lw * 8;
    out_.resize(static_cast<size_t>(3) * H * W);
    if (!m->Run({{"latent", latent.data()}}, {{"image", out_.data()}}, err)) { failed_.insert(lh * 10000 + lw); return false; }
    // 出力は NCHW の 0〜1 付近（ComfyUI と同じく clamp(0,1)×255）。f=2 のときは 2×2 の平均で半分にする
    const int w = W / f, h = H / f;
    rgb->resize(static_cast<size_t>(w) * h * 3);
    const size_t plane = static_cast<size_t>(H) * W;
    const float k = 255.f / (f * f);
    auto convert = [&](int y0, int y1) {
        for (int y = y0; y < y1; y++) {
            for (int x = 0; x < w; x++) {
                for (int c = 0; c < 3; c++) {
                    const float* p = out_.data() + c * plane + static_cast<size_t>(y) * f * W + static_cast<size_t>(x) * f;
                    float s = 0.f;
                    for (int dy = 0; dy < f; dy++) for (int dx = 0; dx < f; dx++) s += std::min(std::max(p[static_cast<size_t>(dy) * W + dx], 0.f), 1.f);
                    (*rgb)[(static_cast<size_t>(y) * w + x) * 3 + c] = static_cast<uint8_t>(s * k + 0.5f);
                }
            }
        }
    };
    if (static_cast<size_t>(w) * h > 400000) {   // 大きい絵は、行を 4 つに分けて並列に変換する
        const int nt = 4, chunk = (h + nt - 1) / nt;
        std::vector<std::thread> th;
        for (int t = 1; t < nt; t++) th.emplace_back(convert, std::min(t * chunk, h), std::min((t + 1) * chunk, h));
        convert(0, std::min(chunk, h));
        for (auto& x : th) x.join();
    } else {
        convert(0, h);
    }
    *w_out = w; *h_out = h;
    if (ms) *ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

bool Previewer::DecodeFull(const std::vector<float>& latent, int lh, int lw, std::vector<uint8_t>* rgb, int* w, int* h, std::string* err) {
    std::lock_guard<std::mutex> lock(run_mu_);
    double ms = 0;
    const bool ok = DecodeLocked(latent, lh, lw, 1, rgb, w, h, &ms, err);
    if (ok) __android_log_print(ANDROID_LOG_INFO, kTag, "draft (TAESD final) %dx%d -> %dx%d in %.0f ms", lh, lw, *w, *h, ms);
    return ok;
}

void Previewer::Loop() {
    for (;;) {
        std::vector<float> lat;
        int lh = 0, lw = 0;
        Fn fn;
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_.wait(lock, [this] { return stop_ || has_ || drop_.load(); });
            if (stop_) return;
            if (drop_.exchange(false)) {
                std::lock_guard<std::mutex> rl(run_mu_);
                models_.clear(); failed_.clear();
            }
            if (!has_) continue;
            lat.swap(pending_);
            lh = plh_; lw = plw_;
            has_ = false;
            fn = fn_;
        }
        if (!fn) continue;
        double busy_ms = 0;
        int w = 0, h = 0;
        std::string e;
        bool ok;
        {
            std::lock_guard<std::mutex> rl(run_mu_);
            // 大きい絵（Hires）は 2×2 の平均で半分にして、JNI と表示の負担を減らす
            ok = DecodeLocked(lat, lh, lw, std::max(lh, lw) >= 200 ? 2 : 1, &rgb_, &w, &h, &busy_ms, &e);
        }
        if (!ok) { __android_log_print(ANDROID_LOG_WARN, kTag, "preview failed: %s", e.c_str()); continue; }
        __android_log_print(ANDROID_LOG_INFO, kTag, "preview %dx%d -> %dx%d in %.0f ms", lh, lw, w, h, busy_ms);
        fn(w, h, rgb_);
        // GPU の使用率の上限に合わせて休む（OS の描画に GPU を空ける。休み中に新しい latent が来ても、最新のものだけ処理する）
        const int cap = cap_ ? cap_() : 100;
        if (cap < 100) {
            double rest = busy_ms * (100 - cap) / cap;
            while (rest > 0) {
                {
                    std::lock_guard<std::mutex> lock(mu_);
                    if (stop_) return;
                }
                const double step = std::min(rest, 20.0);
                std::this_thread::sleep_for(std::chrono::microseconds(static_cast<long>(step * 1000)));
                rest -= step;
            }
        }
    }
}

}  // namespace dsxl
