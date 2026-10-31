#include "engine/clip_service.h"

#include <android/log.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <thread>

#include "engine/clip_tokenizer.h"
#include "engine/qnn_model.h"

namespace dsxl {
namespace {

constexpr int kEncode = 1, kWarm = 2, kUnload = 3;
constexpr int kHidden = 77 * 2048, kPooled = 1280;
// 相手（ソケットの向こう）が送る長さの上限。確保の前に検査する（巨大な値で確保させない）
constexpr uint32_t kMaxPrompt = 16 * 1024, kMaxLora = 256;

struct Header { uint32_t type, prompt_len, lora_len; float te_alpha; };

bool Exists(const std::string& p) { struct stat st{}; return stat(p.c_str(), &st) == 0; }

bool ReadAll(int fd, void* buf, size_t n) {
    auto* p = static_cast<char*>(buf);
    while (n > 0) {
        ssize_t r = ::read(fd, p, n);
        if (r <= 0) return false;
        p += r; n -= static_cast<size_t>(r);
    }
    return true;
}
bool WriteAll(int fd, const void* buf, size_t n) {
    auto* p = static_cast<const char*>(buf);
    while (n > 0) {
        ssize_t r = ::send(fd, p, n, MSG_NOSIGNAL);
        if (r <= 0) return false;
        p += r; n -= static_cast<size_t>(r);
    }
    return true;
}

// 抽象名前空間のソケット（ファイルを作らない）。アプリごとに別の名前にする
socklen_t MakeAddr(sockaddr_un* a) {
    std::memset(a, 0, sizeof(*a));
    a->sun_family = AF_UNIX;
    const std::string name = "dsxl_clip_" + std::to_string(getuid());
    std::memcpy(a->sun_path + 1, name.data(), name.size());
    return static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + 1 + name.size());
}

bool ReadFloats(const std::string& path, std::vector<float>* out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const size_t n = static_cast<size_t>(f.tellg());
    out->resize(n / sizeof(float));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(out->data()), static_cast<std::streamsize>(n));
    return true;
}

// ---------------------------------------------------------------------------------------------
// サーバー側: CLIP の context を持ってエンコードする
class ClipEncoder {
public:
    bool Init(const std::string& model_dir, const std::string& lib_dir, std::string* err) {
        dir_ = model_dir;
        if (!tok_.Load(dir_ + "/clip_vocab.tsv", dir_ + "/clip_merges.txt", err)) return false;
        if (!ReadFloats(dir_ + "/text_projection.f32", &text_proj_) || text_proj_.size() != 1280u * 1280u) {
            if (err) *err = "text_projection.f32 が不正です";
            return false;
        }
        return lib_.Init(lib_dir, err);
    }

    // 準備だけ（トークナイザ・QNN の初期化はここまでで済んでいる）。context はエンコードのときに読み、終わったら手放す（Encode の説明）
    bool Warm(const std::string&, std::string*) { return true; }

    void Unload() { models_.clear(); }

    bool Encode(const std::string& prompt, const std::string& lora, float te_alpha, std::vector<float>* out, std::string* err) {
        int eos_l = 0, eos_g = 0;
        std::vector<int32_t> ids_l = tok_.Encode(prompt, ClipTokenizer::kEos, &eos_l);
        std::vector<int32_t> ids_g = tok_.Encode(prompt, 0, &eos_g);
        std::vector<float> pen_l(77 * 768), pen_g(77 * 1280), last_g(77 * 1280);
        std::string ad_l, ad_g;
        const std::string file_l = ClipFileFor(dir_, "clip_l", lora, &ad_l), file_g = ClipFileFor(dir_, "clip_g", lora, &ad_g);
        const float w = te_alpha;
        lib_.Vote();
        {
            auto m = Model(file_l, err);
            if (!m || (!ad_l.empty() && !m->ApplyAdapter(ad_l, err))) return false;
            if (!m->Run({{"input_ids", ids_l.data()}, {"lora_alpha", &w}}, {{"penultimate", pen_l.data()}}, err)) return false;
        }
        {
            auto m = Model(file_g, err);
            if (!m || (!ad_g.empty() && !m->ApplyAdapter(ad_g, err))) return false;
            if (!m->Run({{"input_ids", ids_g.data()}, {"lora_alpha", &w}}, {{"penultimate", pen_g.data()}, {"last_ln", last_g.data()}}, err)) return false;
        }
        lib_.ReleaseVotes();
        __android_log_print(ANDROID_LOG_INFO, "dsxl-clip", "encode: clip_l=%s clip_g=%s te_alpha=%.2f", file_l.c_str(), file_g.c_str(), ad_l.empty() && ad_g.empty() ? 0.0 : w);
        out->assign(kHidden + kPooled, 0.f);
        for (int t = 0; t < 77; t++) {
            std::copy(pen_l.begin() + t * 768, pen_l.begin() + (t + 1) * 768, out->begin() + t * 2048);
            std::copy(pen_g.begin() + t * 1280, pen_g.begin() + (t + 1) * 1280, out->begin() + t * 2048 + 768);
        }
        // pooled = text_projection(last_ln[EOS 位置])。重み W は [out, in]（y = x @ W^T）
        const float* x = last_g.data() + eos_g * 1280;
        for (int j = 0; j < 1280; j++) {
            const float* wj = text_proj_.data() + j * 1280;
            double s = 0;
            for (int i = 0; i < 1280; i++) s += static_cast<double>(x[i]) * wj[i];
            (*out)[kHidden + j] = static_cast<float>(s);
        }
        return true;
    }

private:
    std::shared_ptr<QnnModel> Model(const std::string& file, std::string* err) {
        auto it = models_.find(file);
        if (it != models_.end()) return it->second;
        // 同じ CLIP の、もう一方の組は手放す（通常と LoRA つきの両方を載せると約 1.7GB 余計に使う）
        const bool lora = file.size() > 9 && file.compare(file.size() - 9, 9, "_lora.bin") == 0;
        models_.erase(lora ? file.substr(0, file.size() - 9) + ".bin" : file.substr(0, file.size() - 4) + "_lora.bin");
        auto m = std::make_shared<QnnModel>();
        const auto t0 = std::chrono::steady_clock::now();
        if (!m->Load(&lib_, dir_ + "/" + file, err)) return nullptr;
        __android_log_print(ANDROID_LOG_INFO, "dsxl-clip", "loaded %s (%.0f ms)", file.c_str(),
                            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        models_[file] = m;
        return m;
    }

    std::string dir_;
    QnnBackendLib lib_;
    ClipTokenizer tok_;
    std::vector<float> text_proj_;
    std::map<std::string, std::shared_ptr<QnnModel>> models_;
};

void Reply(int fd, bool ok, const std::vector<float>& data, const std::string& err) {
    uint32_t head[2];
    if (ok) {
        head[0] = 0; head[1] = static_cast<uint32_t>(data.size());
        WriteAll(fd, head, sizeof head);
        if (!data.empty()) WriteAll(fd, data.data(), data.size() * sizeof(float));
    } else {
        head[0] = 1; head[1] = static_cast<uint32_t>(err.size());
        WriteAll(fd, head, sizeof head);
        WriteAll(fd, err.data(), err.size());
    }
}

}  // namespace

std::string ClipFileFor(const std::string& model_dir, const char* which, const std::string& lora, std::string* adapter) {
    const std::string w = which;
    if (!lora.empty()) {
        const std::string ad = model_dir + "/loras/" + lora + "." + w + ".bin";
        if (Exists(model_dir + "/" + w + "_lora.bin") && Exists(ad)) { if (adapter) *adapter = ad; return w + "_lora.bin"; }
    }
    if (adapter) adapter->clear();
    return w + ".bin";
}

void ClipServe(const std::string& model_dir, const std::string& lib_dir) {
    ClipEncoder enc;
    bool inited = false;
    std::string err;
    sockaddr_un addr;
    const socklen_t alen = MakeAddr(&addr);
    int ls = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (ls < 0 || bind(ls, reinterpret_cast<sockaddr*>(&addr), alen) != 0 || listen(ls, 4) != 0) {
        __android_log_print(ANDROID_LOG_ERROR, "dsxl-clip", "socket の準備に失敗: errno %d", errno);
        if (ls >= 0) close(ls);
        return;
    }
    __android_log_print(ANDROID_LOG_INFO, "dsxl-clip", "CLIP サーバー開始 (pid %d)", getpid());
    for (;;) {
        int fd = accept4(ls, nullptr, nullptr, SOCK_CLOEXEC);
        if (fd < 0) continue;
        // 抽象ソケットはファイルの権限を持たないので、接続元の uid がこのアプリの uid かを確かめる
        ucred cred{};
        socklen_t cred_len = sizeof cred;
        if (getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &cred, &cred_len) != 0 || cred.uid != getuid()) {
            close(fd);
            continue;
        }
        for (;;) {
            Header h;
            if (!ReadAll(fd, &h, sizeof h)) break;
            // 長さと要求の種類を、確保や処理の前に検査する。範囲外なら接続を閉じる
            const bool known = h.type == kEncode || h.type == kWarm || h.type == kUnload;
            if (!known || h.prompt_len > kMaxPrompt || h.lora_len > kMaxLora) break;
            std::string prompt(h.prompt_len, '\0'), lora(h.lora_len, '\0');
            if ((h.prompt_len && !ReadAll(fd, &prompt[0], h.prompt_len)) || (h.lora_len && !ReadAll(fd, &lora[0], h.lora_len))) break;
            std::vector<float> out;
            err.clear();
            bool ok = true;
            if (!inited) { ok = enc.Init(model_dir, lib_dir, &err); inited = ok; }
            if (ok) {
                if (h.type == kEncode) {
                    ok = enc.Encode(prompt, lora, h.te_alpha, &out, &err);
                    // エンコードのたびに context を読み、終わったら手放す: CLIP の重み（約 1.7GB、DMA-BUF）は DSP が固定して読むので圧縮も swap もできず、
                    // 常駐させると端末全体のメモリが逼迫して、他のアプリが lowmemorykiller に落とされる（2026-10-18 に実機で確認）。読み込みは約 1.6 秒。
                    // エンコード結果を使い回す生成（同じプロンプトと LoRA）では呼ばれない。
                    enc.Unload();
                }
                else if (h.type == kWarm) ok = enc.Warm(lora, &err);
                else if (h.type == kUnload) enc.Unload();
                else { ok = false; err = "不明な要求です"; }
            }
            if (!ok) __android_log_print(ANDROID_LOG_ERROR, "dsxl-clip", "失敗: %s", err.c_str());
            Reply(fd, ok, out, err);
        }
        close(fd);
    }
}

bool ClipServeAsync(const std::string& model_dir, const std::string& lib_dir) {
    static std::atomic<bool> started{false};
    if (started.exchange(true)) return true;
    std::thread([model_dir, lib_dir]() { ClipServe(model_dir, lib_dir); }).detach();
    return true;
}

// ---------------------------------------------------------------------------------------------
ClipClient::~ClipClient() { Close(); }

void ClipClient::Close() {
    if (fd_ >= 0) { close(fd_); fd_ = -1; }
}

bool ClipClient::Connect(int wait_ms) {
    if (fd_ >= 0) return true;
    sockaddr_un addr;
    const socklen_t alen = MakeAddr(&addr);
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
        if (fd >= 0 && connect(fd, reinterpret_cast<sockaddr*>(&addr), alen) == 0) { fd_ = fd; return true; }
        if (fd >= 0) close(fd);
        if (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() > wait_ms) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

bool ClipClient::Request(int type, const std::string& prompt, const std::string& lora, float te_alpha, std::vector<float>* out, std::string* err) {
    std::lock_guard<std::mutex> lock(mu_);
    for (int attempt = 0; attempt < 3; attempt++) {   // メモリ逼迫で CLIP のプロセスが落とされても、作り直されるのを待ってやり直す
        if (!Connect(15000)) { if (err) *err = "CLIP のプロセスにつながりません（サービスが起動していません）"; return false; }
        Header h{static_cast<uint32_t>(type), static_cast<uint32_t>(prompt.size()), static_cast<uint32_t>(lora.size()), te_alpha};
        uint32_t head[2];
        if (WriteAll(fd_, &h, sizeof h) && WriteAll(fd_, prompt.data(), prompt.size()) && WriteAll(fd_, lora.data(), lora.size()) && ReadAll(fd_, head, sizeof head)) {
            if (head[0] == 0) {
                if (out) { out->resize(head[1]); if (head[1] && !ReadAll(fd_, out->data(), head[1] * sizeof(float))) { Close(); continue; } }
                else if (head[1]) { std::vector<float> tmp(head[1]); ReadAll(fd_, tmp.data(), head[1] * sizeof(float)); }
                return true;
            }
            std::string e(head[1], '\0');
            if (head[1]) ReadAll(fd_, &e[0], head[1]);
            if (err) *err = "CLIP: " + e;
            return false;
        }
        Close();   // 切れていたら、つなぎ直して 1 回だけやり直す
    }
    if (err) *err = "CLIP のプロセスとの通信に失敗しました";
    return false;
}

bool ClipClient::Encode(const std::string& prompt, const std::string& lora, float te_alpha, std::vector<float>* hidden, std::vector<float>* pooled, std::string* err) {
    std::vector<float> out;
    if (!Request(kEncode, prompt, lora, te_alpha, &out, err)) return false;
    if (out.size() != static_cast<size_t>(kHidden + kPooled)) { if (err) *err = "CLIP の応答の大きさが不正です"; return false; }
    hidden->assign(out.begin(), out.begin() + kHidden);
    pooled->assign(out.begin() + kHidden, out.end());
    return true;
}

bool ClipClient::Warm(const std::string& lora, std::string* err) { return Request(kWarm, "", lora, 0.f, nullptr, err); }

void ClipClient::Unload() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (fd_ < 0) return;
    }
    Request(kUnload, "", "", 0.f, nullptr, nullptr);
}

}  // namespace dsxl
