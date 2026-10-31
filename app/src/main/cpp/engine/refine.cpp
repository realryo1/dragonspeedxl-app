#include "engine/refine.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace dsxl {

namespace {

constexpr int kMargin = 16;   // 領域のまわりの余白（画素）

// PyTorch の reflect と同じ（端の画素は繰り返さない）
inline int Reflect(int i, int n) {
    if (n <= 1) return 0;
    if (i < 0) i = -i;
    if (i >= n) i = 2 * (n - 1) - i;
    return std::min(std::max(i, 0), n - 1);
}

// 1 チャンネル（w × h の float）にガウスぼかしを掛ける（分離型、端は reflect）。refine.py の gauss_blur と同じ。
void GaussBlur(const float* src, float* dst, int w, int h, float sigma, std::vector<float>* tmp) {
    if (sigma <= 0.f) { std::copy(src, src + static_cast<size_t>(w) * h, dst); return; }
    const int r = std::max(1, static_cast<int>(std::ceil(3.0 * sigma)));
    std::vector<float> k(2 * r + 1);
    float sum = 0.f;
    for (int i = -r; i <= r; i++) { k[i + r] = std::exp(-static_cast<float>(i * i) / (2.f * sigma * sigma)); sum += k[i + r]; }
    for (auto& v : k) v /= sum;
    tmp->resize(static_cast<size_t>(w) * h);
    float* t = tmp->data();
    for (int y = 0; y < h; y++) {                       // 横
        const float* row = src + static_cast<size_t>(y) * w;
        float* out = t + static_cast<size_t>(y) * w;
        for (int x = 0; x < w; x++) {
            float a = 0.f;
            for (int i = -r; i <= r; i++) a += k[i + r] * row[Reflect(x + i, w)];
            out[x] = a;
        }
    }
    for (int y = 0; y < h; y++) {                       // 縦
        float* out = dst + static_cast<size_t>(y) * w;
        for (int x = 0; x < w; x++) {
            float a = 0.f;
            for (int i = -r; i <= r; i++) a += k[i + r] * t[static_cast<size_t>(Reflect(y + i, h)) * w + x];
            out[x] = a;
        }
    }
}

// 貼り付けマスク（窓の大きさ、0..1）。領域を feather/2 だけ広げ、ガウスぼかしで縁を落とし、窓の縁は edge 画素かけて 0 にする。
std::vector<float> MakeMask(const RefineWindow& win, const RefineParams& p) {
    const int W = win.ww, H = win.wh;
    std::vector<float> m(static_cast<size_t>(W) * H, 0.f), b(m.size());
    const int e = p.feather / 2;
    const int ya = std::max(win.ry - e, 0), yb = std::min(win.ry + win.rh + e, H);
    const int xa = std::max(win.rx - e, 0), xb = std::min(win.rx + win.rw + e, W);
    for (int y = ya; y < yb; y++) std::fill(m.begin() + static_cast<size_t>(y) * W + xa, m.begin() + static_cast<size_t>(y) * W + xb, 1.f);
    std::vector<float> tmp;
    GaussBlur(m.data(), b.data(), W, H, p.feather / 4.f, &tmp);
    if (p.edge > 0) {
        for (int y = 0; y < H; y++) {
            const float ry = std::min(static_cast<float>(std::min(y, H - 1 - y)) / p.edge, 1.f);
            for (int x = 0; x < W; x++) {
                const float rx = std::min(static_cast<float>(std::min(x, W - 1 - x)) / p.edge, 1.f);
                b[static_cast<size_t>(y) * W + x] *= ry * rx;
            }
        }
    }
    return b;
}

}  // namespace

bool PlanRefineWindow(int x, int y, int w, int h, int img_w, int img_h, int win_w, int win_h, RefineWindow* out, std::string* err) {
    if (w <= 0 || h <= 0 || x < 0 || y < 0 || x + w > img_w || y + h > img_h) { if (err) *err = "領域が画像の外にあります"; return false; }
    if (win_w > img_w || win_h > img_h) { if (err) *err = "窓が画像より大きいです"; return false; }
    // 余白（16 画素ずつ）と、窓を 8 画素の格子に揃えるときのずれ（最大 4 画素ずつ = 8）が入ること
    if (w + 2 * kMargin + 8 > win_w || h + 2 * kMargin + 8 > win_h) {
        if (err) *err = "領域が大きすぎます（" + std::to_string(w) + "x" + std::to_string(h) + "）。最大 " + std::to_string(win_w - 2 * kMargin - 8) + "x" + std::to_string(win_h - 2 * kMargin - 8);
        return false;
    }
    const double cy = y + h / 2.0, cx = x + w / 2.0;   // 領域の中心。窓の左上は、格子（8 画素）に丸める（半分は偶数へ。refine.py の round と同じ）
    const int y0 = std::min(std::max(static_cast<int>(std::nearbyint((cy - win_h / 2.0) / 8.0)) * 8, 0), img_h - win_h);
    const int x0 = std::min(std::max(static_cast<int>(std::nearbyint((cx - win_w / 2.0) / 8.0)) * 8, 0), img_w - win_w);
    out->x0 = x0; out->y0 = y0; out->ww = win_w; out->wh = win_h;
    out->rx = x - x0; out->ry = y - y0; out->rw = w; out->rh = h;
    return true;
}

void BlendRefined(uint8_t* rgb, int img_w, int img_h, const RefineWindow& win, const uint8_t* patch, const RefineParams& p) {
    (void)img_h;
    const int W = win.ww, H = win.wh;
    const size_t N = static_cast<size_t>(W) * H;
    const std::vector<float> mask = MakeMask(win, p);
    std::vector<float> orig(N), ref(N), lo(N), lr(N), tmp;
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < H; y++) {
            const uint8_t* o = rgb + (static_cast<size_t>(win.y0 + y) * img_w + win.x0) * 3 + c;
            const uint8_t* r = patch + static_cast<size_t>(y) * W * 3 + c;
            for (int x = 0; x < W; x++) { orig[static_cast<size_t>(y) * W + x] = o[x * 3]; ref[static_cast<size_t>(y) * W + x] = r[x * 3]; }
        }
        GaussBlur(orig.data(), lo.data(), W, H, p.sigma, &tmp);
        GaussBlur(ref.data(), lr.data(), W, H, p.sigma, &tmp);
        for (int y = 0; y < H; y++) {
            uint8_t* o = rgb + (static_cast<size_t>(win.y0 + y) * img_w + win.x0) * 3 + c;
            for (int x = 0; x < W; x++) {
                const size_t i = static_cast<size_t>(y) * W + x;
                // out = orig + mix · (HP(ref) − HP(orig))。クランプのあと、マスクで貼る
                const float hp_o = orig[i] - lo[i], hp_r = ref[i] - lr[i];
                const float t = std::min(std::max(orig[i] + p.mix * (hp_r - hp_o), 0.f), 255.f);
                const float v = orig[i] * (1.f - mask[i]) + t * mask[i];
                o[x * 3] = static_cast<uint8_t>(std::min(std::max(std::nearbyint(static_cast<double>(v)), 0.0), 255.0));
            }
        }
    }
}

}  // namespace dsxl
