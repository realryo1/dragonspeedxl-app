// 領域リファイン（細部・線の明瞭化）の画像演算。QNN には依存しない（PC で単体検証できる）。計画と実測は docs/refine.md。
// 方式: 領域を収める窓（タイルのグラフと同じ大きさ。ズーム 1 倍）を、エンジンが LCM 4 step で描き直し、VAE で画素に戻す。
// ここでは、その結果（patch）の高周波だけを元の画像へ移植して、マスクで貼る。計算は sdxl-npu/convert/refine.py と同じ。
#pragma once
#include <cstdint>
#include <string>

namespace dsxl {

// 窓（最終画像の画素）と、領域（窓の中の画素）
struct RefineWindow {
    int x0 = 0, y0 = 0, ww = 0, wh = 0;
    int rx = 0, ry = 0, rw = 0, rh = 0;
};

// 高周波の移植と貼り付けの設定。既定値は Phase 0 の振りで決めた（docs/refine.md の 12 節）
struct RefineParams {
    float sigma = 4.f;      // 高周波と低周波の境（ガウスの標準偏差、画素）。小さいと位置のずれた中間周波数まで移植して二重になる
    float mix = 1.f;        // 0 で元のまま、1 で高周波が全部 patch
    int feather = 16;       // マスクの羽根（画素）
    int edge = 8;           // 窓の縁で 0 に落とす幅（画素）
};
constexpr double kRefineDenoise = 0.30;   // 描き直しの denoise の既定値（0.4 は指の間に余計な物が出た）

// 領域 (x, y, w, h)（画素）を余白つきで収める窓を決める。窓の大きさは win_w × win_h（タイルのグラフの大きさ）で、
// 位置は latent の格子（8 画素）に揃え、画像の中に収める。領域が大きすぎる（窓に余白 16 画素ずつ＋格子のずれ 8 画素が入らない）ときは失敗。
bool PlanRefineWindow(int x, int y, int w, int h, int img_w, int img_h, int win_w, int win_h, RefineWindow* out, std::string* err);

// rgb（img_w × img_h × 3、8bit）の窓の位置に、patch（窓と同じ大きさ、rgb 8bit）の高周波を移植して、マスクで貼る（in place）。
void BlendRefined(uint8_t* rgb, int img_w, int img_h, const RefineWindow& win, const uint8_t* patch, const RefineParams& p);

}  // namespace dsxl
