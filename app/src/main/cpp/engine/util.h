#pragma once
// エンジン全体で共通の小さな道具（ファイルの有無・float の読み込み・RGB → ARGB）。
#include <sys/stat.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace dsxl {

inline bool FileExists(const std::string& p) { struct stat st{}; return stat(p.c_str(), &st) == 0; }

// ファイル全体を float32 の並びとして読む
inline bool ReadFloats(const std::string& path, std::vector<float>* out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const size_t n = static_cast<size_t>(f.tellg());
    out->resize(n / sizeof(float));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(out->data()), static_cast<std::streamsize>(n));
    return true;
}

// 8bit の RGB（n 画素）を 0xFFRRGGBB にする（Java の Bitmap に渡す形）
inline void RgbToArgb(const uint8_t* rgb, int32_t* argb, size_t n) {
    for (size_t i = 0; i < n; i++, rgb += 3)
        argb[i] = static_cast<int32_t>(0xFF000000u | (static_cast<uint32_t>(rgb[0]) << 16) | (static_cast<uint32_t>(rgb[1]) << 8) | rgb[2]);
}

}  // namespace dsxl
