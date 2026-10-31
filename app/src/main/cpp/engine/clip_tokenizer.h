#pragma once
// CLIP の BPE トークナイザ（HF CLIPTokenizer 相当。ASCII 中心。強調構文 (tag:1.2) には未対応）。
#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dsxl {

class ClipTokenizer {
public:
    // vocab.tsv: 1 行 "トークン<TAB>id"、merges.txt: HF の merges.txt（先頭の #version 行は読み飛ばす）
    bool Load(const std::string& vocab_tsv, const std::string& merges_txt, std::string* err);
    // [BOS] + tokens(最大 75) + [EOS] + pad を 77 要素で返す。pad_id は CLIP-L: 49407 / bigG: 0。
    std::vector<int32_t> Encode(const std::string& text, int32_t pad_id, int* eos_pos = nullptr) const;

    static constexpr int32_t kBos = 49406;
    static constexpr int32_t kEos = 49407;
    static constexpr int kMaxLen = 77;

private:
    std::vector<std::string> Bpe(const std::string& token) const;
    std::unordered_map<std::string, int32_t> vocab_;
    std::map<std::pair<std::string, std::string>, int> ranks_;
    std::string byte_to_unicode_[256];
    mutable std::unordered_map<std::string, std::vector<std::string>> cache_;
};

}  // namespace dsxl
