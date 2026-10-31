#include "engine/clip_tokenizer.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <sstream>

namespace dsxl {
namespace {

void AppendUtf8(std::string& s, uint32_t cp) {
    if (cp < 0x80) s += static_cast<char>(cp);
    else if (cp < 0x800) { s += static_cast<char>(0xC0 | (cp >> 6)); s += static_cast<char>(0x80 | (cp & 0x3F)); }
    else { s += static_cast<char>(0xE0 | (cp >> 12)); s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); s += static_cast<char>(0x80 | (cp & 0x3F)); }
}

// UTF-8 文字列を 1 文字ずつに分ける
std::vector<std::string> Utf8Chars(const std::string& s) {
    std::vector<std::string> out;
    for (size_t i = 0; i < s.size();) {
        unsigned char c = s[i];
        size_t n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : 4;
        out.push_back(s.substr(i, n));
        i += n;
    }
    return out;
}

bool IsSpace(unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }
bool IsLetter(unsigned char c) { return std::isalpha(c) || c >= 0x80; }   // 非 ASCII は文字扱い（近似）
bool IsDigit(unsigned char c) { return c >= '0' && c <= '9'; }

}  // namespace

bool ClipTokenizer::Load(const std::string& vocab_tsv, const std::string& merges_txt, std::string* err) {
    std::ifstream fv(vocab_tsv);
    if (!fv) { if (err) *err = "vocab を開けません: " + vocab_tsv; return false; }
    std::string line;
    while (std::getline(fv, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        size_t tab = line.rfind('\t');
        if (tab == std::string::npos) continue;
        vocab_[line.substr(0, tab)] = std::stoi(line.substr(tab + 1));
    }
    std::ifstream fm(merges_txt);
    if (!fm) { if (err) *err = "merges を開けません: " + merges_txt; return false; }
    int rank = 0;
    while (std::getline(fm, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (line.back() == '\r') line.pop_back();
        size_t sp = line.find(' ');
        if (sp == std::string::npos) continue;
        ranks_[{line.substr(0, sp), line.substr(sp + 1)}] = rank++;
    }
    // GPT-2 / CLIP の bytes_to_unicode
    std::vector<int> bs;
    for (int b = '!'; b <= '~'; b++) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; b++) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; b++) bs.push_back(b);
    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) { bs.push_back(b); cs.push_back(256 + n); n++; }
    }
    for (size_t i = 0; i < bs.size(); i++) { std::string s; AppendUtf8(s, cs[i]); byte_to_unicode_[bs[i]] = s; }
    if (vocab_.size() < 49000 || ranks_.empty()) { if (err) *err = "vocab / merges の内容が不正です"; return false; }
    return true;
}

std::vector<std::string> ClipTokenizer::Bpe(const std::string& token) const {
    auto it = cache_.find(token);
    if (it != cache_.end()) return it->second;
    std::vector<std::string> word;
    for (unsigned char c : token) word.push_back(byte_to_unicode_[c]);
    if (!word.empty()) word.back() += "</w>";
    while (word.size() > 1) {
        int best = INT32_MAX; size_t bi = 0;
        for (size_t i = 0; i + 1 < word.size(); i++) {
            auto r = ranks_.find({word[i], word[i + 1]});
            if (r != ranks_.end() && r->second < best) { best = r->second; bi = i; }
        }
        if (best == INT32_MAX) break;
        word[bi] += word[bi + 1];
        word.erase(word.begin() + bi + 1);
    }
    cache_[token] = word;
    return word;
}

std::vector<int32_t> ClipTokenizer::Encode(const std::string& text_in, int32_t pad_id, int* eos_pos) const {
    // 小文字化と空白の正規化
    std::string text;
    for (unsigned char c : text_in) text += (c < 0x80) ? static_cast<char>(std::tolower(c)) : static_cast<char>(c);
    // 正規表現 <|startoftext|>|<|endoftext|>|'s|'t|'re|'ve|'m|'ll|'d|\p{L}+|\p{N}|[^\s\p{L}\p{N}]+ に相当する分割
    std::vector<std::string> toks;
    static const char* kContr[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
    for (size_t i = 0; i < text.size();) {
        unsigned char c = text[i];
        if (IsSpace(c)) { i++; continue; }
        bool matched = false;
        if (c == '\'') {
            for (const char* k : kContr) {
                size_t n = std::strlen(k);
                if (text.compare(i, n, k) == 0) { toks.push_back(std::string(k)); i += n; matched = true; break; }
            }
            if (matched) continue;
        }
        size_t j = i;
        if (IsLetter(c)) { while (j < text.size() && IsLetter(text[j])) j++; }
        else if (IsDigit(c)) { j = i + 1; }
        else { while (j < text.size() && !IsSpace(text[j]) && !IsLetter(text[j]) && !IsDigit(text[j])) j++; }
        toks.push_back(text.substr(i, j - i));
        i = j;
    }
    std::vector<int32_t> ids;
    for (auto& t : toks) {
        for (auto& piece : Bpe(t)) {
            auto v = vocab_.find(piece);
            if (v != vocab_.end()) ids.push_back(v->second);
        }
    }
    if (ids.size() > kMaxLen - 2) ids.resize(kMaxLen - 2);
    std::vector<int32_t> out;
    out.push_back(kBos);
    out.insert(out.end(), ids.begin(), ids.end());
    if (eos_pos) *eos_pos = static_cast<int>(out.size());
    out.push_back(kEos);
    while (static_cast<int>(out.size()) < kMaxLen) out.push_back(pad_id);
    return out;
}

}  // namespace dsxl
