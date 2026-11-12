#include "vecengine/bm25.hpp"
#include "parallel.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace vecengine {

namespace {

constexpr std::array<uint32_t, 256> make_crc_table() {
    std::array<uint32_t, 256> t{};
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t c = i;
        for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        t[i] = c;
    }
    return t;
}

constexpr std::array<uint32_t, 256> kCrcTable = make_crc_table();

// A token of up to 8 bytes packed little-endian into an integer, for the stopword check.
constexpr uint64_t pack(std::string_view t) {
    uint64_t v = 0;
    for (size_t i = 0; i < t.size(); ++i) v |= uint64_t(static_cast<unsigned char>(t[i])) << (8 * i);
    return v;
}

constexpr uint64_t kStop[] = {
    pack("a"), pack("an"), pack("and"), pack("are"), pack("as"), pack("at"), pack("be"), pack("but"), pack("by"),
    pack("for"), pack("if"), pack("in"), pack("into"), pack("is"), pack("it"), pack("no"), pack("not"), pack("of"),
    pack("on"), pack("or"), pack("such"), pack("that"), pack("the"), pack("their"), pack("then"), pack("there"),
    pack("these"), pack("they"), pack("this"), pack("to"), pack("was"), pack("will"), pack("with")};

bool is_stopword(uint64_t packed) {
    for (uint64_t w : kStop)
        if (w == packed) return true;
    return false;
}

// Each byte lowercased if it is an ASCII letter or digit, else 0.
constexpr std::array<unsigned char, 256> make_token_table() {
    std::array<unsigned char, 256> t{};
    for (int c = 'a'; c <= 'z'; ++c) t[c] = static_cast<unsigned char>(c);
    for (int c = 'A'; c <= 'Z'; ++c) t[c] = static_cast<unsigned char>(c - 'A' + 'a');
    for (int c = '0'; c <= '9'; ++c) t[c] = static_cast<unsigned char>(c);
    return t;
}

constexpr std::array<unsigned char, 256> kTokenByte = make_token_table();

// Sorted term ids of a text and their counts.
struct Counted {
    std::vector<uint32_t> terms;
    std::vector<uint32_t> tf;
    uint64_t length = 0;
};

Counted count(std::string_view text) {
    std::vector<uint32_t> ids = BM25::terms(text);
    std::sort(ids.begin(), ids.end());
    Counted c;
    c.length = ids.size();
    for (size_t i = 0; i < ids.size();) {
        size_t j = i;
        while (j < ids.size() && ids[j] == ids[i]) ++j;
        c.terms.push_back(ids[i]);
        c.tf.push_back(static_cast<uint32_t>(j - i));
        i = j;
    }
    return c;
}

std::vector<Counted> count_all(std::span<const std::string> texts) {
    std::vector<Counted> out(texts.size());
    detail::parallel_for(texts.size(), [&](size_t i) { out[i] = count(texts[i]); });
    return out;
}

} // namespace

BM25::BM25(double k1, double b) : k1_(k1), b_(b) {}

std::vector<uint32_t> BM25::terms(std::string_view text) {
    std::vector<uint32_t> out;
    uint32_t crc = 0xFFFFFFFFu;  // zlib.crc32 of the token so far
    size_t len = 0;
    uint64_t packed = 0;         // its first 8 bytes
    auto push = [&](unsigned char c) {
        crc = kCrcTable[(crc ^ c) & 0xFF] ^ (crc >> 8);
        if (len < 8) packed |= uint64_t(c) << (8 * len);
        ++len;
    };
    auto flush = [&] {
        if (len && (len > 5 || !is_stopword(packed)))
            out.push_back((crc ^ 0xFFFFFFFFu) & static_cast<uint32_t>(kVocab - 1));
        crc = 0xFFFFFFFFu;
        len = 0;
        packed = 0;
    };
    const auto* p = reinterpret_cast<const unsigned char*>(text.data());
    const size_t n = text.size();
    for (size_t i = 0; i < n; ++i) {
        if (const unsigned char c = kTokenByte[p[i]]) {
            push(c);
        } else if (p[i] == 0xE2 && i + 2 < n && p[i + 1] == 0x84 && p[i + 2] == 0xAA) {
            // Python lowercases the Kelvin sign U+212A to 'k' and U+0130 to 'i'
            // plus a combining dot (a separator); no other non-ASCII character
            // lowercases to an ASCII letter or digit.
            push('k');
            i += 2;
        } else if (p[i] == 0xC4 && i + 1 < n && p[i + 1] == 0xB0) {
            push('i');
            flush();
            i += 1;
        } else {
            flush();
        }
    }
    flush();
    return out;
}

BM25::Rows BM25::encode_documents(std::span<const std::string> texts) {
    const std::vector<Counted> docs = count_all(texts);
    for (const Counted& d : docs) {
        for (uint32_t term : d.terms) ++df_[term];
        total_len_ += d.length;
    }
    n_docs_ += texts.size();
    double avgdl = n_docs_ ? static_cast<double>(total_len_) / static_cast<double>(n_docs_) : 0.0;
    if (avgdl == 0) avgdl = 1.0;

    Rows rows;
    for (const Counted& d : docs) {
        const double norm = k1_ * (1 - b_ + b_ * static_cast<double>(d.length) / avgdl);
        for (size_t i = 0; i < d.terms.size(); ++i) {
            const double tf = d.tf[i];
            rows.indices.push_back(d.terms[i]);
            rows.values.push_back(static_cast<float>(tf * (k1_ + 1) / (tf + norm)));
        }
        rows.indptr.push_back(rows.indices.size());
    }
    return rows;
}

BM25::Rows BM25::encode_queries(std::span<const std::string> texts) const {
    Rows rows;
    const double n = static_cast<double>(n_docs_);
    for (const std::string& t : texts) {
        for (uint32_t term : count(t).terms) {
            const auto it = df_.find(term);
            if (it == df_.end()) continue;
            const double df = it->second;
            rows.indices.push_back(term);
            rows.values.push_back(static_cast<float>(std::log(1 + (n - df + 0.5) / (df + 0.5))));
        }
        rows.indptr.push_back(rows.indices.size());
    }
    return rows;
}

void BM25::remove(std::span<const std::string> texts) {
    for (const Counted& c : count_all(texts)) {
        for (uint32_t term : c.terms) {
            const auto it = df_.find(term);
            if (it != df_.end() && --it->second == 0) df_.erase(it);
        }
        total_len_ -= c.length;
        --n_docs_;
    }
}

void BM25::rebuild(std::span<const std::string> texts) {
    df_.clear();
    total_len_ = 0;
    n_docs_ = 0;
    for (const Counted& c : count_all(texts)) {
        for (uint32_t term : c.terms) ++df_[term];
        total_len_ += c.length;
    }
    n_docs_ = texts.size();
}

} // namespace vecengine
