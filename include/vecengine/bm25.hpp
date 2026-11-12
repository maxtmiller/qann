#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace vecengine {

// BM25 sparse vectors that stay exact as the corpus grows and shrinks (issue #159).
//
// Documents store only their saturated term frequencies,
// tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avgdl)), and each query term
// carries its IDF, ln(1 + (N - df + 0.5) / (df + 0.5)), from the current
// counts. Adding or removing documents updates IDF for every later query
// without re-encoding stored ones. avgdl is the average length when a
// document is encoded.
//
// Tokens are runs of ASCII letters and digits after Unicode lowercasing (as
// Python's re.findall(r"[a-z0-9]+", text.lower())), minus Lucene's English
// stopwords, no stemming. A term's id is the CRC-32 of its UTF-8 bytes
// truncated to 31 bits, so there is no vocabulary to keep: the counts can be
// rebuilt from the texts alone.
class BM25 {
public:
    // Rows in CSR form; term ids in [0, kVocab).
    struct Rows {
        std::vector<uint64_t> indptr{0};
        std::vector<uint32_t> indices;
        std::vector<float> values;
    };

    static constexpr uint64_t kVocab = uint64_t{1} << 31;

    explicit BM25(double k1 = 1.2, double b = 0.75);

    // Counts texts into the statistics, then encodes them, one row each.
    Rows encode_documents(std::span<const std::string> texts);

    // One row of IDF weights per text, for the terms some document has.
    Rows encode_queries(std::span<const std::string> texts) const;

    // Takes previously encoded texts back out of the statistics.
    void remove(std::span<const std::string> texts);

    // Resets the statistics to exactly these texts (after loading a saved store).
    void rebuild(std::span<const std::string> texts);

    // Term ids of text's tokens, in order, repeats kept.
    static std::vector<uint32_t> terms(std::string_view text);

    double k1() const noexcept { return k1_; }
    double b() const noexcept { return b_; }
    uint64_t num_documents() const noexcept { return n_docs_; }
    uint64_t total_length() const noexcept { return total_len_; }
    size_t num_terms() const noexcept { return df_.size(); }

private:
    double k1_, b_;
    std::unordered_map<uint32_t, uint32_t> df_;
    uint64_t n_docs_ = 0;
    uint64_t total_len_ = 0;
};

} // namespace vecengine
