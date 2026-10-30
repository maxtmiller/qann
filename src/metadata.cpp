#include "vecengine/metadata.hpp"
#include "vecengine/id_map.hpp"
#include "serialize.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace vecengine {

namespace {

bool is_comparison(Expr::Op op) {
    return op == Expr::Op::Eq || op == Expr::Op::Ne || op == Expr::Op::Lt || op == Expr::Op::Le ||
           op == Expr::Op::Gt || op == Expr::Op::Ge;
}

template <class T>
bool compare(Expr::Op op, const T& a, const T& b) {
    switch (op) {
        case Expr::Op::Eq: return a == b;
        case Expr::Op::Ne: return a != b;
        case Expr::Op::Lt: return a < b;
        case Expr::Op::Le: return a <= b;
        case Expr::Op::Gt: return a > b;
        case Expr::Op::Ge: return a >= b;
        default: return false;
    }
}

std::string value_string(const Expr::Value& v) {
    std::ostringstream s;
    if (auto* i = std::get_if<int64_t>(&v)) s << *i;
    else if (auto* d = std::get_if<double>(&v)) {
        s << *d;
        if (std::isfinite(*d) && *d == std::floor(*d) && std::abs(*d) < 1e15) s << ".0";  // keep 1.0 a float
    }
    else s << '\'' << std::get<std::string>(v) << '\'';
    return s.str();
}

const char* op_string(Expr::Op op) {
    switch (op) {
        case Expr::Op::Eq: return "==";
        case Expr::Op::Ne: return "!=";
        case Expr::Op::Lt: return "<";
        case Expr::Op::Le: return "<=";
        case Expr::Op::Gt: return ">";
        case Expr::Op::Ge: return ">=";
        default: return "?";
    }
}

} // namespace

Expr Expr::compare(std::string field, Op op, Value value) {
    if (!is_comparison(op)) throw std::invalid_argument("compare needs ==, !=, <, <=, > or >=");
    return Expr(std::make_shared<const Node>(Node{op, std::move(field), {std::move(value)}, {}}));
}

Expr Expr::isin(std::string field, vector<Value> values) {
    return Expr(std::make_shared<const Node>(Node{Op::In, std::move(field), std::move(values), {}}));
}

Expr Expr::is_null(std::string field) {
    return Expr(std::make_shared<const Node>(Node{Op::IsNull, std::move(field), {}, {}}));
}

static void check_depth(size_t depth) {
    if (depth > Expr::kMaxDepth)
        throw std::invalid_argument("filter nested more than " + std::to_string(Expr::kMaxDepth) + " levels deep");
}

Expr Expr::join(Op op, vector<Expr> children) {
    if (op != Op::And && op != Op::Or) throw std::invalid_argument("join needs And or Or");
    if (children.empty()) throw std::invalid_argument("join needs at least one filter");
    if (children.size() == 1) return std::move(children[0]);
    vector<Expr> flat;
    flat.reserve(children.size());
    size_t below = 0;
    for (Expr& c : children) {
        if (c.op() == op) {
            flat.insert(flat.end(), c.children().begin(), c.children().end());
            below = std::max(below, c.node_->depth - 1);
        } else {
            below = std::max(below, c.node_->depth);
            flat.push_back(std::move(c));
        }
    }
    check_depth(below + 1);
    return Expr(std::make_shared<const Node>(Node{op, {}, {}, std::move(flat), below + 1}));
}

Expr Expr::between(std::string field, Value lo, Value hi) {
    Expr ge = compare(field, Op::Ge, std::move(lo));
    return ge & compare(std::move(field), Op::Le, std::move(hi));
}

Expr operator&(const Expr& a, const Expr& b) { return Expr::join(Expr::Op::And, {a, b}); }

Expr operator|(const Expr& a, const Expr& b) { return Expr::join(Expr::Op::Or, {a, b}); }

Expr operator~(const Expr& a) {
    check_depth(a.node_->depth + 1);
    return Expr(std::make_shared<const Expr::Node>(Expr::Node{Expr::Op::Not, {}, {}, {a}, a.node_->depth + 1}));
}

static std::atomic<size_t> g_max_conditions{Expr::kDefaultMaxConditions};

size_t Expr::max_conditions() noexcept { return g_max_conditions.load(std::memory_order_relaxed); }

void Expr::set_max_conditions(size_t n) {
    if (n == 0) throw std::invalid_argument("the filter condition limit must be at least 1");
    g_max_conditions.store(n, std::memory_order_relaxed);
}

namespace {

// Recursive descent over a SQL WHERE clause (see Expr::parse). NOT is pushed
// down to the leaves as it parses (neg), so the tree has SQL's meaning
// without ~: negated comparisons flip their operator, negated IN / IS NULL /
// BETWEEN exclude nulls explicitly, and AND / OR swap by De Morgan.
class SqlParser {
public:
    SqlParser(std::string_view sql, const vector<Expr::Value>* list, const std::unordered_map<std::string, Expr::Value>* named)
        : sql_(sql), list_(list), named_(named), max_conditions_(Expr::max_conditions()) {}

    Expr parse() {
        Expr e = or_expr(false);
        skip_space();
        if (pos_ < sql_.size()) fail("unexpected '" + peek_word() + "'");
        if (list_ && used_ != list_->size())
            throw std::invalid_argument("filter has " + std::to_string(used_) + " ? placeholders but " +
                                        std::to_string(list_->size()) + " parameters were given");
        return e;
    }

private:
    [[noreturn]] void fail(const std::string& what) const {
        throw std::invalid_argument(what + (pos_ < sql_.size() ? " at column " + std::to_string(pos_ + 1) : " at end of filter"));
    }

    void skip_space() {
        while (pos_ < sql_.size() && std::string_view(" \t\n\r\f\v").find(sql_[pos_]) != std::string_view::npos) ++pos_;
    }

    // ASCII only, not <cctype>: its answers for bytes >= 0x80 depend on the locale
    static bool is_alpha(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
    static bool is_digit(char c) { return c >= '0' && c <= '9'; }
    static bool is_ascii(char c) { return static_cast<unsigned char>(c) < 0x80; }
    static char upper(char c) { return c >= 'a' && c <= 'z' ? static_cast<char>(c - 'a' + 'A') : c; }
    static bool word_char(char c) { return is_alpha(c) || is_digit(c) || c == '_' || c == '.'; }

    std::string peek_word() {
        skip_space();
        size_t end = pos_;
        while (end < sql_.size() && word_char(sql_[end])) ++end;
        if (end == pos_ && pos_ < sql_.size()) {
            ++end;
            while (end < sql_.size() && (static_cast<unsigned char>(sql_[end]) & 0xC0) == 0x80) ++end;  // rest of a UTF-8 character
        }
        return std::string(sql_.substr(pos_, end - pos_));
    }

    // Consumes keyword kw (case-insensitive, whole word) if it comes next.
    bool keyword(std::string_view kw) {
        skip_space();
        if (sql_.size() - pos_ < kw.size()) return false;
        for (size_t i = 0; i < kw.size(); ++i)
            if (upper(sql_[pos_ + i]) != kw[i]) return false;
        if (pos_ + kw.size() < sql_.size() && word_char(sql_[pos_ + kw.size()])) return false;
        pos_ += kw.size();
        return true;
    }

    bool symbol(std::string_view sym) {
        skip_space();
        if (sql_.substr(pos_, sym.size()) != sym) return false;
        pos_ += sym.size();
        return true;
    }

    static bool is_keyword(const std::string& w) {
        std::string u = w;
        for (char& c : u) c = upper(c);
        return u == "AND" || u == "OR" || u == "NOT" || u == "IN" || u == "IS" || u == "NULL" || u == "BETWEEN";
    }

    // A chain is collected and joined once, so its length costs O(n).
    Expr or_expr(bool neg) {
        vector<Expr> terms{and_expr(neg)};
        while (keyword("OR")) terms.push_back(and_expr(neg));
        return Expr::join(neg ? Expr::Op::And : Expr::Op::Or, std::move(terms));
    }

    Expr and_expr(bool neg) {
        vector<Expr> terms{not_expr(neg)};
        while (keyword("AND")) terms.push_back(not_expr(neg));
        return Expr::join(neg ? Expr::Op::Or : Expr::Op::And, std::move(terms));
    }

    // Each NOT and parenthesis recurses, so nesting is counted here and
    // deep input fails with a column instead of overflowing the stack.
    Expr not_expr(bool neg) {
        if (++depth_ > Expr::kMaxDepth)
            fail("filter nested more than " + std::to_string(Expr::kMaxDepth) + " levels deep");
        Expr e = unary(neg);
        --depth_;
        return e;
    }

    Expr unary(bool neg) {
        if (keyword("NOT")) return not_expr(!neg);
        if (symbol("(")) {
            Expr e = or_expr(neg);
            if (!symbol(")")) fail("expected ')'");
            return e;
        }
        return predicate(neg);
    }

    // "..." with "" as a quote inside, or a bare word that isn't a keyword.
    std::string name() {
        skip_space();
        if (pos_ < sql_.size() && sql_[pos_] == '"') return quoted('"');
        const size_t start = pos_;
        if (pos_ < sql_.size() && (is_alpha(sql_[pos_]) || sql_[pos_] == '_'))
            while (pos_ < sql_.size() && word_char(sql_[pos_])) ++pos_;
        if (pos_ < sql_.size() && !is_ascii(sql_[pos_])) {
            pos_ = start;
            fail("a column name with non-ASCII characters needs double quotes");
        }
        std::string w(sql_.substr(start, pos_ - start));
        if (w.empty() || is_keyword(w)) {
            pos_ = start;
            fail("expected a column name");
        }
        return w;
    }

    std::string quoted(char q) {
        const size_t start = pos_++;
        std::string out;
        while (true) {
            if (pos_ >= sql_.size()) {
                pos_ = start;
                fail(q == '\'' ? "unterminated string" : "unterminated quoted name");
            }
            if (sql_[pos_] == q) {
                if (pos_ + 1 < sql_.size() && sql_[pos_ + 1] == q) {
                    out += q;
                    pos_ += 2;
                    continue;
                }
                ++pos_;
                return out;
            }
            out += sql_[pos_++];
        }
    }

    Expr::Value value() {
        skip_space();
        if (pos_ >= sql_.size()) fail("expected a value");
        const char c = sql_[pos_];
        if (c == '\'') return quoted('\'');
        if (c == '?') {
            if (!list_) fail(named_ ? "? placeholder, but parameters were given by name" : "? placeholder, but no parameters were given");
            if (used_ >= list_->size()) fail("more ? placeholders than parameters");
            ++pos_;
            return (*list_)[used_++];
        }
        if (c == ':') {
            const size_t start = pos_++;
            while (pos_ < sql_.size() && word_char(sql_[pos_])) ++pos_;
            const std::string key(sql_.substr(start + 1, pos_ - start - 1));
            pos_ = start;
            if (key.empty()) fail("expected a parameter name after ':'");
            if (!named_) fail(list_ ? ":" + key + " placeholder, but parameters were given as a list" : ":" + key + " placeholder, but no parameters were given");
            auto it = named_->find(key);
            if (it == named_->end()) fail("no parameter named '" + key + "'");
            pos_ += key.size() + 1;
            return it->second;
        }
        // a number: optional sign, digits, optional fraction and exponent
        const size_t start = pos_;
        size_t i = pos_;
        if (sql_[i] == '-' || sql_[i] == '+') ++i;
        const size_t digits = i;
        bool is_float = false;
        while (i < sql_.size() && is_digit(sql_[i])) ++i;
        if (i < sql_.size() && sql_[i] == '.') {
            is_float = true;
            ++i;
            while (i < sql_.size() && is_digit(sql_[i])) ++i;
        }
        if (i == digits || (i == digits + 1 && is_float)) fail("expected a value");
        if (i < sql_.size() && (sql_[i] == 'e' || sql_[i] == 'E')) {
            size_t j = i + 1;
            if (j < sql_.size() && (sql_[j] == '-' || sql_[j] == '+')) ++j;
            if (j < sql_.size() && is_digit(sql_[j])) {
                is_float = true;
                i = j;
                while (i < sql_.size() && is_digit(sql_[i])) ++i;
            }
        }
        if (i < sql_.size() && word_char(sql_[i])) fail("expected a value");
        const std::string text(sql_.substr(start, i - start));
        pos_ = i;
        if (is_float) {
            const double d = std::strtod(text.c_str(), nullptr);
            if (std::isinf(d)) {
                pos_ = start;
                fail("number out of range");
            }
            return d;
        }
        errno = 0;
        const long long v = std::strtoll(text.c_str(), nullptr, 10);
        if (errno == ERANGE) {
            pos_ = start;
            fail("integer out of range");
        }
        return static_cast<int64_t>(v);
    }

    static Expr::Op negate(Expr::Op op) {
        switch (op) {
            case Expr::Op::Eq: return Expr::Op::Ne;
            case Expr::Op::Ne: return Expr::Op::Eq;
            case Expr::Op::Lt: return Expr::Op::Ge;
            case Expr::Op::Le: return Expr::Op::Gt;
            case Expr::Op::Gt: return Expr::Op::Le;
            default: return Expr::Op::Lt;  // Ge
        }
    }

    Expr predicate(bool neg) {
        if (++conditions_ > max_conditions_)
            fail("filter has more than " + std::to_string(max_conditions_) +
                 " conditions; combine values with IN (...) or raise the limit");
        const std::string field = name();
        if (keyword("IS")) {
            const bool is_not = keyword("NOT");
            if (!keyword("NULL")) fail("expected NULL");
            const Expr e = Expr::is_null(field);
            return neg != is_not ? ~e : e;
        }
        const bool not_kw = keyword("NOT");
        if (not_kw) neg = !neg;
        if (keyword("IN")) {
            if (!symbol("(")) fail("expected '('");
            vector<Expr::Value> values{value()};
            while (symbol(",")) values.push_back(value());
            if (!symbol(")")) fail("expected ',' or ')'");
            const Expr e = Expr::isin(field, std::move(values));
            return neg ? (~e & ~Expr::is_null(field)) : e;
        }
        if (keyword("BETWEEN")) {
            Expr::Value lo = value();
            if (!keyword("AND")) fail("expected AND");
            Expr::Value hi = value();
            if (neg) return Expr::compare(field, Expr::Op::Lt, std::move(lo)) | Expr::compare(field, Expr::Op::Gt, std::move(hi));
            return Expr::between(field, std::move(lo), std::move(hi));
        }
        if (not_kw) fail("expected IN or BETWEEN");
        Expr::Op op;
        if (symbol("<=")) op = Expr::Op::Le;
        else if (symbol(">=")) op = Expr::Op::Ge;
        else if (symbol("<>") || symbol("!=")) op = Expr::Op::Ne;
        else if (symbol("==") || symbol("=")) op = Expr::Op::Eq;
        else if (symbol("<")) op = Expr::Op::Lt;
        else if (symbol(">")) op = Expr::Op::Gt;
        else fail("expected a comparison, IN, BETWEEN or IS");
        return Expr::compare(field, neg ? negate(op) : op, value());
    }

    std::string_view sql_;
    size_t pos_ = 0;
    const vector<Expr::Value>* list_;
    const std::unordered_map<std::string, Expr::Value>* named_;
    size_t used_ = 0;
    size_t depth_ = 0;
    size_t conditions_ = 0;
    size_t max_conditions_;
};

} // namespace

Expr Expr::parse(std::string_view sql, const vector<Value>& params) {
    return SqlParser(sql, params.empty() ? nullptr : &params, nullptr).parse();
}

Expr Expr::parse(std::string_view sql, const std::unordered_map<std::string, Value>& params) {
    return SqlParser(sql, nullptr, &params).parse();
}

std::string Expr::to_string() const {
    switch (op()) {
        case Op::And:
        case Op::Or: {
            std::string s = "(";
            for (size_t i = 0; i < children().size(); ++i) s += (i ? (op() == Op::And ? " & " : " | ") : "") + children()[i].to_string();
            return s + ")";
        }
        case Op::Not: return "~" + children()[0].to_string();
        case Op::IsNull: return "is_null(" + field() + ")";
        case Op::In: {
            std::string s = field() + " in [";
            for (size_t i = 0; i < values().size(); ++i) s += (i ? ", " : "") + value_string(values()[i]);
            return s + "]";
        }
        default: return "(" + field() + " " + op_string(op()) + " " + value_string(values()[0]) + ")";
    }
}

namespace detail {

namespace {

ColumnType type_of(const MetadataColumn& c) {
    return static_cast<ColumnType>(c.values.index());  // variant order matches ColumnType
}

const char* type_name(ColumnType t) {
    return t == ColumnType::Int64 ? "int" : t == ColumnType::Float64 ? "float" : "string";
}

size_t length_of(const MetadataColumn& c) {
    return std::visit([](const auto& v) { return v.size(); }, c.values);
}

// A number's value as a double; throws on a string.
double as_double(const Expr::Value& v, const std::string& field) {
    if (auto* i = std::get_if<int64_t>(&v)) return static_cast<double>(*i);
    if (auto* d = std::get_if<double>(&v)) return *d;
    throw std::invalid_argument("metadata field '" + field + "' holds numbers; got the string " + value_string(v));
}

const std::string& as_string(const Expr::Value& v, const std::string& field) {
    if (auto* s = std::get_if<std::string>(&v)) return *s;
    throw std::invalid_argument("metadata field '" + field + "' holds strings; got " + value_string(v));
}

// Rows a pass over a column tests in the time it takes to set one listed
// row's bit; evaluate_leaf uses it to pick between them. Measured on an M3 with
// benchmarks/filter_sweep.cpp (1M rows): about 0.41 ns per scanned row,
// 1.7-4.5x that per listed row (more as lists grow), and an OR'd bitmap word
// costs less than a scanned row, so it counts as one.
constexpr size_t kScatterCost = 4;

// The same for a numeric column's sorted rows, which come in random row order
// and so don't stall on one word the way a string's ascending list does: about
// 0.63 ns a row against 0.38 ns per scanned row on 1M rows.
constexpr size_t kSortedScatterCost = 2;

// The same for testing one row picked from a set of bits (MetadataStore::narrow):
// its values are read out of order, so each costs more than a scanned row.
constexpr size_t kTestCost = 8;

// Sets bit i of words where row(i) holds, for i in [0, slots). Each word is
// built in a register without branches: filters often match about half the
// rows at random, where a branch would mispredict on most of them.
template <class Row>
void fill_bits(vector<uint64_t>& words, size_t slots, Row&& row) {
    for (size_t w = 0; w < words.size(); ++w) {
        const size_t first = w * 64, end = std::min(slots, first + 64);
        uint64_t bits = 0;
        for (size_t i = first; i < end; ++i) bits |= uint64_t{row(i)} << (i - first);
        words[w] = bits;
    }
}

// fill_bits for valid rows whose x[i] <op> value, with op chosen outside the loop.
template <class T>
void compare_rows(vector<uint64_t>& words, size_t slots, const vector<uint8_t>& valid, const T* x, Expr::Op op,
                  T value) {
    auto run = [&](auto pred) { fill_bits(words, slots, [&](size_t i) { return bool(valid[i] & pred(x[i])); }); };
    switch (op) {
        case Expr::Op::Eq: run([&](T a) { return a == value; }); break;
        case Expr::Op::Ne: run([&](T a) { return a != value; }); break;
        case Expr::Op::Lt: run([&](T a) { return a < value; }); break;
        case Expr::Op::Le: run([&](T a) { return a <= value; }); break;
        case Expr::Op::Gt: run([&](T a) { return a > value; }); break;
        case Expr::Op::Ge: run([&](T a) { return a >= value; }); break;
        default: break;
    }
}

} // namespace

// Whether values of type `given` can go into a column of type `col`: the
// same type, or ints into a float column (converted; the reverse would lose data).
static bool fits(ColumnType col, ColumnType given) {
    return col == given || (col == ColumnType::Float64 && given == ColumnType::Int64);
}

static std::invalid_argument wrong_type(const std::string& name, ColumnType col, ColumnType given) {
    return std::invalid_argument("metadata column '" + name + "' holds " + type_name(col) + " values; got " + type_name(given));
}

// The rules every column of n values meets; type is the column's, or nullptr
// when the values decide it (a column an add creates).
static void check_values(const MetadataColumn& c, size_t n, const ColumnType* type) {
    if (length_of(c) != n)
        throw std::invalid_argument("metadata column '" + c.name + "' has " + std::to_string(length_of(c)) +
                                    " values for " + std::to_string(n) + " vectors");
    if (!c.valid.empty() && c.valid.size() != n)
        throw std::invalid_argument("metadata column '" + c.name + "' has " + std::to_string(c.valid.size()) +
                                    " null flags for " + std::to_string(n) + " vectors");
    // an all-null column carries no values, so its type doesn't matter, but a new one has none to take
    const bool all_null = n > 0 && !c.valid.empty() && std::find(c.valid.begin(), c.valid.end(), 1) == c.valid.end();
    if (type && n > 0 && !all_null && !fits(*type, type_of(c))) throw wrong_type(c.name, *type, type_of(c));
    if (!type && all_null)
        throw std::invalid_argument("metadata column '" + c.name + "' is new and has only nulls, so it has no type");
}

void MetadataStore::check(size_t n, const Metadata& md) const {
    std::unordered_set<std::string> names;
    for (const MetadataColumn& c : md) {
        if (c.name.empty()) throw std::invalid_argument("metadata column names must not be empty");
        if (!names.insert(c.name).second) throw std::invalid_argument("metadata column '" + c.name + "' appears twice");
        const Column* existing = find(c.name);
        check_values(c, n, existing ? &existing->type : nullptr);
    }
}

void MetadataStore::check_new(const MetadataColumn& c, ColumnType type, size_t n) const {
    if (c.name.empty()) throw std::invalid_argument("metadata column names must not be empty");
    if (find(c.name)) throw std::invalid_argument("metadata column '" + c.name + "' already exists");
    check_values(c, n, &type);
}

void MetadataStore::check_update(size_t n, const Metadata& md) const {
    for (const MetadataColumn& c : md)
        if (!find(c.name))
            throw std::invalid_argument("the index has no metadata column '" + c.name +
                                        "'; add it with add_metadata_column");
    check(n, md);
}

void MetadataStore::write_row(Column& col, size_t to, const MetadataColumn& c, size_t i) {
    const bool has = c.valid.empty() || c.valid[i];
    col.valid[to] = has;
    if (col.type == ColumnType::Int64) col.ints[to] = has ? std::get<vector<int64_t>>(c.values)[i] : 0;
    else if (col.type == ColumnType::Float64) {
        const auto* ints = std::get_if<vector<int64_t>>(&c.values);  // ints fit a float column
        col.floats[to] = !has ? 0.0 : ints ? static_cast<double>((*ints)[i]) : std::get<vector<double>>(c.values)[i];
    }
    else if (!has) col.codes[to] = 0;
    else {
        const std::string& s = std::get<vector<std::string>>(c.values)[i];
        auto [pos, added] = col.code_of.emplace(s, static_cast<uint32_t>(col.dict.size()));
        if (added) col.dict.push_back(s);
        col.codes[to] = pos->second;
    }
}

void MetadataStore::update(const vector<uint32_t>& slots, const Metadata& md) {
    if (slots.empty()) return;
    for (const MetadataColumn& c : md) {
        Column* col = nullptr;
        for (Column& existing : columns_)
            if (existing.name == c.name) col = &existing;
        assert(col);
        if (col->type != ColumnType::String) {
            for (size_t i = 0; i < slots.size(); ++i) write_row(*col, slots[i], c, i);
            for (uint32_t s : slots)
                if (s < col->sorted_rows) col->stale.push_back(s);
            col->maybe_sort();
            continue;
        }
        vector<uint32_t> old_codes(slots.size()), touched;
        for (size_t i = 0; i < slots.size(); ++i) {
            old_codes[i] = col->codes[slots[i]];
            if (col->valid[slots[i]]) touched.push_back(old_codes[i]);
        }
        col->unindex(slots, old_codes);
        for (size_t i = 0; i < slots.size(); ++i) write_row(*col, slots[i], c, i);
        col->rows_of.resize(col->dict.size());
        for (uint32_t s : slots) {
            col->index_row(s);
            if (col->valid[s]) touched.push_back(col->codes[s]);
        }
        col->rebalance(touched);
    }
}

void MetadataStore::Column::index_rows(size_t first) {
    rows_of.resize(dict.size());
    vector<uint32_t> touched;
    for (size_t s = first; s < rows(); ++s) {
        index_row(s);
        if (valid[s]) touched.push_back(codes[s]);
    }
    rebalance(touched);
}

void MetadataStore::Column::index_row(size_t s) {
    if (!valid[s]) return;
    CodeRows& r = rows_of[codes[s]];
    if (r.bits.empty()) r.rows.push_back(static_cast<uint32_t>(s));
    else {
        if (r.bits.size() <= s / 64) r.bits.resize(s / 64 + 1, 0);
        r.bits[s / 64] |= uint64_t{1} << (s % 64);
    }
    ++r.count;
}

void MetadataStore::Column::reindex() {
    rows_of.clear();
    balanced_rows = 0;
    index_rows(0);
}

void MetadataStore::Column::unindex(const vector<uint32_t>& rows, const vector<uint32_t>& old_codes) {
    std::unordered_map<uint32_t, std::unordered_set<uint32_t>> gone;  // code -> its listed rows to drop
    for (size_t i = 0; i < rows.size(); ++i) {
        if (!valid[rows[i]]) continue;
        CodeRows& r = rows_of[old_codes[i]];
        --r.count;
        if (r.bits.empty()) gone[old_codes[i]].insert(rows[i]);
        else r.bits[rows[i] / 64] &= ~(uint64_t{1} << (rows[i] % 64));
    }
    for (auto& [code, drop] : gone) {
        vector<uint32_t>& list = rows_of[code].rows;
        list.erase(std::remove_if(list.begin(), list.end(), [&](uint32_t s) { return drop.count(s); }), list.end());
    }
}

void MetadataStore::Column::rebalance(const vector<uint32_t>& touched) {
    // every code only each time the column grows by a quarter, so a small add
    // into a column of many distinct strings costs its own rows, not the dictionary
    if (rows() * 4 >= balanced_rows * 5) {
        balanced_rows = rows();
        for (CodeRows& r : rows_of) balance(r);
    } else {
        for (uint32_t c : touched) balance(rows_of[c]);
    }
}

void MetadataStore::Column::balance(CodeRows& r) const {
    const size_t n = rows();
    if (r.bits.empty() && r.count >= 64 && r.count * 32 >= n) {
        r.bits.assign((n + 63) / 64, 0);
        for (uint32_t s : r.rows) r.bits[s / 64] |= uint64_t{1} << (s % 64);
        vector<uint32_t>().swap(r.rows);
    } else if (!r.bits.empty() && r.count * 64 < n) {
        r.rows.reserve(r.count);
        for (size_t w = 0; w < r.bits.size(); ++w)
            for (uint64_t b = r.bits[w]; b; b &= b - 1)
                r.rows.push_back(static_cast<uint32_t>(w * 64 + std::countr_zero(b)));
        vector<uint64_t>().swap(r.bits);
    }
}

void MetadataStore::Column::sort_rows() {
    const bool ints_col = type == ColumnType::Int64;
    sorted.clear();
    stale.clear();
    for (size_t s = 0; s < rows(); ++s)
        if (valid[s] && (ints_col || !std::isnan(floats[s]))) sorted.push_back(static_cast<uint32_t>(s));
    sorted_ints.clear();
    sorted_floats.clear();
    if (ints_col) {
        std::sort(sorted.begin(), sorted.end(), [&](uint32_t a, uint32_t b) { return ints[a] < ints[b]; });
        for (uint32_t s : sorted) sorted_ints.push_back(ints[s]);
    } else {
        std::sort(sorted.begin(), sorted.end(), [&](uint32_t a, uint32_t b) { return floats[a] < floats[b]; });
        for (uint32_t s : sorted) sorted_floats.push_back(floats[s]);
    }
    sorted_rows = rows();
}

void MetadataStore::Column::maybe_sort() {
    if ((rows() - sorted_rows + stale.size()) * 8 > rows()) sort_rows();
}

void MetadataStore::add_column(std::string name, ColumnType type, size_t rows, const std::optional<Expr::Value>& value) {
    if (name.empty()) throw std::invalid_argument("metadata column names must not be empty");
    if (find(name)) throw std::invalid_argument("metadata column '" + name + "' already exists");
    const ColumnType given = value ? static_cast<ColumnType>(value->index()) : type;  // Value's order matches ColumnType
    if (!fits(type, given)) throw wrong_type(name, type, given);
    Column col{std::move(name), type, vector<uint8_t>(rows, value.has_value()), {}, {}, {}, {}, {}};
    if (type == ColumnType::Int64) {
        col.ints.assign(rows, value ? std::get<int64_t>(*value) : 0);
    } else if (type == ColumnType::Float64) {
        const auto* i = value ? std::get_if<int64_t>(&*value) : nullptr;
        col.floats.assign(rows, !value ? 0.0 : i ? static_cast<double>(*i) : std::get<double>(*value));
    }
    if (type != ColumnType::String) {
        col.sort_rows();
    } else {
        col.codes.assign(rows, 0);
        if (value && rows > 0) {
            col.dict.push_back(std::get<std::string>(*value));
            col.code_of.emplace(col.dict[0], 0);
        }
        col.reindex();
    }
    columns_.push_back(std::move(col));
}

void MetadataStore::drop_column(const std::string& name) {
    auto it = std::find_if(columns_.begin(), columns_.end(), [&](const Column& c) { return c.name == name; });
    if (it == columns_.end()) throw std::invalid_argument("the index has no metadata column '" + name + "'");
    columns_.erase(it);
}

void MetadataStore::append(size_t rows, size_t n, const Metadata& md) {
    std::unordered_map<std::string, const MetadataColumn*> given;
    for (const MetadataColumn& c : md) given.emplace(c.name, &c);

    for (const MetadataColumn& c : md) {
        // an add of no vectors creates no columns: its empty values say nothing about the type
        if (n == 0 || find(c.name)) continue;
        add_column(c.name, type_of(c), rows);
    }

    for (Column& col : columns_) {
        assert(col.rows() == rows);
        auto it = given.find(col.name);
        if (it == given.end() || n == 0) {  // left out of this add: null
            col.valid.insert(col.valid.end(), n, 0);
            if (col.type == ColumnType::Int64) col.ints.insert(col.ints.end(), n, 0);
            else if (col.type == ColumnType::Float64) col.floats.insert(col.floats.end(), n, 0.0);
            if (col.type != ColumnType::String) col.maybe_sort();
            else {
                col.codes.insert(col.codes.end(), n, 0);
                col.rebalance({});
            }
            continue;
        }
        col.valid.resize(rows + n);
        col.ints.resize(col.type == ColumnType::Int64 ? rows + n : 0);
        col.floats.resize(col.type == ColumnType::Float64 ? rows + n : 0);
        col.codes.resize(col.type == ColumnType::String ? rows + n : 0);
        for (size_t i = 0; i < n; ++i) write_row(col, rows + i, *it->second, i);
        if (col.type == ColumnType::String) col.index_rows(rows);
        else col.maybe_sort();
    }
}

void MetadataStore::compact(const vector<uint32_t>& remap) {
    for (Column& col : columns_) {
        assert(col.rows() == remap.size());
        size_t kept = 0;
        for (size_t old = 0; old < remap.size(); ++old) {
            const uint32_t to = remap[old];
            if (to == IdMap::kRemoved) continue;
            col.valid[to] = col.valid[old];
            if (col.type == ColumnType::Int64) col.ints[to] = col.ints[old];
            else if (col.type == ColumnType::Float64) col.floats[to] = col.floats[old];
            else col.codes[to] = col.codes[old];
            ++kept;
        }
        col.valid.resize(kept);
        col.ints.resize(col.type == ColumnType::Int64 ? kept : 0);
        col.floats.resize(col.type == ColumnType::Float64 ? kept : 0);
        col.codes.resize(col.type == ColumnType::String ? kept : 0);
        if (col.type != ColumnType::String) col.sort_rows();
    }
    drop_unused_strings();
}

void MetadataStore::drop_unused_strings() {
    for (Column& col : columns_) {
        if (col.type != ColumnType::String) continue;
        // renumber the strings rows still use, in first-seen order
        vector<uint32_t> renum(col.dict.size(), UINT32_MAX);
        vector<std::string> dict;
        for (size_t i = 0; i < col.rows(); ++i) {
            if (!col.valid[i]) {
                col.codes[i] = 0;
                continue;
            }
            uint32_t& code = renum[col.codes[i]];
            if (code == UINT32_MAX) {
                code = static_cast<uint32_t>(dict.size());
                dict.push_back(std::move(col.dict[col.codes[i]]));
            }
            col.codes[i] = code;
        }
        col.dict = std::move(dict);
        col.code_of.clear();
        for (uint32_t c = 0; c < col.dict.size(); ++c) col.code_of.emplace(col.dict[c], c);
        col.reindex();
    }
}

const MetadataStore::Column* MetadataStore::find(const std::string& name) const {
    for (const Column& col : columns_)
        if (col.name == name) return &col;
    return nullptr;
}

namespace {

// Whether a and b are the same filter, node for node (values compared with ==,
// so a NaN value matches nothing).
bool same(const Expr& a, const Expr& b) {
    if (a.op() != b.op() || a.field() != b.field() || a.values() != b.values() ||
        a.children().size() != b.children().size())
        return false;
    for (size_t i = 0; i < a.children().size(); ++i)
        if (!same(a.children()[i], b.children()[i])) return false;
    return true;
}

// An And's children, or the filter itself: the terms an Or branch is the And of.
vector<Expr> terms_of(const Expr& e) {
    if (e.op() == Expr::Op::And) return e.children();
    return {e};
}

// e with each condition that two or more branches of an Or share pulled out
// in front: (a & x) | (a & y) | z becomes (a & (x | y)) | z, so a is
// evaluated once and x | y only on a's rows. A branch that is only a (a | (a
// & x)) absorbs the rest of its group. The rows matched never change: the
// filter logic is two-valued (nulls fail, ~ is a complement), where & and |
// distribute exactly.
Expr factor(const Expr& e) {
    const Expr::Op op = e.op();
    if (op == Expr::Op::Not) return ~factor(e.children()[0]);
    if (op != Expr::Op::And && op != Expr::Op::Or) return e;
    vector<Expr> kids;
    kids.reserve(e.children().size());
    for (const Expr& c : e.children()) kids.push_back(factor(c));
    if (op == Expr::Op::And) return Expr::join(op, std::move(kids));

    while (kids.size() > 1 && std::any_of(kids.begin(), kids.end(), [](const Expr& k) { return k.op() == Expr::Op::And; })) {
        // the term in the most branches, keyed by to_string and checked with same()
        std::unordered_map<std::string, vector<size_t>> branches;  // term -> branches holding it
        vector<vector<Expr>> terms(kids.size());
        for (size_t k = 0; k < kids.size(); ++k) {
            terms[k] = terms_of(kids[k]);
            for (const Expr& t : terms[k]) {
                vector<size_t>& with = branches[t.to_string()];
                if (with.empty() || with.back() != k) with.push_back(k);
            }
        }
        const Expr* shared = nullptr;
        vector<size_t> group;
        std::string shared_key;  // ties go to the smallest key, so the result doesn't depend on hashing
        for (const auto& [key, with] : branches) {
            if (with.size() < 2 || with.size() < group.size()) continue;
            const Expr* t = nullptr;
            for (const Expr& x : terms[with[0]])
                if (x.to_string() == key) t = &x;
            vector<size_t> holding;
            for (size_t k : with)
                if (std::any_of(terms[k].begin(), terms[k].end(), [&](const Expr& x) { return same(x, *t); })) holding.push_back(k);
            if (holding.size() >= 2 && (holding.size() > group.size() || (holding.size() == group.size() && key < shared_key))) {
                shared = t;
                group = std::move(holding);
                shared_key = key;
            }
        }
        if (!shared) break;

        const Expr t = *shared;
        vector<Expr> rests;
        bool absorbed = false;  // a branch is just t, so the group is t
        for (size_t k : group) {
            vector<Expr> rest;
            for (const Expr& x : terms[k])
                if (!same(x, t)) rest.push_back(x);
            if (rest.empty()) absorbed = true;
            else rests.push_back(Expr::join(Expr::Op::And, std::move(rest)));
        }
        vector<Expr> next;
        next.push_back(absorbed ? t : t & factor(Expr::join(Expr::Op::Or, std::move(rests))));
        for (size_t k = 0; k < kids.size(); ++k)
            if (std::find(group.begin(), group.end(), k) == group.end()) next.push_back(std::move(kids[k]));
        kids = std::move(next);
    }
    return Expr::join(Expr::Op::Or, std::move(kids));
}

size_t count_bits(const vector<uint64_t>& words) {
    size_t n = 0;
    for (uint64_t w : words) n += static_cast<size_t>(std::popcount(w));
    return n;
}

// Whether `count` rows are few enough to work on as a row list: one is then
// smaller than a bitmap of every slot (4 bytes a row against a bit a slot).
bool few_rows(size_t count, size_t slots) { return count * 32 < slots; }

vector<uint32_t> rows_of_bits(const vector<uint64_t>& words) {
    vector<uint32_t> rows;
    for (size_t w = 0; w < words.size(); ++w)
        for (uint64_t b = words[w]; b; b &= b - 1) rows.push_back(static_cast<uint32_t>(w * 64 + std::countr_zero(b)));
    return rows;
}

vector<uint64_t> bits_of_rows(const vector<uint32_t>& rows, size_t slots) {
    vector<uint64_t> words((slots + 63) / 64, 0);
    for (uint32_t s : rows) words[s / 64] |= uint64_t{1} << (s % 64);
    return words;
}

// The rows of a (sorted) that aren't in b (sorted, a subset of a).
vector<uint32_t> without(const vector<uint32_t>& a, const vector<uint32_t>& b) {
    vector<uint32_t> out;
    out.reserve(a.size() - b.size());
    std::set_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(out));
    return out;
}

} // namespace

vector<uint64_t> MetadataStore::evaluate(const Expr& where, size_t slots) const {
    // factoring can nest one level deeper per group; past Expr::kMaxDepth, evaluate as given
    std::optional<Expr> factored;
    try {
        factored = factor(where);
    } catch (const std::invalid_argument&) {
    }
    return evaluate_tree(factored ? *factored : where, slots, nullptr);
}

MetadataStore::Estimate MetadataStore::estimate_tree(const Expr& where, size_t slots) const {
    switch (where.op()) {
        case Expr::Op::And:
        case Expr::Op::Or: {
            const bool all = where.op() == Expr::Op::And;
            Estimate est{all ? slots : 0, 0};
            for (const Expr& c : where.children()) {
                const Estimate e = estimate_tree(c, slots);
                est.matches = all ? std::min(est.matches, e.matches) : std::min(slots, est.matches + e.matches);
                est.cost += e.cost;
            }
            return est;
        }
        case Expr::Op::Not: return {slots, estimate_tree(where.children()[0], slots).cost};
        default: {
            const Column* col = find(where.field());
            if (!col) throw std::invalid_argument("unknown metadata field '" + where.field() + "'");
            return estimate(*col, {&where}, slots);
        }
    }
}

vector<uint64_t> MetadataStore::evaluate_tree(const Expr& where, size_t slots, const vector<uint64_t>* within,
                                              size_t inside) const {
    if (within && few_rows(inside, slots)) {
        vector<uint32_t> rows = rows_of_bits(*within);
        evaluate_rows(where, rows, slots);
        return bits_of_rows(rows, slots);
    }
    auto mask = [&](vector<uint64_t> words) {
        if (within)
            for (size_t w = 0; w < words.size(); ++w) words[w] &= (*within)[w];
        return words;
    };
    switch (where.op()) {
        case Expr::Op::And: {
            // Parts: each child, except that comparisons on one numeric column
            // form one part, a single run of its sorted rows (as in a BETWEEN,
            // though each alone may match too many). The part estimated to
            // match fewest rows runs first; each later part then works only on
            // the rows still set: a leaf part tests just those rows when that is
            // cheaper than its own evaluation, and a nested part passes them down.
            struct Part {
                const Column* col;          // leaf parts
                vector<const Expr*> leaves;
                const Expr* other;          // And / Or / Not children
                Estimate est;
            };
            vector<Part> parts;
            for (const Expr& child : where.children()) {
                const Expr::Op op = child.op();
                if (op == Expr::Op::And || op == Expr::Op::Or || op == Expr::Op::Not) {
                    parts.push_back({nullptr, {}, &child, estimate_tree(child, slots)});
                    continue;
                }
                const Column* col = find(child.field());
                if (!col) throw std::invalid_argument("unknown metadata field '" + child.field() + "'");
                const bool range = is_comparison(op) && op != Expr::Op::Ne && col->type != ColumnType::String;
                auto it = !range ? parts.end() : std::find_if(parts.begin(), parts.end(), [&](const Part& p) {
                    return p.col == col && is_comparison(p.leaves[0]->op()) && p.leaves[0]->op() != Expr::Op::Ne;
                });
                if (it == parts.end()) parts.push_back({col, {&child}, nullptr, {}});
                else it->leaves.push_back(&child);
            }
            for (Part& p : parts)
                if (p.col) p.est = estimate(*p.col, p.leaves, slots);
            std::stable_sort(parts.begin(), parts.end(), [](const Part& x, const Part& y) { return x.est.matches < y.est.matches; });

            auto run = [&](const Part& p) {
                vector<uint64_t> words((slots + 63) / 64, 0);
                if (p.leaves.size() > 1 && sorted_leaves(*p.col, p.leaves, words, slots)) return words;
                words = evaluate_leaf(*p.leaves[0], slots);
                for (size_t l = 1; l < p.leaves.size(); ++l) {
                    const vector<uint64_t> b = evaluate_leaf(*p.leaves[l], slots);
                    for (size_t w = 0; w < words.size(); ++w) words[w] &= b[w];
                }
                return words;
            };
            vector<uint64_t> a;
            size_t i = 0;
            if (within) a = *within;
            else if (parts[0].other) a = evaluate_tree(*parts[i++].other, slots, nullptr);
            else a = run(parts[i++]);
            size_t left = within ? inside : count_bits(a);
            for (; i < parts.size(); ++i) {
                if (few_rows(left, slots)) {  // the rest on the rows left alone
                    vector<uint32_t> rows = rows_of_bits(a);
                    for (; i < parts.size(); ++i) {
                        if (parts[i].other) evaluate_rows(*parts[i].other, rows, slots);
                        else
                            for (const Expr* leaf : parts[i].leaves) evaluate_rows(*leaf, rows, slots);
                    }
                    return bits_of_rows(rows, slots);
                }
                if (parts[i].other) {
                    a = evaluate_tree(*parts[i].other, slots, &a, left);
                    left = count_bits(a);
                } else if (left * kTestCost < parts[i].est.cost) {
                    for (const Expr* leaf : parts[i].leaves) narrow(*leaf, a);
                    left = count_bits(a);
                } else {
                    const vector<uint64_t> b = run(parts[i]);
                    left = 0;
                    for (size_t w = 0; w < a.size(); ++w) {
                        a[w] &= b[w];
                        left += static_cast<size_t>(std::popcount(a[w]));
                    }
                }
            }
            return a;
        }
        case Expr::Op::Or: {
            if (!within) {
                vector<uint64_t> a = evaluate_tree(where.children()[0], slots, nullptr);
                for (size_t c = 1; c < where.children().size(); ++c) {
                    const vector<uint64_t> b = evaluate_tree(where.children()[c], slots, nullptr);
                    for (size_t w = 0; w < a.size(); ++w) a[w] |= b[w];
                }
                return a;
            }
            // each branch only on the rows of within no earlier branch matched
            vector<uint64_t> left = *within, a(within->size(), 0);
            size_t remaining = inside;
            for (const Expr& c : where.children()) {
                const vector<uint64_t> b = evaluate_tree(c, slots, &left, remaining);
                remaining = 0;
                for (size_t w = 0; w < a.size(); ++w) {
                    a[w] |= b[w];
                    left[w] &= ~b[w];
                    remaining += static_cast<size_t>(std::popcount(left[w]));
                }
            }
            return a;
        }
        case Expr::Op::Not: {
            vector<uint64_t> a = evaluate_tree(where.children()[0], slots, within, inside);
            for (uint64_t& w : a) w = ~w;
            if (slots % 64) a.back() &= (uint64_t{1} << (slots % 64)) - 1;  // padding bits past the last slot
            return mask(std::move(a));
        }
        default: {
            if (!within) return evaluate_leaf(where, slots);
            const Column* col = find(where.field());
            if (!col) throw std::invalid_argument("unknown metadata field '" + where.field() + "'");
            if (inside * kTestCost < estimate(*col, {&where}, slots).cost) {
                vector<uint64_t> a = *within;
                narrow(where, a);
                return a;
            }
            return mask(evaluate_leaf(where, slots));
        }
    }
}

bool MetadataStore::sorted_leaves(const Column& col, const vector<const Expr*>& leaves, vector<uint64_t>& words,
                                  size_t slots, Estimate* estimate) {
    // each value compares exactly (an int column against an int) or in double, as in the pass
    const bool ints = col.type == ColumnType::Int64;
    struct Bound {
        bool exact;
        int64_t i;
        double d;
    };
    struct Cond {
        Expr::Op op;  // In becomes Eq against each value
        vector<Bound> bounds;
    };
    vector<Cond> conds;
    for (const Expr* leaf : leaves) {
        Cond c{leaf->op() == Expr::Op::In ? Expr::Op::Eq : leaf->op(), {}};
        if (c.op == Expr::Op::Ne) {  // matches nearly every row (and NaN rows), so it scans
            if (estimate) *estimate = {slots, slots};
            return false;
        }
        for (const auto& v : leaf->values()) {
            const auto* i = std::get_if<int64_t>(&v);
            if (ints && i) c.bounds.push_back({true, *i, 0.0});
            else c.bounds.push_back({false, 0, as_double(v, leaf->field())});
        }
        conds.push_back(std::move(c));
    }
    auto holds = [&](int64_t i, double d, Expr::Op o, const Bound& b) {  // a value: i in an int column, else d
        if (b.exact) return compare(o, i, b.i);
        return compare(o, ints ? static_cast<double>(i) : d, b.d);
    };
    // the first of sorted's positions [from, n) whose value at sort time fails `o b`
    const size_t n = col.sorted.size();
    auto first_failing = [&](size_t from, Expr::Op o, const Bound& b) {
        size_t lo = from, hi = n;
        while (lo < hi) {
            const size_t mid = lo + (hi - lo) / 2;
            const bool ok = ints ? holds(col.sorted_ints[mid], 0.0, o, b) : holds(0, col.sorted_floats[mid], o, b);
            if (ok) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    };

    // each bound's matching rows are one run of sorted, so a leaf's are its
    // bounds' runs, and an And's the overlaps of its leaves' runs
    using Run = std::pair<size_t, size_t>;
    auto runs_of = [&](const Cond& c) {
        vector<Run> runs;
        for (const Bound& b : c.bounds) {
            if (!b.exact && std::isnan(b.d)) continue;  // compares false with every row
            const size_t lo = first_failing(0, Expr::Op::Lt, b);
            const size_t hi = first_failing(lo, Expr::Op::Le, b);
            runs.push_back(c.op == Expr::Op::Lt   ? Run{0, lo}
                           : c.op == Expr::Op::Le ? Run{0, hi}
                           : c.op == Expr::Op::Gt ? Run{hi, n}
                           : c.op == Expr::Op::Ge ? Run{lo, n}
                                                  : Run{lo, hi});
        }
        return runs;
    };
    vector<Run> runs = runs_of(conds[0]);
    for (size_t c = 1; c < conds.size(); ++c) {
        vector<Run> both;
        for (const Run& q : runs_of(conds[c]))
            for (const Run& r : runs)
                if (std::max(q.first, r.first) < std::min(q.second, r.second))
                    both.push_back({std::max(q.first, r.first), std::min(q.second, r.second)});
        runs = std::move(both);
    }
    size_t cost = col.rows() - col.sorted_rows + col.stale.size() * kScatterCost;
    size_t matches = col.rows() - col.sorted_rows + col.stale.size();  // at most, among the unsorted rows
    for (const Run& r : runs) {
        cost += (r.second - r.first) * kSortedScatterCost;
        matches += r.second - r.first;
    }
    if (estimate) {
        *estimate = {matches, std::min(cost, slots)};
        return false;
    }
    if (cost >= slots) return false;

    auto set = [&](size_t s) { words[s / 64] |= uint64_t{1} << (s % 64); };
    for (auto [a, b] : runs)
        for (size_t k = a; k < b; ++k) set(col.sorted[k]);
    // rows sorted under an old value, and rows added since the sort, are tested directly
    auto test = [&](size_t s) {
        if (!col.valid[s]) return false;
        const int64_t i = ints ? col.ints[s] : 0;
        const double d = ints ? 0.0 : col.floats[s];
        return std::all_of(conds.begin(), conds.end(), [&](const Cond& c) {
            return std::any_of(c.bounds.begin(), c.bounds.end(), [&](const Bound& b) { return holds(i, d, c.op, b); });
        });
    };
    for (uint32_t s : col.stale) words[s / 64] &= ~(uint64_t{1} << (s % 64));
    for (uint32_t s : col.stale)
        if (test(s)) set(s);
    for (size_t s = col.sorted_rows; s < col.rows(); ++s)
        if (test(s)) set(s);
    return true;
}

vector<uint8_t> MetadataStore::string_match(const Column& col, const Expr& leaf) {
    // decide each distinct string once, so rows can be looked up by code
    vector<uint8_t> match(col.dict.size(), 0);
    if (leaf.op() == Expr::Op::In || leaf.op() == Expr::Op::Eq) {
        for (const auto& v : leaf.values()) {
            auto it = col.code_of.find(as_string(v, leaf.field()));
            if (it != col.code_of.end()) match[it->second] = 1;
        }
    } else {
        const std::string& value = as_string(leaf.values()[0], leaf.field());
        for (size_t c = 0; c < col.dict.size(); ++c) match[c] = compare(leaf.op(), col.dict[c], value);
    }
    return match;
}

MetadataStore::Estimate MetadataStore::estimate(const Column& col, const vector<const Expr*>& leaves, size_t slots) const {
    const Expr& leaf = *leaves[0];
    if (leaf.op() == Expr::Op::IsNull) return {slots, slots};
    if (col.type == ColumnType::String) {
        const vector<uint8_t> match = string_match(col, leaf);
        size_t matches = 0, cost = 0;
        for (size_t c = 0; c < col.dict.size(); ++c) {
            if (!match[c]) continue;
            const Column::CodeRows& r = col.rows_of[c];
            matches += r.count;
            cost += r.bits.empty() ? r.count * kScatterCost : r.bits.size();
        }
        return {matches, std::min(cost, slots)};
    }
    Estimate est;
    vector<uint64_t> unused;
    sorted_leaves(col, leaves, unused, slots, &est);
    return est;
}

namespace {

// Clears each set bit of words whose row fails test(row).
template <class Test>
void keep_if(vector<uint64_t>& words, Test&& test) {
    for (size_t w = 0; w < words.size(); ++w) {
        uint64_t kept = words[w];
        for (uint64_t b = words[w]; b; b &= b - 1) {
            const int bit = std::countr_zero(b);
            if (!test(w * 64 + bit)) kept &= ~(uint64_t{1} << bit);
        }
        words[w] = kept;
    }
}

// Drops each row of a sorted row list that fails test(row), keeping the order.
template <class Test>
void keep_if(vector<uint32_t>& rows, Test&& test) {
    rows.erase(std::remove_if(rows.begin(), rows.end(), [&](uint32_t s) { return !test(s); }), rows.end());
}

// keep_if for valid rows whose V(x[row]) <op> value, with op chosen outside the loop.
template <class Set, class V, class T>
void keep_compared(Set& set, const vector<uint8_t>& valid, const T* x, Expr::Op op, V value) {
    auto run = [&](auto pred) { keep_if(set, [&](size_t s) { return valid[s] && pred(static_cast<V>(x[s])); }); };
    switch (op) {
        case Expr::Op::Eq: run([&](V a) { return a == value; }); break;
        case Expr::Op::Ne: run([&](V a) { return a != value; }); break;
        case Expr::Op::Lt: run([&](V a) { return a < value; }); break;
        case Expr::Op::Le: run([&](V a) { return a <= value; }); break;
        case Expr::Op::Gt: run([&](V a) { return a > value; }); break;
        case Expr::Op::Ge: run([&](V a) { return a >= value; }); break;
        default: break;
    }
}

} // namespace

template <class Set>
void MetadataStore::narrow(const Expr& leaf, Set& words) const {
    const Column* col = find(leaf.field());
    if (!col) throw std::invalid_argument("unknown metadata field '" + leaf.field() + "'");
    const Expr::Op op = leaf.op();
    const vector<uint8_t>& valid = col->valid;

    if (op == Expr::Op::IsNull) {
        keep_if(words, [&](size_t s) { return !valid[s]; });
    } else if (col->type == ColumnType::String) {
        const vector<uint8_t> match = string_match(*col, leaf);
        const uint32_t* codes = col->codes.data();
        keep_if(words, [&](size_t s) { return valid[s] && match[codes[s]]; });
    } else if (col->type == ColumnType::Int64) {
        const int64_t* x = col->ints.data();
        if (op == Expr::Op::In) {
            std::unordered_set<int64_t> ints;
            std::unordered_set<double> floats;
            for (const auto& v : leaf.values()) {
                if (auto* i = std::get_if<int64_t>(&v)) ints.insert(*i);
                else floats.insert(as_double(v, leaf.field()));
            }
            keep_if(words, [&](size_t s) {
                return valid[s] && (ints.count(x[s]) || (!floats.empty() && floats.count(static_cast<double>(x[s]))));
            });
        } else if (auto* value = std::get_if<int64_t>(&leaf.values()[0])) {
            keep_compared(words, valid, x, op, *value);
        } else {
            keep_compared(words, valid, x, op, as_double(leaf.values()[0], leaf.field()));
        }
    } else {
        const double* x = col->floats.data();
        if (op == Expr::Op::In) {
            std::unordered_set<double> wanted;
            for (const auto& v : leaf.values()) wanted.insert(as_double(v, leaf.field()));
            keep_if(words, [&](size_t s) { return valid[s] && wanted.count(x[s]); });
        } else {
            keep_compared(words, valid, x, op, as_double(leaf.values()[0], leaf.field()));
        }
    }
}

void MetadataStore::evaluate_rows(const Expr& where, vector<uint32_t>& rows, size_t slots) const {
    switch (where.op()) {
        case Expr::Op::And: {
            vector<std::pair<size_t, const Expr*>> kids;  // estimated matches, child: fewest first
            for (const Expr& c : where.children()) kids.push_back({estimate_tree(c, slots).matches, &c});
            std::stable_sort(kids.begin(), kids.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
            for (const auto& [matches, c] : kids) evaluate_rows(*c, rows, slots);
            return;
        }
        case Expr::Op::Or: {
            // each branch only on the rows no earlier branch matched
            vector<uint32_t> left = rows, matched;
            for (const Expr& c : where.children()) {
                vector<uint32_t> got = left;
                evaluate_rows(c, got, slots);
                left = without(left, got);
                matched.insert(matched.end(), got.begin(), got.end());
            }
            std::sort(matched.begin(), matched.end());
            rows = std::move(matched);
            return;
        }
        case Expr::Op::Not: {
            vector<uint32_t> got = rows;
            evaluate_rows(where.children()[0], got, slots);
            rows = without(rows, got);
            return;
        }
        default: {
            const Column* col = find(where.field());
            if (!col) throw std::invalid_argument("unknown metadata field '" + where.field() + "'");
            if (estimate(*col, {&where}, slots).cost < rows.size() * kTestCost) {
                const vector<uint64_t> words = evaluate_leaf(where, slots);
                keep_if(rows, [&](size_t s) { return (words[s / 64] >> (s % 64)) & 1; });
            } else {
                narrow(where, rows);
            }
        }
    }
}

vector<uint64_t> MetadataStore::evaluate_leaf(const Expr& leaf, size_t slots) const {
    const Column* col = find(leaf.field());
    if (!col) throw std::invalid_argument("unknown metadata field '" + leaf.field() + "'");
    assert(col->rows() == slots);

    vector<uint64_t> words((slots + 63) / 64, 0);
    const Expr::Op op = leaf.op();
    const vector<uint8_t>& valid = col->valid;

    if (op == Expr::Op::IsNull) {
        fill_bits(words, slots, [&](size_t i) { return !valid[i]; });
        return words;
    }

    if (col->type == ColumnType::String) {
        const vector<uint8_t> match = string_match(*col, leaf);
        // a few matching strings: set their rows' bits instead of passing over every row
        size_t cost = 0;
        for (size_t c = 0; c < col->dict.size(); ++c)
            if (match[c]) cost += col->rows_of[c].bits.empty() ? col->rows_of[c].count * kScatterCost : col->rows_of[c].bits.size();
        if (cost < slots) {
            for (size_t c = 0; c < col->dict.size(); ++c) {
                if (!match[c]) continue;
                const Column::CodeRows& r = col->rows_of[c];
                for (size_t w = 0; w < r.bits.size(); ++w) words[w] |= r.bits[w];
                for (uint32_t s : r.rows) words[s / 64] |= uint64_t{1} << (s % 64);
            }
            return words;
        }
        // a null row's code is 0, which may not be in an empty dictionary, so test valid first
        const uint32_t* codes = col->codes.data();
        fill_bits(words, slots, [&](size_t i) { return valid[i] && match[codes[i]]; });
        return words;
    }

    if (sorted_leaves(*col, {&leaf}, words, slots)) return words;

    if (col->type == ColumnType::Int64) {
        const vector<int64_t>& x = col->ints;
        if (op == Expr::Op::In) {
            // ints compare exactly; a float value matches an int that equals it as a double
            std::unordered_set<int64_t> ints;
            std::unordered_set<double> floats;
            for (const auto& v : leaf.values()) {
                if (auto* i = std::get_if<int64_t>(&v)) ints.insert(*i);
                else floats.insert(as_double(v, leaf.field()));
            }
            fill_bits(words, slots, [&](size_t i) {
                return valid[i] && (ints.count(x[i]) || (!floats.empty() && floats.count(static_cast<double>(x[i]))));
            });
        } else if (auto* value = std::get_if<int64_t>(&leaf.values()[0])) {
            compare_rows(words, slots, valid, x.data(), op, *value);
        } else {
            // against a float value, compare in double
            const double threshold = as_double(leaf.values()[0], leaf.field());
            vector<double> as_doubles(x.begin(), x.end());
            compare_rows(words, slots, valid, as_doubles.data(), op, threshold);
        }
        return words;
    }

    const vector<double>& x = col->floats;
    if (op == Expr::Op::In) {
        std::unordered_set<double> wanted;
        for (const auto& v : leaf.values()) wanted.insert(as_double(v, leaf.field()));
        fill_bits(words, slots, [&](size_t i) { return valid[i] && wanted.count(x[i]); });
    } else {
        compare_rows(words, slots, valid, x.data(), op, as_double(leaf.values()[0], leaf.field()));
    }
    return words;
}

vector<std::pair<std::string, ColumnType>> MetadataStore::schema() const {
    vector<std::pair<std::string, ColumnType>> out;
    for (const Column& col : columns_) out.emplace_back(col.name, col.type);
    return out;
}

namespace {

void write_string(std::ostream& out, const std::string& s) { write_vec(out, vector<char>(s.begin(), s.end())); }

std::string read_string(std::istream& in) {
    const vector<char> v = read_vec<char>(in, uint64_t{1} << 24);
    return {v.begin(), v.end()};
}

} // namespace

void MetadataStore::save(std::ostream& out) const {
    write_pod(out, static_cast<uint64_t>(columns_.size()));
    for (const Column& col : columns_) {
        write_string(out, col.name);
        write_pod(out, static_cast<uint8_t>(col.type));
        write_vec(out, col.valid);
        if (col.type == ColumnType::Int64) write_vec(out, col.ints);
        else if (col.type == ColumnType::Float64) write_vec(out, col.floats);
        else {
            write_vec(out, col.codes);
            write_pod(out, static_cast<uint64_t>(col.dict.size()));
            for (const std::string& s : col.dict) write_string(out, s);
        }
    }
}

MetadataStore MetadataStore::load(std::istream& in, uint64_t slots) {
    auto corrupt = [](const std::string& what) { return std::runtime_error("corrupt file: " + what); };

    MetadataStore store;
    const auto count = read_pod<uint64_t>(in);
    if (count > (uint64_t{1} << 16)) throw corrupt("too many metadata columns");
    std::unordered_set<std::string> names;
    for (uint64_t c = 0; c < count; ++c) {
        Column col;
        col.name = read_string(in);
        if (col.name.empty() || !names.insert(col.name).second) throw corrupt("invalid metadata column name");
        const auto type = read_pod<uint8_t>(in);
        if (type > static_cast<uint8_t>(ColumnType::String)) throw corrupt("invalid metadata column type");
        col.type = static_cast<ColumnType>(type);

        col.valid = read_vec<uint8_t>(in, slots);
        if (col.valid.size() != slots) throw corrupt("metadata column length mismatch");
        for (uint8_t v : col.valid)
            if (v > 1) throw corrupt("invalid metadata null flag");

        if (col.type == ColumnType::Int64) {
            col.ints = read_vec<int64_t>(in, slots);
            if (col.ints.size() != slots) throw corrupt("metadata column length mismatch");
        } else if (col.type == ColumnType::Float64) {
            col.floats = read_vec<double>(in, slots);
            if (col.floats.size() != slots) throw corrupt("metadata column length mismatch");
        }
        if (col.type != ColumnType::String) {
            col.sort_rows();
        } else {
            col.codes = read_vec<uint32_t>(in, slots);
            if (col.codes.size() != slots) throw corrupt("metadata column length mismatch");
            const auto dict_size = read_pod<uint64_t>(in);
            if (dict_size > UINT32_MAX) throw corrupt("metadata dictionary too large");
            for (uint64_t d = 0; d < dict_size; ++d) {
                std::string s = read_string(in);
                if (!col.code_of.emplace(s, static_cast<uint32_t>(col.dict.size())).second)
                    throw corrupt("duplicate metadata dictionary entry");
                col.dict.push_back(std::move(s));
            }
            for (uint64_t i = 0; i < slots; ++i)
                if (col.valid[i] && col.codes[i] >= col.dict.size()) throw corrupt("metadata string code out of range");
            col.reindex();
        }
        store.columns_.push_back(std::move(col));
    }
    return store;
}

} // namespace detail
} // namespace vecengine
