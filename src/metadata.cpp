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
// benchmarks/string_filter_sweep.cpp (1M rows): about 0.41 ns per scanned row,
// 1.7-4.5x that per listed row (more as lists grow), and an OR'd bitmap word
// costs less than a scanned row, so it counts as one.
constexpr size_t kScatterCost = 4;

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

vector<uint64_t> MetadataStore::evaluate(const Expr& where, size_t slots) const {
    switch (where.op()) {
        case Expr::Op::And:
        case Expr::Op::Or: {
            vector<uint64_t> a = evaluate(where.children()[0], slots);
            for (size_t c = 1; c < where.children().size(); ++c) {
                const vector<uint64_t> b = evaluate(where.children()[c], slots);
                if (where.op() == Expr::Op::And)
                    for (size_t w = 0; w < a.size(); ++w) a[w] &= b[w];
                else
                    for (size_t w = 0; w < a.size(); ++w) a[w] |= b[w];
            }
            return a;
        }
        case Expr::Op::Not: {
            vector<uint64_t> a = evaluate(where.children()[0], slots);
            for (uint64_t& w : a) w = ~w;
            if (slots % 64) a.back() &= (uint64_t{1} << (slots % 64)) - 1;  // padding bits past the last slot
            return a;
        }
        default:
            return evaluate_leaf(where, slots);
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
        // decide each distinct string once, then look rows up by code
        vector<uint8_t> match(col->dict.size(), 0);
        if (op == Expr::Op::In) {
            std::unordered_set<std::string> wanted;
            for (const auto& v : leaf.values()) wanted.insert(as_string(v, leaf.field()));
            for (size_t c = 0; c < col->dict.size(); ++c) match[c] = wanted.count(col->dict[c]);
        } else {
            const std::string& value = as_string(leaf.values()[0], leaf.field());
            for (size_t c = 0; c < col->dict.size(); ++c) match[c] = compare(op, col->dict[c], value);
        }
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
