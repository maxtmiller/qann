#include "vecengine/metadata.hpp"
#include "vecengine/id_map.hpp"
#include "serialize.hpp"

#include <algorithm>
#include <cassert>
#include <cctype>
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

Expr operator&(const Expr& a, const Expr& b) {
    return Expr(std::make_shared<const Expr::Node>(Expr::Node{Expr::Op::And, {}, {}, {a, b}}));
}

Expr operator|(const Expr& a, const Expr& b) {
    return Expr(std::make_shared<const Expr::Node>(Expr::Node{Expr::Op::Or, {}, {}, {a, b}}));
}

Expr operator~(const Expr& a) {
    return Expr(std::make_shared<const Expr::Node>(Expr::Node{Expr::Op::Not, {}, {}, {a}}));
}

namespace {

// Recursive descent over a SQL WHERE clause (see Expr::parse). NOT is pushed
// down to the leaves as it parses (neg), so the tree has SQL's meaning
// without ~: negated comparisons flip their operator, negated IN / IS NULL /
// BETWEEN exclude nulls explicitly, and AND / OR swap by De Morgan.
class SqlParser {
public:
    SqlParser(std::string_view sql, const vector<Expr::Value>* list, const std::unordered_map<std::string, Expr::Value>* named)
        : sql_(sql), list_(list), named_(named) {}

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
        while (pos_ < sql_.size() && std::isspace(static_cast<unsigned char>(sql_[pos_]))) ++pos_;
    }

    static bool word_char(char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.'; }

    std::string peek_word() {
        skip_space();
        size_t end = pos_;
        while (end < sql_.size() && word_char(sql_[end])) ++end;
        if (end == pos_ && pos_ < sql_.size()) end = pos_ + 1;
        return std::string(sql_.substr(pos_, end - pos_));
    }

    // Consumes keyword kw (case-insensitive, whole word) if it comes next.
    bool keyword(std::string_view kw) {
        skip_space();
        if (sql_.size() - pos_ < kw.size()) return false;
        for (size_t i = 0; i < kw.size(); ++i)
            if (std::toupper(static_cast<unsigned char>(sql_[pos_ + i])) != kw[i]) return false;
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
        for (char& c : u) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        return u == "AND" || u == "OR" || u == "NOT" || u == "IN" || u == "IS" || u == "NULL" || u == "BETWEEN";
    }

    Expr combine(Expr a, Expr b, bool use_and) { return use_and ? (a & b) : (a | b); }

    Expr or_expr(bool neg) {
        Expr e = and_expr(neg);
        while (keyword("OR")) e = combine(e, and_expr(neg), neg);
        return e;
    }

    Expr and_expr(bool neg) {
        Expr e = not_expr(neg);
        while (keyword("AND")) e = combine(e, not_expr(neg), !neg);
        return e;
    }

    Expr not_expr(bool neg) {
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
        if (pos_ < sql_.size() && (std::isalpha(static_cast<unsigned char>(sql_[pos_])) || sql_[pos_] == '_'))
            while (pos_ < sql_.size() && word_char(sql_[pos_])) ++pos_;
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
        while (i < sql_.size() && std::isdigit(static_cast<unsigned char>(sql_[i]))) ++i;
        if (i < sql_.size() && sql_[i] == '.') {
            is_float = true;
            ++i;
            while (i < sql_.size() && std::isdigit(static_cast<unsigned char>(sql_[i]))) ++i;
        }
        if (i == digits || (i == digits + 1 && is_float)) fail("expected a value");
        if (i < sql_.size() && (sql_[i] == 'e' || sql_[i] == 'E')) {
            size_t j = i + 1;
            if (j < sql_.size() && (sql_[j] == '-' || sql_[j] == '+')) ++j;
            if (j < sql_.size() && std::isdigit(static_cast<unsigned char>(sql_[j]))) {
                is_float = true;
                i = j;
                while (i < sql_.size() && std::isdigit(static_cast<unsigned char>(sql_[i]))) ++i;
            }
        }
        if (i < sql_.size() && word_char(sql_[i])) fail("expected a value");
        const std::string text(sql_.substr(start, i - start));
        pos_ = i;
        if (is_float) return std::strtod(text.c_str(), nullptr);
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
            return Expr::compare(field, Expr::Op::Ge, std::move(lo)) & Expr::compare(field, Expr::Op::Le, std::move(hi));
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
        case Op::And: return "(" + children()[0].to_string() + " & " + children()[1].to_string() + ")";
        case Op::Or: return "(" + children()[0].to_string() + " | " + children()[1].to_string() + ")";
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

void MetadataStore::check(size_t n, const Metadata& md) const {
    std::unordered_set<std::string> names;
    for (const MetadataColumn& c : md) {
        if (c.name.empty()) throw std::invalid_argument("metadata column names must not be empty");
        if (!names.insert(c.name).second) throw std::invalid_argument("metadata column '" + c.name + "' appears twice");
        if (length_of(c) != n)
            throw std::invalid_argument("metadata column '" + c.name + "' has " + std::to_string(length_of(c)) +
                                        " values for " + std::to_string(n) + " vectors");
        if (const Column* existing = find(c.name); existing && existing->type != type_of(c))
            throw std::invalid_argument("metadata column '" + c.name + "' holds " + type_name(existing->type) +
                                        " values; got " + type_name(type_of(c)));
    }
}

void MetadataStore::append(size_t rows, size_t n, const Metadata& md) {
    std::unordered_map<std::string, const MetadataColumn*> given;
    for (const MetadataColumn& c : md) given.emplace(c.name, &c);

    for (const MetadataColumn& c : md) {
        // an add of no vectors creates no columns: its empty values say nothing about the type
        if (n == 0 || find(c.name)) continue;
        Column col{c.name, type_of(c), vector<uint8_t>(rows, 0), {}, {}, {}, {}, {}};
        if (col.type == ColumnType::Int64) col.ints.assign(rows, 0);
        else if (col.type == ColumnType::Float64) col.floats.assign(rows, 0.0);
        else col.codes.assign(rows, 0);
        columns_.push_back(std::move(col));
    }

    for (Column& col : columns_) {
        assert(col.rows() == rows);
        auto it = given.find(col.name);
        if (it == given.end() || n == 0) {  // left out of this add: null
            col.valid.insert(col.valid.end(), n, 0);
            if (col.type == ColumnType::Int64) col.ints.insert(col.ints.end(), n, 0);
            else if (col.type == ColumnType::Float64) col.floats.insert(col.floats.end(), n, 0.0);
            else col.codes.insert(col.codes.end(), n, 0);
            continue;
        }
        col.valid.insert(col.valid.end(), n, 1);
        if (col.type == ColumnType::Int64) {
            const auto& v = std::get<vector<int64_t>>(it->second->values);
            col.ints.insert(col.ints.end(), v.begin(), v.end());
        } else if (col.type == ColumnType::Float64) {
            const auto& v = std::get<vector<double>>(it->second->values);
            col.floats.insert(col.floats.end(), v.begin(), v.end());
        } else {
            for (const std::string& s : std::get<vector<std::string>>(it->second->values)) {
                auto [pos, added] = col.code_of.emplace(s, static_cast<uint32_t>(col.dict.size()));
                if (added) col.dict.push_back(s);
                col.codes.push_back(pos->second);
            }
        }
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
            const vector<uint64_t> b = evaluate(where.children()[1], slots);
            for (size_t w = 0; w < a.size(); ++w) a[w] = where.op() == Expr::Op::And ? a[w] & b[w] : a[w] | b[w];
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
        }
        store.columns_.push_back(std::move(col));
    }
    return store;
}

} // namespace detail
} // namespace vecengine
