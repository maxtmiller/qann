#pragma once

// Per-vector metadata (issue #38): named columns of ints, floats or strings
// stored with an index, and filter expressions over them that a query can
// take instead of (or together with) a list of allowed ids.

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace vecengine {

using std::size_t;
using std::vector;

// A column's type, fixed when the column is created. Values are part of the
// file format.
enum class ColumnType : uint8_t {
    Int64 = 0,
    Float64 = 1,
    String = 2,  // stored dictionary-encoded: each distinct string once, rows hold codes
};

// One column of an add: a value for each of the batch's n rows.
struct MetadataColumn {
    std::string name;
    std::variant<vector<int64_t>, vector<double>, vector<std::string>> values;
};

// The metadata of one add: any number of columns, each with one value per row.
using Metadata = vector<MetadataColumn>;

// A filter over metadata: a comparison or test on one column, or a
// combination of filters. Immutable; copies share their subtrees.
//
// Rows whose value is missing (null) fail every comparison and isin, so
// ~(price < 50) includes them; use is_null to select or exclude them.
// Ints and floats compare as numbers (an int column against 2.5 compares in
// double); a string column takes only string values and compares them
// lexicographically. A column the index doesn't have, or a value of the
// wrong kind, throws std::invalid_argument when the filter is evaluated.
class Expr {
public:
    using Value = std::variant<int64_t, double, std::string>;
    enum class Op : uint8_t { Eq, Ne, Lt, Le, Gt, Ge, In, IsNull, And, Or, Not };

    // The deepest filter allowed, in levels of nesting. Evaluation recurses
    // per level, so deeper filters throw std::invalid_argument instead of
    // risking the stack. And / Or chains are flattened, so their length
    // doesn't count.
    static constexpr size_t kMaxDepth = 256;

    // field <op> value, for op one of Eq, Ne, Lt, Le, Gt, Ge.
    static Expr compare(std::string field, Op op, Value value);

    // field equals one of values.
    static Expr isin(std::string field, vector<Value> values);

    // field has no value.
    static Expr is_null(std::string field);

    // A SQL WHERE clause (issue #46), e.g. "category = 'shoes' AND price < 50":
    // comparisons (= != <> < <= > >=), [NOT] IN (...), IS [NOT] NULL,
    // [NOT] BETWEEN x AND y, combined with AND, OR, NOT and parentheses, with
    // SQL's precedence. Keywords are case-insensitive; strings are '...' ('' is
    // a quote) and names that clash with keywords are "...". NOT keeps SQL's
    // meaning: NOT (price < 50) excludes rows with no price, unlike ~.
    // Placeholders take values from params, bound after parsing so they can't
    // change the filter's structure: ? in order from the list, or :name from
    // the map. Throws std::invalid_argument naming the column of an error.
    static Expr parse(std::string_view sql, const vector<Value>& params = {});
    static Expr parse(std::string_view sql, const std::unordered_map<std::string, Value>& params);

    // All (op And) or any (op Or) of children, as one node: children that are
    // themselves the same op are merged in, so a long chain stays one level.
    static Expr join(Op op, vector<Expr> children);

    // Both, either, and the negation of filters.
    friend Expr operator&(const Expr& a, const Expr& b);
    friend Expr operator|(const Expr& a, const Expr& b);
    friend Expr operator~(const Expr& a);

    Op op() const noexcept { return node_->op; }
    const std::string& field() const noexcept { return node_->field; }        // leaves only
    const vector<Value>& values() const noexcept { return node_->values; }    // one for comparisons, any number for In
    const vector<Expr>& children() const noexcept { return node_->children; } // two or more for And / Or, one for Not

    // A readable form, e.g. (price < 50) & ~is_null(category).
    std::string to_string() const;

private:
    struct Node {
        Op op;
        std::string field;
        vector<Value> values;
        vector<Expr> children;
        size_t depth = 1;  // levels in this subtree, at most kMaxDepth
    };
    explicit Expr(std::shared_ptr<const Node> node) : node_(std::move(node)) {}

    std::shared_ptr<const Node> node_;
};

namespace detail {

// Metadata columns of one index, aligned to its slots: row s of every column
// belongs to slot s. Each index owns one (Index::metadata_). A column is
// created by the first add that mentions it, with earlier rows null; an add
// that leaves a column out fills its rows with null. remove() needs nothing
// here (deleted slots stay until compact(), and filters drop them).
class MetadataStore {
public:
    // Runs append()'s validation without changing anything, for a batch of n
    // rows: every column has n values, names are non-empty and unique, and a
    // column that exists keeps its type. Throws std::invalid_argument.
    void check(size_t n, const Metadata& md) const;

    // Appends n rows after the current `rows` (the index's slots before the
    // add). Requires check() to have passed. With n == 0 nothing changes, so
    // an empty add creates no columns.
    void append(size_t rows, size_t n, const Metadata& md);

    // Moves row old to remap[old] and drops rows marked IdMap::kRemoved, like
    // the index's own compaction.
    void compact(const vector<uint32_t>& remap);

    // One bit per slot (slots bits, 64 per word), set where the row matches
    // `where`. Deleted slots are not cleared here. Throws
    // std::invalid_argument on an unknown column or a value of the wrong kind.
    vector<uint64_t> evaluate(const Expr& where, size_t slots) const;

    // Each column's name and type, in creation order.
    vector<std::pair<std::string, ColumnType>> schema() const;

    // Whether there are no columns.
    bool empty() const noexcept { return columns_.empty(); }

    // Writes the columns (no header: it ends each index body, format 9+).
    void save(std::ostream& out) const;

    // Reads what save() wrote for an index with `slots` slots. Throws
    // std::runtime_error unless every column has exactly `slots` rows, names
    // are unique and non-empty, types and null flags are valid, and string
    // codes point into a dictionary without duplicates.
    static MetadataStore load(std::istream& in, uint64_t slots);

private:
    struct Column {
        std::string name;
        ColumnType type;
        vector<uint8_t> valid;    // 1 where the row has a value
        vector<int64_t> ints;     // Int64
        vector<double> floats;    // Float64
        vector<uint32_t> codes;   // String: index into dict
        vector<std::string> dict; // String: distinct values, in first-seen order
        std::unordered_map<std::string, uint32_t> code_of;  // String: dict lookup

        size_t rows() const noexcept { return valid.size(); }
    };

    // The column with this name, or nullptr.
    const Column* find(const std::string& name) const;

    // Bits of the rows of one leaf (comparison, In or IsNull).
    vector<uint64_t> evaluate_leaf(const Expr& leaf, size_t slots) const;

    vector<Column> columns_;
};

} // namespace detail
} // namespace vecengine
