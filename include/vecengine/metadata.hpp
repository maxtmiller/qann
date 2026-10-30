#pragma once

// Per-vector metadata (issue #38): named columns of ints, floats or strings
// stored with an index, and filter expressions over them that a query can
// take instead of (or together with) a list of allowed ids.

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <optional>
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

// One column of an add or update: a value for each of the batch's n rows.
// valid is empty when every row has a value, or n flags with 0 where the row
// is null (its entry in values is then ignored).
struct MetadataColumn {
    using Values = std::variant<vector<int64_t>, vector<double>, vector<std::string>>;

    MetadataColumn(std::string name, Values values, vector<uint8_t> valid = {})
        : name(std::move(name)), values(std::move(values)), valid(std::move(valid)) {}

    std::string name;
    Values values;
    vector<uint8_t> valid;
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

    // lo <= field <= hi, as (field >= lo) & (field <= hi); SQL's BETWEEN.
    static Expr between(std::string field, Value lo, Value hi);

    // A SQL WHERE clause (issue #46), e.g. "category = 'shoes' AND price < 50":
    // comparisons (= != <> < <= > >=), [NOT] IN (...), IS [NOT] NULL,
    // [NOT] BETWEEN x AND y, combined with AND, OR, NOT and parentheses, with
    // SQL's precedence. Keywords are case-insensitive; strings are '...' ('' is
    // a quote) and names that clash with keywords or use non-ASCII characters
    // are "...". NOT keeps SQL's meaning: NOT (price < 50) excludes rows with
    // no price, unlike ~. Numbers that overflow (1e400) are an error.
    // Placeholders take values from params, bound after parsing so they can't
    // change the filter's structure: ? in order from the list, or :name from
    // the map. Throws std::invalid_argument naming the column of an error.
    static Expr parse(std::string_view sql, const vector<Value>& params = {});
    static Expr parse(std::string_view sql, const std::unordered_map<std::string, Value>& params);

    // The most conditions (comparisons, IN, IS NULL, BETWEEN; an IN list
    // counts once) parse() accepts, so a filter string from user input can't
    // make every query scan its columns thousands of times. Process-wide;
    // filters built with compare / isin / & / | aren't limited. n must be >= 1.
    static constexpr size_t kDefaultMaxConditions = 1024;
    static size_t max_conditions() noexcept;
    static void set_max_conditions(size_t n);

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
// update() overwrites rows in place.
class MetadataStore {
public:
    // Runs append()'s validation without changing anything, for a batch of n
    // rows: every column has n values, names are non-empty and unique, and a
    // column that exists keeps its type. Throws std::invalid_argument.
    void check(size_t n, const Metadata& md) const;

    // update()'s validation without changing anything: check()'s rules, and
    // every column must already exist (update never creates one).
    void check_update(size_t n, const Metadata& md) const;

    // Writes md's rows into the given slots (one per row, distinct), leaving
    // other columns and rows unchanged. Requires check_update() to have
    // passed. With no slots nothing changes.
    void update(const vector<uint32_t>& slots, const Metadata& md);

    // check()'s rules for c as a column the store doesn't have yet, of n
    // values that must fit `type`. Throws std::invalid_argument.
    void check_new(const MetadataColumn& c, ColumnType type, size_t n) const;

    // A new column of this type for all `rows` slots, each set to value, or
    // null without one. Throws std::invalid_argument, changing nothing, on an
    // empty or taken name or a value that doesn't fit the type.
    void add_column(std::string name, ColumnType type, size_t rows, const std::optional<Expr::Value>& value = std::nullopt);

    // Removes the column and its values. Throws std::invalid_argument if the
    // store has no such column.
    void drop_column(const std::string& name);

    // Appends n rows after the current `rows` (the index's slots before the
    // add). Requires check() to have passed. With n == 0 nothing changes, so
    // an empty add creates no columns.
    void append(size_t rows, size_t n, const Metadata& md);

    // Moves row old to remap[old] and drops rows marked IdMap::kRemoved, like
    // the index's own compaction. Strings no remaining row uses are freed.
    void compact(const vector<uint32_t>& remap);

    // Frees strings no row uses (left behind by update()), for a compaction
    // that removes no rows.
    void drop_unused_strings();

    // One bit per slot (slots bits, 64 per word), set where the row matches
    // `where`. Deleted slots are not cleared here. Throws
    // std::invalid_argument on an unknown column or a value of the wrong kind.
    // Conditions shared by Or branches are factored out first (see factor()).
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

        // String: the rows holding one code, so a filter on a few strings
        // costs about their rows instead of a pass over the column (#72).
        // Null rows are in none. Kept in memory only; load() rebuilds it.
        struct CodeRows {
            vector<uint32_t> rows;  // the rows in no order, while the code is rare
            vector<uint64_t> bits;  // a bit per row once it is common (rows is then empty); may end before the last row
            size_t count = 0;
        };
        vector<CodeRows> rows_of;  // String: one per dict entry
        size_t balanced_rows = 0;  // String: rows() when every code was last rebalanced

        // Int64 / Float64: the non-null rows below sorted_rows (but no NaN),
        // ordered by value, so a range filter costs about its matching rows
        // (#72). Later rows and overwritten ones are tested one by one until
        // they reach rows() / 8, then everything is sorted again. Memory only.
        vector<uint32_t> sorted;
        vector<int64_t> sorted_ints;    // Int64: sorted's values when sorted (a stale row's may differ now)
        vector<double> sorted_floats;   // Float64: the same
        size_t sorted_rows = 0;  // rows [sorted_rows, rows()) are not in sorted yet
        vector<uint32_t> stale;  // rows below sorted_rows overwritten since the sort; may repeat

        size_t rows() const noexcept { return valid.size(); }

        // Adds rows [first, rows()) to rows_of, then rebalances. String columns only.
        void index_rows(size_t first);

        // Adds row s under its code, if it isn't null; rows_of must cover dict.
        void index_row(size_t s);

        // Rebuilds rows_of from scratch, for after codes are renumbered or rows move.
        void reindex();

        // Removes each valid row in `rows` from rows_of under its code before
        // an overwrite; old_codes[i] is rows[i]'s code. Leaves the switch to rebalance().
        void unindex(const vector<uint32_t>& rows, const vector<uint32_t>& old_codes);

        // balance() on the touched codes, or on every code once rows() has
        // grown by a quarter since the last time (also the first time).
        void rebalance(const vector<uint32_t>& touched);

        // Switches r to a bitmap when it holds at least 1/32 of the rows (and
        // 64 rows), and back to a list below 1/64.
        void balance(CodeRows& r) const;

        // Sorts every row into sorted and empties stale. Int64 / Float64 only.
        void sort_rows();

        // sort_rows() once the unsorted rows (after sorted_rows, or stale)
        // exceed rows() / 8. Int64 / Float64 only.
        void maybe_sort();
    };

    // The column with this name, or nullptr.
    const Column* find(const std::string& name) const;

    // Writes row i of c into row `to` of col, which has c's type.
    static void write_row(Column& col, size_t to, const MetadataColumn& c, size_t i);

    // Bits of the rows of one leaf (comparison, In or IsNull).
    vector<uint64_t> evaluate_leaf(const Expr& leaf, size_t slots) const;

    // What evaluating part of an And would match and cost, in rows a pass
    // tests in the same time (a pass over the column costs slots).
    struct Estimate {
        size_t matches;
        size_t cost;
    };

    // Sets the bits of the And of leaves (comparisons or In on numeric column
    // col; more than one leaf only for comparisons) in words, all zero, from
    // col.sorted when that costs less than a pass over the column, and
    // returns whether it did. With estimate, only fills it in (a pass's
    // figures when sorted rows can't help) and returns false. Throws like
    // evaluate_leaf on a string value.
    static bool sorted_leaves(const Column& col, const vector<const Expr*>& leaves, vector<uint64_t>& words, size_t slots,
                              Estimate* estimate = nullptr);

    // For a string column: 1 for each dictionary code that leaf (a
    // comparison or In) matches. Throws on a value that isn't a string.
    static vector<uint8_t> string_match(const Column& col, const Expr& leaf);

    // The Estimate for the And of leaves on col (several only for numeric
    // comparisons). Throws like evaluate_leaf on a value of the wrong kind.
    Estimate estimate(const Column& col, const vector<const Expr*>& leaves, size_t slots) const;

    // Clears the bits of `words` whose rows fail leaf, testing only the rows
    // set, for an And whose other parts left few. Throws like evaluate_leaf.
    void narrow(const Expr& leaf, vector<uint64_t>& words) const;

    // evaluate() without factoring, for the rows set in `within` only: the
    // result is where & within (where alone without within). Leaves test
    // just those rows when that is cheaper than evaluating over the column.
    vector<uint64_t> evaluate_tree(const Expr& where, size_t slots, const vector<uint64_t>* within) const;

    // The Estimate for any filter: a leaf's from estimate(), an And's fewest
    // matches, an Or's summed matches, and their summed costs. Throws like
    // evaluate on an unknown column or a value of the wrong kind.
    Estimate estimate_tree(const Expr& where, size_t slots) const;

    vector<Column> columns_;
};

} // namespace detail
} // namespace vecengine
