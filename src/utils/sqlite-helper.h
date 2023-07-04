#pragma once
#include <sqlite3.h>
#include <string>
#include <variant>
#include <filesystem>
#include <vector>

namespace utils
{

// CellType         C++               SQLite
// ===========================================
// String           std::string       TEXT
// Integer          int64_t           INTEGER
// Real             double            REAL
// Boolean          bool              BOOLEAN
// Date             int64_t           TIMESTAMP
// Null             NullType          NULL

struct NullType
{
};

extern NullType g_null;

enum class CellType
{
    kString,
    kInteger,
    kReal,
    kBoolean,
    kDate,
};

using CellValue = std::variant<std::string, int64_t, double, bool, NullType>;
using RowValue = std::vector<CellValue>;
using RowsValue = std::vector<RowValue>;

struct SQLUnit
{
    std::string sql;
    RowsValue rows;
};

class SQLiteHelper
{
public:
    static AtlasErr execSQLs(std::filesystem::path const& dbpath, std::string const& sqls);
    static AtlasErr execSQLs(std::filesystem::path const& dbpath, std::vector<SQLUnit> const& sqls);
    static AtlasErr querySQL(std::filesystem::path const& dbpath, SQLUnit const& sql,
                             std::vector<CellType> const& types, RowsValue& rows);

private:
    class Stmt
    {
    public:
        friend class SQLiteWrapper;
        Stmt() = default;
        Stmt(Stmt const&) = delete;
        ~Stmt();
        bool bind(std::vector<CellValue> const& vals);
        bool step(bool& done);
        bool stepUtilDone();
        bool column(int index, CellType const& type, CellValue& val);
        bool reset();

    private:
        sqlite3* db_ = nullptr;
        sqlite3_stmt* stmt_ = nullptr;
    };

    SQLiteWrapper() = default;
    SQLiteWrapper(SQLiteWrapper const&) = delete;
    ~SQLiteWrapper();

    SVErrCode open(std::string const& filePath);
    SVErrCode prepare(std::string const& sql, Stmt& stmt);
    SVErrCode exec(std::string const& sql);

    std::string filePath;
    sqlite3* db = nullptr;
};

}  // namespace utils
