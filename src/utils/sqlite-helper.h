#pragma once
#include "error.h"
#include <string>
#include <variant>
#include <filesystem>
#include <vector>

class sqlite3;
class sqlite3_stmt;

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

extern NullType g_null_v;

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
    SQLiteHelper(SQLiteHelper const&) = delete;
    static MyErrCode execSQLs(std::filesystem::path const& db_path, std::string const& sqls);
    static MyErrCode execSQLs(std::filesystem::path const& db_path,
                              std::vector<SQLUnit> const& sqls);
    static MyErrCode querySQL(std::filesystem::path const& db_path, SQLUnit const& sql,
                              std::vector<CellType> const& types, RowsValue& rows);

private:
    class Stmt
    {
    public:
        friend class SQLiteHelper;
        Stmt() = default;
        Stmt(Stmt const&) = delete;
        ~Stmt();
        MyErrCode bind(std::vector<CellValue> const& vals);
        MyErrCode step(bool& done);
        MyErrCode stepUtilDone();
        MyErrCode column(int index, CellType const& type, CellValue& val);
        MyErrCode reset();

    private:
        sqlite3* db_ = nullptr;
        sqlite3_stmt* stmt_ = nullptr;
    };

    SQLiteHelper() = default;
    ~SQLiteHelper();

    MyErrCode open(std::string const& db_path);
    MyErrCode prepare(std::string const& sql, Stmt& stmt);
    MyErrCode exec(std::string const& sql);

    std::string db_path_;
    sqlite3* db_ = nullptr;
};

}  // namespace utils
