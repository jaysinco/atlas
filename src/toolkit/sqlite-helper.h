#pragma once
#include "error.h"
#include "variant.h"
#include <filesystem>

class sqlite3;
class sqlite3_stmt;

namespace toolkit
{

using CellValue = Variant;
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
    static MyErrCode execSQLs(std::filesystem::path const& db_path, std::string const& sqls);
    static MyErrCode execSQLs(std::filesystem::path const& db_path,
                              std::vector<SQLUnit> const& sqls);
    static MyErrCode querySQL(std::filesystem::path const& db_path, SQLUnit const& sql,
                              RowsValue& rows);

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
        int columnCount();
        MyErrCode column(int index, CellValue& val);
        MyErrCode reset();

    private:
        int bindCell(CellValue const& val, int index);

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

}  // namespace toolkit
