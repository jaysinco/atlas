#include "sqlite-helper.h"
#include "logging.h"
#include <boost/algorithm/string.hpp>
#include <sqlite3.h>
#include <sstream>

namespace utils
{

NullType g_null_v;

struct CellValueBindVisitor
{
    int operator()(std::string const& v) const
    {
        return sqlite3_bind_text(stmt, index, v.c_str(), v.size(), SQLITE_TRANSIENT);
    }

    int operator()(int64_t v) const { return sqlite3_bind_int64(stmt, index, v); }

    int operator()(double v) const { return sqlite3_bind_double(stmt, index, v); }

    int operator()(bool v) const { return sqlite3_bind_int(stmt, index, v); }

    int operator()(NullType const& v) const { return sqlite3_bind_null(stmt, index); }

    sqlite3_stmt* stmt;
    int index;
};

struct CellValueToSQLVisitor
{
    std::string operator()(std::string const& v) const
    {
        std::ostringstream ss;
        std::string copy(v);
        boost::replace_all(copy, "'", "''");
        ss << "'" << copy << "'";
        return ss.str();
    }

    std::string operator()(int64_t v) const { return std::to_string(v); }

    std::string operator()(double v) const { return std::to_string(v); }

    std::string operator()(bool v) const { return v ? "TRUE" : "FALSE"; }

    std::string operator()(NullType const& v) const { return "NULL"; }
};

SQLiteHelper::Stmt::~Stmt()
{
    if (this->stmt_ != nullptr) {
        DLOG("[{}] finalize sql", sqlite3_finalize(this->stmt_));
    }
}

MyErrCode SQLiteHelper::Stmt::bind(std::vector<CellValue> const& vals)
{
    for (int i = 0; i < vals.size(); ++i) {
        int code = std::visit(CellValueBindVisitor{this->stmt_, i + 1}, vals.at(i));
        DLOG("[{}] bind value {} -> {}", code, i + 1,
             std::visit(CellValueToSQLVisitor{}, vals.at(i)));
        if (code != SQLITE_OK) {
            ELOG(sqlite3_errmsg(this->db_));
            return MyErrCode::kFailed;
        }
    }
    return MyErrCode::kOk;
}

MyErrCode SQLiteHelper::Stmt::step(bool& done)
{
    done = false;
    int code = sqlite3_step(this->stmt_);
    DLOG("[{}] step sql", code);
    if (code == SQLITE_OK || code == SQLITE_ROW) {
        return MyErrCode::kOk;
    } else if (code == SQLITE_DONE) {
        done = true;
        return MyErrCode::kOk;
    } else {
        ELOG(sqlite3_errmsg(this->db_));
        return MyErrCode::kFailed;
    }
}

MyErrCode SQLiteHelper::Stmt::stepUtilDone()
{
    bool done = false;
    do {
        CHECK_ERR_RET(this->step(done));
    } while (!done);
    return MyErrCode::kOk;
}

MyErrCode SQLiteHelper::Stmt::column(int index, CellType const& type, CellValue& val)
{
    if (index >= sqlite3_column_count(this->stmt_)) {
        return MyErrCode::kOutOfRange;
    }
    if (sqlite3_column_type(this->stmt_, index) == SQLITE_NULL) {
        val = g_null_v;
    } else if (type == CellType::kString) {
        unsigned char const* begin = sqlite3_column_text(this->stmt_, index);
        int size = sqlite3_column_bytes(this->stmt_, index);
        val = std::string(begin, begin + size);
    } else if (type == CellType::kInteger || type == CellType::kDate) {
        val = static_cast<int64_t>(sqlite3_column_int64(this->stmt_, index));
    } else if (type == CellType::kReal) {
        val = sqlite3_column_double(this->stmt_, index);
    } else if (type == CellType::kBoolean) {
        val = static_cast<bool>(sqlite3_column_int(this->stmt_, index));
    } else {
        ELOG("unknow cell type: {}", static_cast<int>(type));
        return MyErrCode::kUnknown;
    }
    DLOG("[{}] get column {} -> {}", 0, index, std::visit(CellValueToSQLVisitor{}, val));
    return MyErrCode::kOk;
}

MyErrCode SQLiteHelper::Stmt::reset()
{
    int code = sqlite3_reset(this->stmt_);
    DLOG("[{}] reset sql", code);
    if (code != SQLITE_OK) {
        ELOG(sqlite3_errmsg(this->db_));
        return MyErrCode::kFailed;
    }
    return MyErrCode::kOk;
}

SQLiteHelper::~SQLiteHelper()
{
    if (this->db_ != nullptr) {
        DLOG("[{}] close database {}", sqlite3_close_v2(this->db_), this->db_path_);
    }
}

MyErrCode SQLiteHelper::open(std::string const& db_path)
{
    this->db_path_ = db_path;
    int code =
        sqlite3_open_v2(db_path.c_str(), &this->db_,
                        SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_NOMUTEX, nullptr);
    DLOG("[{}] open database {}", code, this->db_path_);
    if (code != SQLITE_OK) {
        ELOG(sqlite3_errmsg(this->db_));
        return MyErrCode::kFailed;
    }
    return MyErrCode::kOk;
}

MyErrCode SQLiteHelper::prepare(std::string const& sql, Stmt& stmt)
{
    stmt.db_ = this->db_;
    int code = sqlite3_prepare_v2(this->db_, sql.c_str(), sql.size(), &stmt.stmt_, nullptr);
    DLOG("[{}] prepare sql =>\n{}", code, sql);
    if (code != SQLITE_OK) {
        ELOG(sqlite3_errmsg(this->db_));
        return MyErrCode::kFailed;
    }
    return MyErrCode::kOk;
}

MyErrCode SQLiteHelper::exec(std::string const& sql)
{
    char* errmsg = nullptr;
    int code = sqlite3_exec(this->db_, sql.c_str(), nullptr, nullptr, &errmsg);
    DLOG("[{}] exec sql =>\n{}", code, sql);
    if (errmsg != nullptr) {
        std::string message = errmsg;
        sqlite3_free(errmsg);
        ELOG(sqlite3_errmsg(this->db_));
        return MyErrCode::kFailed;
    }
    return MyErrCode::kOk;
}

MyErrCode SQLiteHelper::execSQLs(std::filesystem::path const& db_path, std::string const& sqls)
{
    SQLiteHelper db;
    CHECK_ERR_RET(db.open(db_path.string()));
    CHECK_ERR_RET(db.exec(sqls));
    return MyErrCode::kOk;
}

MyErrCode SQLiteHelper::execSQLs(std::filesystem::path const& db_path,
                                 std::vector<SQLUnit> const& sqls)
{
    SQLiteHelper db;
    CHECK_ERR_RET(db.open(db_path.string()));
    CHECK_ERR_RET(db.exec("BEGIN TRANSACTION;"));
    for (auto const& sql: sqls) {
        SQLiteHelper::Stmt stmt;
        CHECK_ERR_RET(db.prepare(sql.sql, stmt));
        for (auto const& row: sql.rows) {
            CHECK_ERR_RET(stmt.reset());
            CHECK_ERR_RET(stmt.bind(row));
            CHECK_ERR_RET(stmt.stepUtilDone());
        }
    }
    CHECK_ERR_RET(db.exec("END TRANSACTION;"));
    return MyErrCode::kOk;
}

MyErrCode SQLiteHelper::querySQL(std::filesystem::path const& db_path, SQLUnit const& sql,
                                 std::vector<CellType> const& types, RowsValue& rows)
{
    SQLiteHelper db;
    CHECK_ERR_RET(db.open(db_path.string()));
    SQLiteHelper::Stmt stmt;
    CHECK_ERR_RET(db.prepare(sql.sql, stmt));
    if (sql.rows.size() > 0) {
        CHECK_ERR_RET(stmt.bind(sql.rows.at(0)));
    }
    bool done = false;
    while (true) {
        CHECK_ERR_RET(stmt.step(done));
        if (done) {
            break;
        }
        RowValue row;
        for (int i = 0; i < types.size(); ++i) {
            CellValue val;
            CHECK_ERR_RET(stmt.column(i, types.at(i), val));
            row.push_back(std::move(val));
        }
        rows.push_back(std::move(row));
    };
    return MyErrCode::kOk;
}

}  // namespace utils