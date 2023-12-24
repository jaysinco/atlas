#include "sqlite-helper.h"
#include "logging.h"
#include "toolkit/error.h"
#include <boost/algorithm/string.hpp>
#include <sqlite3.h>
#include <sstream>

namespace toolkit
{

using fmt::enums::format_as;

static std::string cellValueToSQL(CellValue const& val)
{
    switch (val.getType()) {
        case Variant::kVoid:
            return "NULL";
        case Variant::kBool:
            return val.asBool() ? "TRUE" : "FALSE";
        case Variant::kInt:
        case Variant::kUint:
            return std::to_string(val.asInt());
        case Variant::kDouble:
            return std::to_string(val.asDouble());
        case Variant::kStr: {
            std::ostringstream ss;
            std::string copy(val.asStr());
            boost::replace_all(copy, "'", "''");
            ss << "'" << copy << "'";
            return ss.str();
        }
        case Variant::kVec:
        case Variant::kMap: {
            return cellValueToSQL(val.toJsonStr());
        }
        default:
            throw std::runtime_error(fmt::format("variant bad type: {}", val.getType()));
    }
}

SQLiteHelper::Stmt::~Stmt()
{
    if (this->stmt_ != nullptr) {
        DLOG("[{}] finalize sql", sqlite3_finalize(this->stmt_));
    }
}

int SQLiteHelper::Stmt::bindCell(CellValue const& val, int index)
{
    switch (val.getType()) {
        case Variant::kVoid:
            return sqlite3_bind_null(stmt_, index);
        case Variant::kBool:
            return sqlite3_bind_int(stmt_, index, val.asBool());
        case Variant::kInt:
        case Variant::kUint:
            return sqlite3_bind_int64(stmt_, index, val.asInt());
        case Variant::kDouble:
            return sqlite3_bind_double(stmt_, index, val.asDouble());
        case Variant::kStr: {
            auto& str = val.asStr();
            return sqlite3_bind_text(stmt_, index, str.c_str(), str.size(), SQLITE_TRANSIENT);
        }
        case Variant::kVec:
        case Variant::kMap: {
            auto str = val.toJsonStr();
            return sqlite3_bind_text(stmt_, index, str.c_str(), str.size(), SQLITE_TRANSIENT);
        }
        default:
            throw std::runtime_error(fmt::format("variant bad type: {}", val.getType()));
    }
}

MyErrCode SQLiteHelper::Stmt::bind(std::vector<CellValue> const& vals)
{
    for (int i = 0; i < vals.size(); ++i) {
        int code = bindCell(vals.at(i), i + 1);
        DLOG("[{}] bind value {} -> {}", code, i + 1, cellValueToSQL(vals.at(i)));
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

int SQLiteHelper::Stmt::columnCount() { return sqlite3_column_count(this->stmt_); }

MyErrCode SQLiteHelper::Stmt::column(int index, CellValue& val)
{
    int type = sqlite3_column_type(this->stmt_, index);
    switch (type) {
        case SQLITE_NULL:
            val = {};
            break;
        case SQLITE_INTEGER:
            val = static_cast<int64_t>(sqlite3_column_int64(this->stmt_, index));
            break;
        case SQLITE_FLOAT:
            val = sqlite3_column_double(this->stmt_, index);
            break;
        case SQLITE_TEXT: {
            auto begin = reinterpret_cast<char const*>(sqlite3_column_text(this->stmt_, index));
            int size = sqlite3_column_bytes(this->stmt_, index);
            val = std::string(begin, begin + size);
            break;
        }
        case SQLITE_BLOB:
            return MyErrCode::kUnimplemented;
        default:
            throw std::runtime_error(fmt::format("sqlite3 bad column type: {}", type));
    }
    DLOG("[{}] get column {} -> {}", 0, index, cellValueToSQL(val));
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
                                 RowsValue& rows)
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
        for (int i = 0; i < stmt.columnCount(); ++i) {
            CellValue val;
            CHECK_ERR_RET(stmt.column(i, val));
            row.push_back(std::move(val));
        }
        rows.push_back(std::move(row));
    };
    return MyErrCode::kOk;
}

}  // namespace toolkit