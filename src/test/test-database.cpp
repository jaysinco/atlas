#include "toolkit/logging.h"
#include "toolkit/sqlite-helper.h"
#include <catch2/catch_all.hpp>

TEST_CASE("database")
{
    char const* sqls = R"(
BEGIN TRANSACTION;

CREATE TABLE IF NOT EXISTS "Users" (
	"account"	TEXT NOT NULL,
	"role"	INTEGER NOT NULL,
	"password"	TEXT NOT NULL,
	PRIMARY KEY("account"),
	FOREIGN KEY("role") REFERENCES "SVRoles"("id")
);

INSERT INTO "Users" VALUES ('<preset>',0,'');
INSERT INTO "Users" VALUES ('<anonym>',0,'');
INSERT INTO "Users" VALUES ('test',1,'123');
INSERT INTO "Users" VALUES ('admin',2,'admin');

COMMIT;
)";

    auto db_path = toolkit::currentExeDir() / "sql.db";
    auto db_guard = toolkit::scopeExit([&] {
        if (std::filesystem::exists(db_path)) {
            std::filesystem::remove(db_path);
        }
    });
    REQUIRE(toolkit::SQLiteHelper::execSQLs(db_path, sqls) == MyErrCode::kOk);
}
