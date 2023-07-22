#include "utils/args.h"
#include "main-window.h"
#include <QApplication>
#include <QFontDatabase>

int main(int argc, char** argv)
{
    MY_TRY;
    INIT_LOG(argc, argv);
    std::string ced = utils::currentExeDir().string();
    utils::setEnv("QT_QPA_PLATFORM_PLUGIN_PATH", ced.c_str());
    QApplication app(argc, argv);
    // int font_id = QFontDatabase::addApplicationFont(":/fonts/FangZhengHeiTiJianTi.ttf");
    // if (font_id != -1) {
    //     QFont font("FangZhengHeiTiJianTi");
    //     QApplication::setFont(font);
    // } else {
    //     ELOG("failed to load font!");
    // }
    MainWindow w;
    w.show();
    return QApplication::exec();
    MY_CATCH;
}
