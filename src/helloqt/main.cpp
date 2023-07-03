#include "utils/args.h"
#include "app-window.h"
#include <QApplication>

int main(int argc, char** argv)
{
    MY_TRY;
    INIT_LOG(argc, argv);
    utils::setenv("QT_QPA_PLATFORM_PLUGIN_PATH", utils::currentExeDir().string().c_str());
    QApplication app(argc, argv);
    AppWindow w;
    w.show();
    return QApplication::exec();
    MY_CATCH;
}
