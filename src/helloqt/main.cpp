#include "utils/args.h"
#include "app-window.h"
#include <QApplication>

int main(int argc, char** argv)
{
    MY_TRY;
    INIT_LOG(argc, argv);
    QApplication app(argc, argv);
    AppWindow w;
    w.show();
    return QApplication::exec();
    MY_CATCH;
}
