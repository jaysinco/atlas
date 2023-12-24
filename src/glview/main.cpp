#include "top-window.h"
#include "toolkit/args.h"

int main(int argc, char** argv)
{
    INIT_LOG(argc, argv);
    FrameWindow::Init(argc, argv);
    FrameWindow::Run();
    FrameWindow::Destory();
    return 0;
}