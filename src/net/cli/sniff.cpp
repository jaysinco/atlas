#include "widgets/sniff-frame.h"
#include <wx/app.h>

class App : public wxApp
{
public:
    virtual bool OnInit() override;
    virtual int OnExit() override;
};

wxIMPLEMENT_APP(App);

bool App::OnInit()
{
    char **argv_ = argv;
    GUI_INIT_LOG(argc, argv_);
    SniffFrame *frame = new SniffFrame;
    frame->Show(true);
    return true;
}

int App::OnExit()
{
    VLOG(1) << "application about to exit.";
    return wxApp::OnExit();
}
