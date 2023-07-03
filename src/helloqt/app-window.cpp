#include "app-window.h"
#include <QLabel>
#include <QPushButton>

AppWindow::AppWindow(QWidget* parent): QMainWindow(parent)
{
    resize(800, 600);
    label_ = new QLabel("YES", this);
    btn_ = new QPushButton("Toggle", this);
    connect(btn_, &QPushButton::clicked, [&]() {
        if (label_->text() == "YES") {
            label_->setText("NO");
        } else {
            label_->setText("YES");
        }
    });
    retranslateUi();
}

AppWindow::~AppWindow() = default;

void AppWindow::retranslateUi() { setWindowTitle(tr("helloqt")); }