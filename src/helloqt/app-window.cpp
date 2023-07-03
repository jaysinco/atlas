#include "app-window.h"
#include "utils/logging.h"
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>

AppWindow::AppWindow(QWidget* parent): QWidget(parent)
{
    resize(800, 600);
    label_ = new QLabel("YES", this);
    btn_ = new QPushButton("Toggle", this);
    QVBoxLayout* layout = new QVBoxLayout;
    layout->addWidget(label_);
    layout->addWidget(btn_);
    setLayout(layout);
    connect(btn_, &QPushButton::clicked, [&]() {
        ILOG("btn clicked");
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