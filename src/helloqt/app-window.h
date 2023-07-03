#pragma once
#include <QMainWindow>

class QLabel;
class QPushButton;

class AppWindow: public QMainWindow
{
    Q_OBJECT

public:
    explicit AppWindow(QWidget* parent = nullptr);
    ~AppWindow() override;

    void retranslateUi();

private:
    QLabel* label_;
    QPushButton* btn_;
};