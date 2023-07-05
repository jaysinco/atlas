#pragma once
#include <QWidget>

class QLabel;
class QPushButton;

class MainWindow: public QWidget
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

    void retranslateUi();

private:
    QLabel* label_;
    QPushButton* btn_;
};