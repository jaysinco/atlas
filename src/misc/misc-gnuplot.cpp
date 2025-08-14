#include "toolkit/args.h"
#include "toolkit/logging.h"
#include <gnuplot-iostream.h>
#include <thread>

MY_MAIN
{
    toolkit::Args args(argc, argv);
    args.parse();

    std::vector<std::tuple<double, double, double, double> > pts_a;
    std::vector<double> pts_b_x;
    std::vector<double> pts_b_y;
    std::vector<double> pts_b_dx;
    std::vector<double> pts_b_dy;

    for (double alpha = 0; alpha < 1; alpha += 1.0 / 24.0) {
        double theta = alpha * 2.0 * 3.14159;
        pts_a.emplace_back(cos(theta), sin(theta), -cos(theta) * 0.1, -sin(theta) * 0.1);
        pts_b_x.push_back(cos(theta) * 0.8);
        pts_b_y.push_back(sin(theta) * 0.8);
        pts_b_dx.push_back(sin(theta) * 0.1);
        pts_b_dy.push_back(-cos(theta) * 0.1);
    }

    Gnuplot gp("gnuplot");
    gp << "plot sin(x)" << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(3));
    gp << "set xrange [-2:2]\nset yrange [-2:2]\n";
    gp << "plot '-' with vectors title 'pts_A', '-' with vectors title 'pts_B'\n";
    gp.send1d(pts_a);
    gp.send1d(std::make_tuple(pts_b_x, pts_b_y, pts_b_dx, pts_b_dy));
    std::cin.get();

    return MyErrCode::kOk;
}
