#include <string>
#include <vector>
#include "ftxui/component/component.hpp"
#include "ftxui/component/screen_interactive.hpp"

int main()
{
    using namespace ftxui;

    std::vector<std::string> entries = {
        "tribute",  "clearance", "ally",        "bend",        "electronics", "module", "era",
        "cultural", "sniff",     "nationalism", "negotiation", "deliver",     "figure", "east",
        "tribute",  "clearance", "ally",        "bend",        "electronics", "module", "era",
        "cultural", "sniff",     "nationalism", "negotiation", "deliver",     "figure", "east",
        "tribute",  "clearance", "ally",        "bend",        "electronics", "module", "era",
        "cultural", "sniff",     "nationalism", "negotiation", "deliver",     "figure", "east",
    };

    int selected_1 = 0;
    int selected_2 = 0;
    int selected_3 = 0;
    int selected_4 = 0;

    auto layout = Container::Vertical({
        Container::Horizontal({
            Dropdown(&entries, &selected_1),
            Dropdown(&entries, &selected_2),
        }),
        Container::Horizontal({
            Dropdown(&entries, &selected_3),
            Dropdown(&entries, &selected_4),
        }),
    });

    auto screen = ScreenInteractive::FitComponent();
    screen.Loop(layout);
}