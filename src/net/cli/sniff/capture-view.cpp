#include "capture-view.h"
#include <ftxui/component/component.hpp>

CaptureView::CaptureView()
    : tab_entries_({
          "Forest",
          "Water",
          "I don't know",
      })
{
    body_ = ftxui::Radiobox(&tab_entries_, &tab_selected_);
    Add(body_);
}