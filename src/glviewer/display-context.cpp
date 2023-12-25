#include "display-context.h"

#define WINDOW_WIDTH 1536
#define WINDOW_HEIGHT 864

DisplayContext& DisplayContext::Instance()
{
    static DisplayContext ctx{
        .wl =
            {
                .opaque = 1,
                .running = 1,
            },

        .window_size =
            {
                .width = WINDOW_WIDTH,
                .height = WINDOW_HEIGHT,
            },

        .ime =
            {
                .ascii_mode = true,
            },
    };

    return ctx;
}
