#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <gbm.h>
#define GL_GLEXT_PROTOTYPES 1
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <assert.h>

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

static struct
{
    EGLDisplay display;
    EGLConfig config;
    EGLContext context;
    EGLSurface surface;
    GLuint program;
    GLint modelviewmatrix, modelviewprojectionmatrix, normalmatrix;
    GLuint vbo;
    GLuint positionsoffset, colorsoffset, normalsoffset;
} gl;

static struct
{
    struct gbm_device* dev;
    struct gbm_surface* surface;
} gbm;

static struct
{
    int fd;
    drmModeModeInfo* mode;
    uint32_t crtc_id;
    uint32_t connector_id;
} drm;

struct DrmFb
{
    struct gbm_bo* bo;
    uint32_t fb_id;
};

static uint32_t findCrtcForEncoder(drmModeRes const* resources, drmModeEncoder const* encoder)
{
    int i;

    for (i = 0; i < resources->count_crtcs; i++) {
        /* possible_crtcs is a bitmask as described here:
         * https://dvdhrm.wordpress.com/2012/09/13/linux-drm-mode-setting-api
         */
        const uint32_t crtc_mask = 1 << i;
        const uint32_t crtc_id = resources->crtcs[i];
        if (encoder->possible_crtcs & crtc_mask) {
            return crtc_id;
        }
    }

    /* no match found */
    return -1;
}

static uint32_t findCrtcForConnector(drmModeRes const* resources, drmModeConnector const* connector)
{
    int i;

    for (i = 0; i < connector->count_encoders; i++) {
        const uint32_t encoder_id = connector->encoders[i];
        drmModeEncoder* encoder = drmModeGetEncoder(drm.fd, encoder_id);

        if (encoder) {
            const uint32_t crtc_id = findCrtcForEncoder(resources, encoder);

            drmModeFreeEncoder(encoder);
            if (crtc_id != 0) {
                return crtc_id;
            }
        }
    }

    /* no match found */
    return -1;
}

static int initDrm(int disp_idx)
{
    drmModeRes* resources;
    drmModeConnector* connector = nullptr;
    drmModeEncoder* encoder = nullptr;
    int i, area;

    drm.fd = open("/dev/dri/card0", O_RDWR);

    if (drm.fd < 0) {
        printf("could not open drm device\n");
        return -1;
    }

    resources = drmModeGetResources(drm.fd);
    if (!resources) {
        printf("drmModeGetResources failed: %s\n", strerror(errno));
        return -1;
    }

    /* find a connected connector: */
    int connected_idx = -1;
    for (i = 0; i < resources->count_connectors; i++) {
        connector = drmModeGetConnector(drm.fd, resources->connectors[i]);
        if (connector->connection == DRM_MODE_CONNECTED) {
            ++connected_idx;
            if (connected_idx == disp_idx) {
                break;
            }
        }
        drmModeFreeConnector(connector);
        connector = nullptr;
    }

    if (!connector) {
        /* we could be fancy and listen for hotplug events and wait for
         * a connector..
         */
        printf("no connected connector!\n");
        return -1;
    }

    /* find prefered mode or the highest resolution mode: */
    for (i = 0, area = 0; i < connector->count_modes; i++) {
        drmModeModeInfo* current_mode = &connector->modes[i];

        if (current_mode->type & DRM_MODE_TYPE_PREFERRED) {
            drm.mode = current_mode;
        }

        int current_area = current_mode->hdisplay * current_mode->vdisplay;
        if (current_area > area) {
            drm.mode = current_mode;
            area = current_area;
        }
    }

    if (!drm.mode) {
        printf("could not find mode!\n");
        return -1;
    }

    /* find encoder: */
    for (i = 0; i < resources->count_encoders; i++) {
        encoder = drmModeGetEncoder(drm.fd, resources->encoders[i]);
        if (encoder->encoder_id == connector->encoder_id) {
            break;
        }
        drmModeFreeEncoder(encoder);
        encoder = nullptr;
    }

    if (encoder) {
        drm.crtc_id = encoder->crtc_id;
    } else {
        uint32_t crtc_id = findCrtcForConnector(resources, connector);
        if (crtc_id == 0) {
            printf("no crtc found!\n");
            return -1;
        }

        drm.crtc_id = crtc_id;
    }

    drm.connector_id = connector->connector_id;

    return 0;
}

static int initGbm()
{
    gbm.dev = gbm_create_device(drm.fd);

    gbm.surface =
        gbm_surface_create(gbm.dev, drm.mode->hdisplay, drm.mode->vdisplay, GBM_FORMAT_XRGB8888,
                           GBM_BO_USE_SCANOUT | GBM_BO_USE_RENDERING);
    if (!gbm.surface) {
        printf("failed to create gbm surface\n");
        return -1;
    }

    return 0;
}

static int initGL()
{
    EGLint major, minor, n;
    GLuint vertex_shader, fragment_shader;
    GLint ret;

    static EGLint context_attribs[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};

    static EGLint config_attribs[] = {EGL_SURFACE_TYPE,
                                      EGL_WINDOW_BIT,
                                      EGL_RED_SIZE,
                                      1,
                                      EGL_GREEN_SIZE,
                                      1,
                                      EGL_BLUE_SIZE,
                                      1,
                                      EGL_ALPHA_SIZE,
                                      0,
                                      EGL_RENDERABLE_TYPE,
                                      EGL_OPENGL_ES2_BIT,
                                      EGL_NONE};

    PFNEGLGETPLATFORMDISPLAYEXTPROC get_platform_display = nullptr;
    get_platform_display = reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(
        eglGetProcAddress("eglGetPlatformDisplayEXT"));
    assert(get_platform_display != nullptr);

    gl.display = get_platform_display(EGL_PLATFORM_GBM_KHR, gbm.dev, nullptr);

    if (!eglInitialize(gl.display, &major, &minor)) {
        printf("failed to initialize\n");
        return -1;
    }

    printf("Using display %p with EGL version %d.%d\n", gl.display, major, minor);

    printf("EGL Version \"%s\"\n", eglQueryString(gl.display, EGL_VERSION));
    printf("EGL Vendor \"%s\"\n", eglQueryString(gl.display, EGL_VENDOR));
    printf("EGL Extensions \"%s\"\n", eglQueryString(gl.display, EGL_EXTENSIONS));

    if (!eglBindAPI(EGL_OPENGL_ES_API)) {
        printf("failed to bind api EGL_OPENGL_ES_API\n");
        return -1;
    }

    if (!eglChooseConfig(gl.display, config_attribs, &gl.config, 1, &n) || n != 1) {
        printf("failed to choose config: %d\n", n);
        return -1;
    }

    gl.context = eglCreateContext(gl.display, gl.config, EGL_NO_CONTEXT, context_attribs);
    if (gl.context == nullptr) {
        printf("failed to create context\n");
        return -1;
    }

    gl.surface = eglCreateWindowSurface(gl.display, gl.config, gbm.surface, nullptr);
    if (gl.surface == EGL_NO_SURFACE) {
        printf("failed to create egl surface\n");
        return -1;
    }

    /* connect the context to the surface */
    eglMakeCurrent(gl.display, gl.surface, gl.surface, gl.context);

    printf("GL Extensions: \"%s\"\n", glGetString(GL_EXTENSIONS));

    return 0;
}

/* Draw code here */
static void draw(uint32_t i)
{
    glClear(GL_COLOR_BUFFER_BIT);
    glClearColor(0.2f + (i % 80 / 100.0), 0.3f, 0.5f, 1.0f);
}

static void drmFbDestroyCallback(struct gbm_bo* bo, void* data)
{
    struct DrmFb* fb = reinterpret_cast<DrmFb*>(data);
    struct gbm_device* gbm = gbm_bo_get_device(bo);

    if (fb->fb_id) {
        drmModeRmFB(drm.fd, fb->fb_id);
    }

    free(fb);
}

static struct DrmFb* drmFbGetFromBo(struct gbm_bo* bo)
{
    struct DrmFb* fb = reinterpret_cast<DrmFb*>(gbm_bo_get_user_data(bo));
    uint32_t width, height, stride, handle;
    int ret;

    if (fb) {
        return fb;
    }

    fb = reinterpret_cast<DrmFb*>(calloc(1, sizeof *fb));
    fb->bo = bo;

    width = gbm_bo_get_width(bo);
    height = gbm_bo_get_height(bo);
    stride = gbm_bo_get_stride(bo);
    handle = gbm_bo_get_handle(bo).u32;

    ret = drmModeAddFB(drm.fd, width, height, 24, 32, stride, handle, &fb->fb_id);
    if (ret) {
        printf("failed to create fb: %s\n", strerror(errno));
        free(fb);
        return nullptr;
    }

    gbm_bo_set_user_data(bo, fb, drmFbDestroyCallback);

    return fb;
}

static void pageFlipHandler(int fd, unsigned int frame, unsigned int sec, unsigned int usec,
                            void* data)
{
    int* waiting_for_flip = reinterpret_cast<int*>(data);
    *waiting_for_flip = 0;
}

int main(int argc, char* argv[])
{
    fd_set fds;
    drmEventContext evctx = {
        .version = DRM_EVENT_CONTEXT_VERSION,
        .page_flip_handler = pageFlipHandler,
    };
    struct gbm_bo* bo;
    struct DrmFb* fb;
    uint32_t i = 0;
    int ret;

    ret = initDrm(std::atoi(argv[1]));
    if (ret) {
        printf("failed to initialize DRM\n");
        return ret;
    }

    FD_ZERO(&fds);
    FD_SET(0, &fds);
    FD_SET(drm.fd, &fds);

    ret = initGbm();
    if (ret) {
        printf("failed to initialize GBM\n");
        return ret;
    }

    ret = initGL();
    if (ret) {
        printf("failed to initialize EGL\n");
        return ret;
    }

    /* clear the color buffer */
    glClearColor(0.5, 0.5, 0.5, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);
    eglSwapBuffers(gl.display, gl.surface);
    bo = gbm_surface_lock_front_buffer(gbm.surface);
    fb = drmFbGetFromBo(bo);

    /* set mode: */
    ret = drmModeSetCrtc(drm.fd, drm.crtc_id, fb->fb_id, 0, 0, &drm.connector_id, 1, drm.mode);
    if (ret) {
        printf("failed to set mode: %s\n", strerror(errno));
        return ret;
    }

    while (true) {
        struct gbm_bo* next_bo;
        int waiting_for_flip = 1;

        draw(i++);

        eglSwapBuffers(gl.display, gl.surface);
        next_bo = gbm_surface_lock_front_buffer(gbm.surface);
        fb = drmFbGetFromBo(next_bo);

        /*
         * Here you could also update drm plane layers if you want
         * hw composition
         */

        ret = drmModePageFlip(drm.fd, drm.crtc_id, fb->fb_id, DRM_MODE_PAGE_FLIP_EVENT,
                              &waiting_for_flip);
        if (ret) {
            printf("failed to queue page flip: %s\n", strerror(errno));
            return -1;
        }

        while (waiting_for_flip) {
            ret = select(drm.fd + 1, &fds, nullptr, nullptr, nullptr);
            if (ret < 0) {
                printf("select err: %s\n", strerror(errno));
                return ret;
            } else if (ret == 0) {
                printf("select timeout!\n");
                return -1;
            } else if (FD_ISSET(0, &fds)) {
                printf("user interrupted!\n");
                break;
            }
            drmHandleEvent(drm.fd, &evctx);
        }

        /* release last buffer to render on again: */
        gbm_surface_release_buffer(gbm.surface, bo);
        bo = next_bo;
    }

    return ret;
}
