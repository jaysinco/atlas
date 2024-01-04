#include "top-window.h"
#include "display-context.h"
#include "toolkit/toolkit.h"
#include "wayland-listeners.h"
#include <GLES3/gl3.h>
#include <map>
#include <thread>
#include "toolkit/logging.h"
#include "imgui/imgui.h"
#include "imgui/backends/imgui_impl_opengl3.h"
#include <boost/algorithm/string.hpp>
#include "ime-editor.h"

void FrameWindow::Init(int argc, char** argv)
{
    auto& ctx = DisplayContext::Instance();

    std::map<std::string, std::string> args = {
        {"app-id", "glviewer"},
        {"display-id", "0"},
    };
    for (int i = 1; i < argc; ++i) {
        std::vector<std::string> vec;
        boost::split(vec, argv[i], boost::is_any_of("="));
        std::string name = vec[0].substr(2);
        ILOG("args[\"{}\"]={}", name, vec[1]);
        args[name] = vec[1];
    }

    ctx.wl.display = wl_display_connect(NULL);
    assert(ctx.wl.display);
    ctx.wl.registry = wl_display_get_registry(ctx.wl.display);
    wl_registry_add_listener(ctx.wl.registry, &WaylandListeners::registry, nullptr);
    wl_display_dispatch(ctx.wl.display);

    InitEgl();
    InitSurface(args.at("app-id"));
    InitGl();
    InitImgui();
    ImeEditor::Initialize();
    ImeEditor::CreateSession(ctx.ime.session);

    ctx.wl.cursor_surface = wl_compositor_create_surface(ctx.wl.compositor);
}

void FrameWindow::Destory()
{
    auto& ctx = DisplayContext::Instance();

    ImeEditor::DestroySession(ctx.ime.session);
    ImeEditor::Destory();
    DestoryImgui();
    DestoryGl();
    DestorySurface();
    DestoryEgl();

    wl_surface_destroy(ctx.wl.cursor_surface);
    if (ctx.wl.cursor_theme) wl_cursor_theme_destroy(ctx.wl.cursor_theme);

    if (ctx.wl.shell) wl_shell_destroy(ctx.wl.shell);

    if (ctx.wl.compositor) wl_compositor_destroy(ctx.wl.compositor);

    wl_registry_destroy(ctx.wl.registry);
    wl_display_flush(ctx.wl.display);
    wl_display_disconnect(ctx.wl.display);
}

void FrameWindow::Run()
{
    auto& ctx = DisplayContext::Instance();

    int ret = 0;
    while (ctx.wl.running && ret != -1) {
        ret = wl_display_dispatch(ctx.wl.display);
    }

    fprintf(stdout, "\n");
}

void FrameWindow::InitEgl()
{
    static const EGLint context_attribs[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};

    auto& ctx = DisplayContext::Instance();

    EGLint config_attribs[] = {EGL_SURFACE_TYPE,
                               EGL_WINDOW_BIT,
                               EGL_RED_SIZE,
                               8,  // 10
                               EGL_GREEN_SIZE,
                               8,  // 10
                               EGL_BLUE_SIZE,
                               8,  // 10
                               EGL_ALPHA_SIZE,
                               8,  // 2
                               EGL_RENDERABLE_TYPE,
                               EGL_OPENGL_ES2_BIT,
                               EGL_NONE};

    EGLint major, minor, n;
    EGLBoolean ret;

    if (ctx.wl.opaque) config_attribs[9] = 0;

    ctx.egl.dpy = eglGetDisplay(ctx.wl.display);
    assert(ctx.egl.dpy);

    ret = eglInitialize(ctx.egl.dpy, &major, &minor);
    assert(ret == EGL_TRUE);
    ret = eglBindAPI(EGL_OPENGL_ES_API);
    assert(ret == EGL_TRUE);

    ret = eglChooseConfig(ctx.egl.dpy, config_attribs, &ctx.egl.conf, 1, &n);
    assert(ret && n == 1);

    ctx.egl.ctx = eglCreateContext(ctx.egl.dpy, ctx.egl.conf, EGL_NO_CONTEXT, context_attribs);
    assert(ctx.egl.ctx);
}

void FrameWindow::DestoryEgl()
{
    auto& ctx = DisplayContext::Instance();
    eglTerminate(ctx.egl.dpy);
    eglReleaseThread();
}

void FrameWindow::InitSurface(std::string const& app_id)
{
    auto& ctx = DisplayContext::Instance();
    EGLBoolean ret;

    ctx.wl.surface = wl_compositor_create_surface(ctx.wl.compositor);

    ctx.wl.xdg_surf = xdg_wm_base_get_xdg_surface(ctx.wl.xdg_wm, ctx.wl.surface);
    xdg_surface_add_listener(ctx.wl.xdg_surf, &WaylandListeners::xdg_surface, nullptr);

    ctx.wl.xdg_top = xdg_surface_get_toplevel(ctx.wl.xdg_surf);
    xdg_toplevel_add_listener(ctx.wl.xdg_top, &WaylandListeners::xdg_toplevel, nullptr);
    ILOG("xdg set app_id to '{}'", app_id);
    xdg_toplevel_set_app_id(ctx.wl.xdg_top, app_id.c_str());

    ctx.wl.native =
        wl_egl_window_create(ctx.wl.surface, ctx.window_size.width, ctx.window_size.height);
    ctx.wl.egl_surface = eglCreateWindowSurface(ctx.egl.dpy, ctx.egl.conf, ctx.wl.native, NULL);

    ret = eglMakeCurrent(ctx.egl.dpy, ctx.wl.egl_surface, ctx.wl.egl_surface, ctx.egl.ctx);
    assert(ret == EGL_TRUE);

    WaylandListeners::toggle_fullscreen(ctx.wl.fullscreen);
}

void FrameWindow::DestorySurface()
{
    auto& ctx = DisplayContext::Instance();

    /* Required, otherwise segfault in egl_dri2.c: dri2_make_current()
     * on eglReleaseThread(). */
    eglMakeCurrent(ctx.egl.dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);

    eglDestroySurface(ctx.egl.dpy, ctx.wl.egl_surface);
    wl_egl_window_destroy(ctx.wl.native);

    xdg_toplevel_destroy(ctx.wl.xdg_top);
    xdg_surface_destroy(ctx.wl.xdg_surf);

    // wl_shell_surface_destroy(ctx.wl.shell_surface);
    wl_surface_destroy(ctx.wl.surface);

    if (ctx.wl.callback) wl_callback_destroy(ctx.wl.callback);
}

unsigned FrameWindow::CreateShader(char const* source, unsigned shader_type)
{
    GLuint shader;
    GLint status;

    shader = glCreateShader(shader_type);
    assert(shader != 0);

    glShaderSource(shader, 1, (char const**)&source, NULL);
    glCompileShader(shader);

    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (!status) {
        char log[1000];
        GLsizei len;
        glGetShaderInfoLog(shader, 1000, &len, log);
        ELOG("failed to compiling {}: {}", shader_type == GL_VERTEX_SHADER ? "vertex" : "fragment",
             std::string_view(log, len));
        exit(1);
    }

    return shader;
}

void FrameWindow::InitGl()
{
    static char const* vert_shader_text = R"(#version 300 es
precision mediump float;

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aTexCoord;
out vec2 my_TexCoord;

void main() {
  gl_Position = vec4(aPos, 1.0);
  my_TexCoord = aTexCoord;
}

)";

    static char const* frag_shader_text = R"(#version 300 es
precision mediump float;

in vec2 my_TexCoord;
out vec4 my_FragColor;
uniform sampler2D my_texture0;

void main() {
    vec2 pos = vec2(my_TexCoord.x, 1.0 - my_TexCoord.y);
    my_FragColor = vec4(texture(my_texture0, pos).bgr, 1.0);
}

)";

    auto& ctx = DisplayContext::Instance();

    float vertices[] = {
        // positions        // texture coords
        1.0f,  1.0f,  0.0f, 1.0f, 1.0f,  // top right
        1.0f,  -1.0f, 0.0f, 1.0f, 0.0f,  // bottom right
        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,  // bottom left
        -1.0f, 1.0f,  0.0f, 0.0f, 1.0f   // top left
    };
    unsigned int indices[] = {
        0, 1, 3,  // first triangle
        1, 2, 3   // second triangle
    };

    glGenVertexArrays(1, &ctx.gl.vao);
    glGenBuffers(1, &ctx.gl.vbo);
    glGenBuffers(1, &ctx.gl.ebo);

    glBindVertexArray(ctx.gl.vao);

    glBindBuffer(GL_ARRAY_BUFFER, ctx.gl.vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ctx.gl.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // texture coord attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // texture
    glGenTextures(1, &ctx.gl.text);

    // shader
    GLuint frag, vert;
    GLuint program;
    GLint status;

    frag = CreateShader(frag_shader_text, GL_FRAGMENT_SHADER);
    vert = CreateShader(vert_shader_text, GL_VERTEX_SHADER);

    program = glCreateProgram();
    glAttachShader(program, frag);
    glAttachShader(program, vert);
    glLinkProgram(program);

    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (!status) {
        char log[1000];
        GLsizei len;
        glGetProgramInfoLog(program, 1000, &len, log);
        ELOG("failed to linking{}", std::string_view(log, len));
        exit(1);
    }

    glUseProgram(program);
    glLinkProgram(program);
}

void FrameWindow::DestoryGl() {}

static void set_platform_ime_data(ImGuiViewport* viewport, ImGuiPlatformImeData* data)
{
    auto& ctx = DisplayContext::Instance();

    if (!data->WantVisible) {
        // EsImeClearX(ctx.ime.session);

        // EsStringHndl state_json;
        // EsImeGetStateX(ctx.ime.session, &state_json);
        // sv::toolkit::SVConvertFromJsonStr(*(std::string*)state_json, ctx.ime.state);
        // EsStringFree(state_json);
    }

    ctx.ime.actived = data->WantVisible;
    ctx.ime.input_region.x = data->InputPos.x;
    ctx.ime.input_region.y = data->InputPos.y;
    ctx.ime.input_region.lh = data->InputLineHeight;
}

void FrameWindow::InitImgui()
{
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.BackendFlags |= ImGuiBackendFlags_HasMouseCursors;

    auto fontPath = (toolkit::getDataDir() / "FangZhengHeiTi.ttf").string();
    io.Fonts->AddFontFromFileTTF(fontPath.c_str(), 20.0f, nullptr,
                                 io.Fonts->GetGlyphRangesChineseSimplifiedCommon());
    ImFontConfig font_config;
    font_config.MergeMode = true;
    io.Fonts->AddFontFromFileTTF(fontPath.c_str(), 20.0f, &font_config,
                                 io.Fonts->GetGlyphRangesChineseFull());
    io.Fonts->Build();

    io.SetPlatformImeDataFn = set_platform_ime_data;
    io.IniFilename = nullptr;
    io.LogFilename = nullptr;

    ImGuiStyle& style = ImGui::GetStyle();
    style.Alpha = 1.0;
    style.ButtonTextAlign = ImVec2(0.0, 0.0);

    ImGui_ImplOpenGL3_Init("#version 300 es");
}

void FrameWindow::DestoryImgui()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui::DestroyContext();
}

void FrameWindow::Draw()
{
    auto& ctx = DisplayContext::Instance();

    glClearColor(0.0, 0.5, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);
}
