#pragma once
#include <string>

class FrameWindow
{
public:
    static void Init(int argc, char** argv);
    static void Destory();
    static void Run();
    static void Draw();

private:
    static void InitGl();
    static void DestoryGl();
    static void InitEgl();
    static void DestoryEgl();
    static void InitSurface(std::string const& app_id);
    static void DestorySurface();
    static void InitImgui();
    static void DestoryImgui();
    static unsigned CreateShader(char const* source, unsigned shader_type);
};
