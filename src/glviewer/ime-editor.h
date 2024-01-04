#pragma once
#include "toolkit/error.h"
#include "display-context.h"
#include <rime_api.h>

class ImeEditor
{
public:
    static void Draw();
    static MyErrCode Initialize();
    static MyErrCode Destory();
    static MyErrCode CreateSession(RimeSessionId& session_id);
    static MyErrCode DestroySession(RimeSessionId session_id);
    static MyErrCode ProcessKey(RimeSessionId session_id, int keycode, int mask = 0);
    static MyErrCode GetState(RimeSessionId session_id, ImeState& state);
    static MyErrCode PrintState(RimeSessionId session_id);
    static MyErrCode PrintState(ImeState const& state);
    static MyErrCode Clear(RimeSessionId session_id);
    static MyErrCode GetCommit(RimeSessionId session_id, std::string& output);
    static MyErrCode SetAsciiMode(RimeSessionId session_id, bool on);

private:
    static std::string GetComposition(ImeState& state);
    static std::string GetMenu(ImeState& state);
    static void OnMessage(void* context_object, RimeSessionId session_id, char const* message_type,
                          char const* message_value);

private:
    static RimeApi* _rime_api;
};