#include "ime-editor.h"
#include "toolkit/logging.h"
#include "imgui/imgui.h"
#include <sstream>
#include <iostream>
#include <X11/keysym.h>

RimeApi* ImeEditor::_rime_api;

void ImeEditor::Draw()
{
    auto& ctx = DisplayContext::Instance();
    if (!ctx.ime.actived || !ctx.ime.state.isComposing) {
        return;
    }
    ImGui::SetNextWindowSize(ImVec2(300, -1));
    ImGui::SetNextWindowPos(
        ImVec2(ctx.ime.input_region.x, ctx.ime.input_region.y + ctx.ime.input_region.lh));
    ImGui::Begin("ImeEditor", nullptr,
                 ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoDecoration);
    ImGui::Text("%s", GetComposition(ctx.ime.state).c_str());
    ImGui::TextWrapped("%s", GetMenu(ctx.ime.state).c_str());
    ImGui::End();
}

std::string ImeEditor::GetComposition(ImeState& state)
{
    std::ostringstream ss;
    size_t len = state.preEdit.size();
    size_t start = state.selStart;
    size_t end = state.selEnd;
    size_t cursor = state.cursorPos;
    for (size_t i = 0; i <= len; ++i) {
        // if (start < end)
        // {
        //     if (i == start)
        //     {
        //         ss << '[';
        //     }
        //     else if (i == end)
        //     {
        //         ss << ']';
        //     }
        // }
        if (i == cursor) ss << '|';
        if (i < len) ss << state.preEdit[i];
    }
    return ss.str();
}

std::string ImeEditor::GetMenu(ImeState& state)
{
    std::ostringstream ss;
    for (int i = 0; i < state.candidates.size(); ++i) {
        if (i > 0) {
            ss << "  ";
        }
        ss << i + 1 << ". " << state.candidates.at(i);
    }
    return ss.str();
}

MyErrCode ImeEditor::Initialize()
{
    _rime_api = rime_get_api();

    RIME_STRUCT(RimeTraits, traits);
    auto rootdir = toolkit::currentExeDir();
    std::string log_dir = toolkit::getLoggingDir();
    std::string ime_dir = (rootdir / "ime").string();
    std::string staging_dir = (rootdir / "ime/staging").string();

    traits.app_name = "ime";
    traits.log_dir = log_dir.c_str();
    traits.shared_data_dir = ime_dir.c_str();
    traits.user_data_dir = ime_dir.c_str();
    traits.staging_dir = staging_dir.c_str();
    traits.prebuilt_data_dir = staging_dir.c_str();
    traits.min_log_level = 2;
    _rime_api->setup(&traits);

    _rime_api->set_notification_handler(&ImeEditor::OnMessage, NULL);

    ILOG("ime initializing");
    _rime_api->initialize(nullptr);
    Bool full_check = True;
    if (_rime_api->start_maintenance(full_check)) {
        _rime_api->join_maintenance_thread();
    }
    ILOG("ime ready");

    return MyErrCode::kOk;
}

MyErrCode ImeEditor::Destory()
{
    _rime_api->finalize();
    return MyErrCode::kOk;
}

void ImeEditor::OnMessage(void* context_object, RimeSessionId session_id, char const* message_type,
                          char const* message_value)
{
    ILOG("ime: [{}] [{}] {}", session_id, message_type, message_value);
    RimeApi* rime = rime_get_api();
    if (RIME_API_AVAILABLE(rime, get_state_label) && !strcmp(message_type, "option")) {
        bool state = message_value[0] != '!';
        char const* option_name = message_value + !state;
        char const* state_label = rime->get_state_label(session_id, option_name, state);
        if (state_label) {
            ILOG("ime: updated option: {} = {} // {}", option_name, state, state_label);
        }
    }
}

MyErrCode ImeEditor::CreateSession(RimeSessionId& session_id)
{
    session_id = _rime_api->create_session();
    if (!session_id) {
        ELOG("failed to creating rime session");
        return MyErrCode::kFailed;
    }
    return MyErrCode::kOk;
}

MyErrCode ImeEditor::DestroySession(RimeSessionId session_id)
{
    if (!_rime_api->destroy_session(session_id)) {
        ELOG("failed to destroy ime session");
        return MyErrCode::kFailed;
    }
    return MyErrCode::kOk;
}

MyErrCode ImeEditor::ProcessKey(RimeSessionId session_id, int keycode, int mask)
{
    if (!_rime_api->process_key(session_id, keycode, mask)) {
        ELOG("failed to process key: {} {}", keycode, mask);
        return MyErrCode::kFailed;
    }
    return MyErrCode::kOk;
}

MyErrCode ImeEditor::GetState(RimeSessionId session_id, ImeState& state)
{
    state = {};
    RIME_STRUCT(RimeStatus, stat);
    RIME_STRUCT(RimeContext, ctx);

    if (!_rime_api->get_status(session_id, &stat)) {
        ELOG("failed to get status");
        return MyErrCode::kFailed;
    }
    auto stat_guard = toolkit::scopeExit([&] { _rime_api->free_status(&stat); });
    state.isComposing = stat.is_composing;
    if (!state.isComposing) {
        return MyErrCode::kOk;
    }

    if (!_rime_api->get_context(session_id, &ctx)) {
        ELOG("failed to get context");
        return MyErrCode::kFailed;
    }
    auto ctx_guard = toolkit::scopeExit([&] { _rime_api->free_context(&ctx); });
    if (ctx.composition.length <= 0) {
        // not composing
        return MyErrCode::kOk;
    }

    state.preEdit = std::string(ctx.composition.preedit, ctx.composition.length);
    state.cursorPos = ctx.composition.cursor_pos;
    state.selStart = ctx.composition.sel_start;
    state.selEnd = ctx.composition.sel_end;

    state.pageNo = ctx.menu.page_no + 1;
    state.isLastPage = ctx.menu.is_last_page;
    state.candidates.clear();
    for (int i = 0; i < ctx.menu.num_candidates; ++i) {
        state.candidates.push_back(ctx.menu.candidates[i].text);
    }

    return MyErrCode::kOk;
}

MyErrCode ImeEditor::PrintState(ImeState const& state)
{
    std::cout << "=======================" << std::endl;
    if (!state.isComposing) {
        std::cout << "<not composing>" << std::endl;
        return MyErrCode::kOk;
    }
    for (size_t i = 0; i <= state.preEdit.size(); ++i) {
        if (state.selStart < state.selEnd) {
            if (i == state.selStart) {
                std::cout << '[';
            } else if (i == state.selEnd) {
                std::cout << ']';
            }
        }
        if (i == state.cursorPos) {
            std::cout << '|';
        }
        std::cout << state.preEdit[i];
    }
    std::cout << std::endl;
    std::cout << fmt::format("page: {}{}", state.pageNo, state.isLastPage ? '$' : ' ') << std::endl;
    for (int i = 0; i < state.candidates.size(); ++i) {
        std::cout << fmt::format("{}. {}", i + 1, state.candidates[i]) << std::endl;
    }
    return MyErrCode::kOk;
}

MyErrCode ImeEditor::PrintState(RimeSessionId session_id)
{
    ImeState state;
    CHECK_ERR_RET(GetState(session_id, state));
    CHECK_ERR_RET(PrintState(state));
    return MyErrCode::kOk;
}

MyErrCode ImeEditor::GetCommit(RimeSessionId session_id, std::string& output)
{
    RIME_STRUCT(RimeCommit, cmt);
    if (!_rime_api->get_commit(session_id, &cmt)) {
        // skip invalid commit
        return MyErrCode::kOk;
    }
    auto cmt_guard = toolkit::scopeExit([&] { _rime_api->free_commit(&cmt); });
    output = std::string(cmt.text);
    return MyErrCode::kOk;
}

MyErrCode ImeEditor::SetAsciiMode(RimeSessionId session_id, bool on)
{
    _rime_api->set_option(session_id, "ascii_mode", on);
    return MyErrCode::kOk;
}

MyErrCode ImeEditor::Clear(RimeSessionId session_id)
{
    _rime_api->clear_composition(session_id);
    return MyErrCode::kOk;
}
