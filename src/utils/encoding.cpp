#include "encoding.h"
#include <iconv.h>
#include <libcharset.h>
#include <clocale>
#include <memory>

namespace utils
{

class IconvWrapper
{
public:
    IconvWrapper(CodePage tocode, CodePage fromcode)
    {
        cd_ = iconv_open(code2str(tocode), code2str(fromcode));
    }

    ~IconvWrapper()
    {
        if (reinterpret_cast<iconv_t>(-1) == cd_) {
            return;
        }
        iconv_close(cd_);
    }

    size_t convert(char** inbuf, size_t* inbytesleft, char** outbuf, size_t* outbytesleft)
    {
        if (reinterpret_cast<iconv_t>(-1) == cd_) {
            return -1;
        }
        return iconv(cd_, inbuf, inbytesleft, outbuf, outbytesleft);
    }

    template <typename T = std::string>
    T convert(std::string_view in, size_t max_outbytes)
    {
        char* inbuf = const_cast<char*>(in.data());
        size_t inbytes = in.size();
        size_t outbytes = max_outbytes;
        std::unique_ptr<char> buf(new char[outbytes]{0});
        char* outbuf = buf.get();
        size_t ret = convert(&inbuf, &inbytes, &outbuf, &outbytes);
        if (static_cast<size_t>(-1) == ret) {
            return {};
        }
        using CharT = typename T::value_type;
        return T(reinterpret_cast<CharT*>(buf.get()), reinterpret_cast<CharT*>(outbuf));
    }

    static char const* code2str(CodePage cp)
    {
        switch (cp) {
            case CodePage::kLOCAL:
                setlocale(LC_ALL, "");
                return locale_charset();
            case CodePage::kUTF8:
                return "UTF-8";
            case CodePage::kGBK:
                return "GBK";
            case CodePage::kWCHAR:
                if constexpr (sizeof(wchar_t) == 2) {
                    return "UTF-16LE";
                } else {
                    return "UTF-32LE";
                }
            default:
                return "";
        }
    }

private:
    iconv_t cd_;
};

std::string ws2s(std::wstring_view ws, CodePage cp)
{
    IconvWrapper conv(cp, CodePage::kWCHAR);
    std::string_view sv{reinterpret_cast<char const*>(ws.data()), ws.size() * sizeof(wchar_t)};
    return conv.convert<std::string>(sv, ws.size() * 6);
}

std::wstring s2ws(std::string_view s, CodePage cp)
{
    IconvWrapper conv(CodePage::kWCHAR, cp);
    return conv.convert<std::wstring>(s, s.size() * 4);
}

}  // namespace utils
