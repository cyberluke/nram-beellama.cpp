// tools/kelvin-tui/src/file_picker.cpp
#include "file_picker.h"

#include <windows.h>
#include <shobjidl.h>
#include <comdef.h>

std::string pick_folder(const std::string& title) {
    // IFileDialog folder picker (Vista+). CoInitialize is expected by caller.
    IFileDialog* pfd = nullptr;
    HRESULT hr = CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER,
                                  IID_PPV_ARGS(&pfd));
    if (FAILED(hr)) return "";

    // Force folder-picking mode (no files selectable).
    DWORD opts = 0;
    pfd->GetOptions(&opts);
    pfd->SetOptions(opts | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM);

    if (!title.empty()) {
        // Best-effort; IFileDialog has no simple title override, so skip.
    }

    hr = pfd->Show(nullptr);
    std::string result;
    if (SUCCEEDED(hr)) {
        IShellItem* psi = nullptr;
        if (SUCCEEDED(pfd->GetResult(&psi))) {
            PWSTR path = nullptr;
            if (SUCCEEDED(psi->GetDisplayName(SIGDN_FILESYSPATH, &path))) {
                // Convert wchar -> narrow (best-effort, assume ASCII paths).
                int len = WideCharToMultiByte(CP_ACP, 0, path, -1, nullptr, 0, nullptr, nullptr);
                if (len > 0) {
                    result.resize(len - 1);
                    WideCharToMultiByte(CP_ACP, 0, path, -1, &result[0], len - 1, nullptr, nullptr);
                }
                CoTaskMemFree(path);
            }
            psi->Release();
        }
    }
    pfd->Release();
    return result;
}