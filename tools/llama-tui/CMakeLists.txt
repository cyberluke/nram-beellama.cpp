# tools/kelvin-tui/CMakeLists.txt
#
# "The Lab" - Kelvin the Software Chemist's dependency-distilling TUI.
# A single-exe, vendored-FTXUI front-end that owns a llama-server
# subprocess and provides a streaming ASCII-art chat TUI.
#
# Runtime deps: none beyond the Windows SDK (winhttp, ws2_32). FTXUI is
# vendored under vendor/ftxui and built here via add_subdirectory.

# --- FTXUI (vendored, lean) -------------------------------------------------
option(FTXUI_BUILD_EXAMPLES   "" OFF)
option(FTXUI_BUILD_DOCS       "" OFF)
option(FTXUI_BUILD_TESTS      "" OFF)
option(FTXUI_BUILD_MODULES    "" OFF)
option(FTXUI_ENABLE_INSTALL   "" OFF)
option(FTXUI_QUIET            "" ON)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../vendor/ftxui
                 ${CMAKE_CURRENT_BINARY_DIR}/ftxui EXCLUDE_FROM_ALL)

# --- kelvin-tui executable --------------------------------------------------
add_executable(kelvin-tui
    src/main.cpp
    src/version.h
    src/model_detector.h
    src/model_detector.cpp
    src/presets.h
    src/presets.cpp
    src/settings.h
    src/settings.cpp
    src/i18n.h
    src/i18n.cpp
    src/file_picker.h
    src/file_picker.cpp
    src/server_proc.h
    src/server_proc.cpp
    src/http_client.h
    src/http_client.cpp
    src/chat_client.h
    src/chat_client.cpp
    src/tui_ui.h
    src/tui_ui.cpp
)
target_include_directories(kelvin-tui PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(kelvin-tui PRIVATE
    ftxui::ftxui
    winhttp
    ws2_32
    ole32
    shlwapi
    uuid
    shell32
    ${CMAKE_DL_LIBS}
)

# The TUI talks HTTP to a running llama-server; it does NOT link llama/ggml,
# keeping the exe small and dependency-free.

target_compile_features(kelvin-tui PRIVATE cxx_std_17)

if (WIN32)
    target_compile_definitions(kelvin-tui PRIVATE
        _UNICODE
        UNICODE
        WIN32_LEAN_AND_MEAN
        NOMINMAX
    )
endif()

install(TARGETS kelvin-tui RUNTIME DESTINATION bin)


add_executable(repro2 src/repro2.cpp)
target_include_directories(repro2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(repro2 PRIVATE ftxui::ftxui winhttp ws2_32 ole32 shlwapi uuid shell32 ${CMAKE_DL_LIBS})
target_compile_features(repro2 PRIVATE cxx_std_17)
if (WIN32)
    target_compile_definitions(repro2 PRIVATE _UNICODE UNICODE WIN32_LEAN_AND_MEAN NOMINMAX)
endif()
