// tools/kelvin-tui/src/file_picker.h
#pragma once

#include <string>

// Opens the native Windows folder-picker (IFileDialog). Returns the chosen
// folder path, or empty string if the user cancelled.
std::string pick_folder(const std::string& title);