# Build the llama-lab Rust TUI with Cargo and place llama-lab.exe in the same
# bin/ directory as llama-server.exe (and its DLLs), so the TUI finds the
# server right next to itself — no separate dist assembly needed.
set(LLAMA_LAB_CARGO_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

add_custom_target(llama-lab
    COMMENT "Building llama-lab Rust TUI (cargo build --release)"
    COMMAND cargo build --release --manifest-path "${LLAMA_LAB_CARGO_DIR}/Cargo.toml"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${LLAMA_LAB_CARGO_DIR}/target/release/llama-lab${CMAKE_EXECUTABLE_SUFFIX}"
            "${CMAKE_BINARY_DIR}/bin/llama-lab${CMAKE_EXECUTABLE_SUFFIX}"
    WORKING_DIRECTORY "${LLAMA_LAB_CARGO_DIR}"
    VERBATIM
)