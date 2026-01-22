cmake_minimum_required(VERSION 3.18)
project(marvin-cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# =============================================================================
# ONNX Runtime GPU Download
# =============================================================================
# ONNX Runtime 1.20.1 requires CUDA 12.x and cuDNN 9.x
set(ONNXRUNTIME_VERSION "1.20.1")
set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-win-x64-gpu-${ONNXRUNTIME_VERSION}.zip")

message(STATUS "Downloading ONNX Runtime GPU ${ONNXRUNTIME_VERSION}...")

FetchContent_Declare(
    onnxruntime
    URL ${ONNXRUNTIME_URL}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_MakeAvailable(onnxruntime)

set(ONNXRUNTIME_ROOT "${onnxruntime_SOURCE_DIR}")
message(STATUS "ONNX Runtime root: ${ONNXRUNTIME_ROOT}")

# =============================================================================
# cuDNN 9.x Download from NVIDIA
# =============================================================================
# Official NVIDIA redistribution: https://developer.download.nvidia.com/compute/cudnn/redist/
set(CUDNN_VERSION "9.6.0.74")
set(CUDNN_URL "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-${CUDNN_VERSION}_cuda12-archive.zip")

message(STATUS "Downloading cuDNN ${CUDNN_VERSION} from NVIDIA...")

FetchContent_Declare(
    cudnn
    URL ${CUDNN_URL}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_MakeAvailable(cudnn)

set(CUDNN_ROOT "${cudnn_SOURCE_DIR}")
message(STATUS "cuDNN root: ${CUDNN_ROOT}")

# =============================================================================
# ONNX Runtime Library Setup
# =============================================================================
add_library(onnxruntime SHARED IMPORTED)

set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
    IMPORTED_IMPLIB "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include"
)

# =============================================================================
# Main Executable
# =============================================================================
add_executable(marvin-cpp main.cpp)

target_link_libraries(marvin-cpp PRIVATE onnxruntime)

target_include_directories(marvin-cpp PRIVATE
    ${ONNXRUNTIME_ROOT}/include
)

# =============================================================================
# Post-Build: Copy DLLs
# =============================================================================
# Copy ONNX Runtime DLLs
add_custom_command(TARGET marvin-cpp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
        "$<TARGET_FILE_DIR:marvin-cpp>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.dll"
        "$<TARGET_FILE_DIR:marvin-cpp>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_cuda.dll"
        "$<TARGET_FILE_DIR:marvin-cpp>"
    COMMENT "Copying ONNX Runtime DLLs..."
)

# Copy cuDNN DLLs from NVIDIA download
add_custom_command(TARGET marvin-cpp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CUDNN_ROOT}/bin/cudnn64_9.dll"
        "$<TARGET_FILE_DIR:marvin-cpp>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CUDNN_ROOT}/bin/cudnn_adv64_9.dll"
        "$<TARGET_FILE_DIR:marvin-cpp>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CUDNN_ROOT}/bin/cudnn_cnn64_9.dll"
        "$<TARGET_FILE_DIR:marvin-cpp>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CUDNN_ROOT}/bin/cudnn_engines_precompiled64_9.dll"
        "$<TARGET_FILE_DIR:marvin-cpp>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CUDNN_ROOT}/bin/cudnn_engines_runtime_compiled64_9.dll"
        "$<TARGET_FILE_DIR:marvin-cpp>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CUDNN_ROOT}/bin/cudnn_graph64_9.dll"
        "$<TARGET_FILE_DIR:marvin-cpp>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CUDNN_ROOT}/bin/cudnn_heuristic64_9.dll"
        "$<TARGET_FILE_DIR:marvin-cpp>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CUDNN_ROOT}/bin/cudnn_ops64_9.dll"
        "$<TARGET_FILE_DIR:marvin-cpp>"
    COMMENT "Copying cuDNN DLLs from NVIDIA..."
)

# Copy model file
set(MODEL_PATH "${CMAKE_SOURCE_DIR}/../inference/marvin_small.onnx")
if(EXISTS ${MODEL_PATH})
    add_custom_command(TARGET marvin-cpp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${MODEL_PATH}"
            "$<TARGET_FILE_DIR:marvin-cpp>"
        COMMENT "Copying ONNX model..."
    )
endif()
