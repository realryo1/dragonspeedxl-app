cmake_minimum_required(VERSION 3.22)
project(dsxl_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# QNN ヘッダーは android/third_party/qnn/include（wsl/qnn.sh headers で QAIRT SDK からコピーする）
set(QNN_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/qnn/include)

add_library(dsxl_native SHARED
    jni_bridge.cpp
    engine/qnn_model.cpp
    engine/clip_tokenizer.cpp
    engine/clip_service.cpp
    engine/preview.cpp
    engine/refine.cpp
    engine/pipeline.cpp)

target_include_directories(dsxl_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${QNN_INCLUDE})
# debug でも最適化する（端末に入れるのは主に debug。-O0 だと VAE の合成・RGB 変換・FP16 変換などの CPU のループが遅い）。
# -ffast-math は使わない（浮動小数の結果を変えない）。外に見せる記号は JNIEXPORT の関数だけにし、使わないコードは捨てる
target_compile_options(dsxl_native PRIVATE -O3 -fvisibility=hidden -fvisibility-inlines-hidden -ffunction-sections -fdata-sections)
target_link_options(dsxl_native PRIVATE -Wl,--gc-sections -Wl,--exclude-libs,ALL)
target_link_libraries(dsxl_native dl log android)
