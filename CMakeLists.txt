cmake_minimum_required(VERSION 3.10)

project(amd_jni)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-pthread")

if(ENABLE_ASAN)
    message(STATUS "BUILD WITH ADDRESS SANITIZER")
    set(CMAKE_C_FLAGS_DEBUG
        "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address -fsanitize=undefined"
    )
    set(CMAKE_CXX_FLAGS_DEBUG
        "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address -fsanitize=undefined"
    )
    set(CMAKE_LINKER_FLAGS_DEBUG
        "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address -fsanitize=undefined"
    )
endif()

set(BUILD_SHARED_LIBS ON)

# JNI
find_package(JNI REQUIRED)
if(JNI_FOUND)
    set(JNI_INCLUDE_DIRS ${JAVA_INCLUDE_PATH} ${JAVA_INCLUDE_PATH2})
    message(STATUS "JNI_INCLUDE_DIRS=${JNI_INCLUDE_DIRS}")
endif()

# Threads
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# ONNX Runtime (you’ll need to point this to the right path on the SER8)
set(ONNXRUNTIME_DIR ${CMAKE_SOURCE_DIR}/src/main/native/lib/onnxruntime)
set(ONNXRUNTIME_LIB ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so)
set(ONNXRUNTIME_INCLUDE ${ONNXRUNTIME_DIR}/include)

# OpenCV (PhotonVision depends on it)
set(OPENCV_YEAR "frc2025")
set(OPENCV_VERSION "4.10.0-3")
set(OPENCV_TYPE "")

include(FetchContent)
fetchcontent_declare(
    opencv_lib
    URL
        https://frcmaven.wpi.edu/artifactory/release/edu/wpi/first/thirdparty/${OPENCV_YEAR}/opencv/opencv-cpp/${OPENCV_VERSION}/opencv-cpp-${OPENCV_VERSION}-linuxx86-64${OPENCV_TYPE}.zip
)
fetchcontent_makeavailable(opencv_lib)

fetchcontent_declare(
    opencv_header
    URL
        https://frcmaven.wpi.edu/artifactory/release/edu/wpi/first/thirdparty/${OPENCV_YEAR}/opencv/opencv-cpp/${OPENCV_VERSION}/opencv-cpp-${OPENCV_VERSION}-headers.zip
)
fetchcontent_makeavailable(opencv_header)

file(
    GLOB_RECURSE OPENCV_LIB_PATH
    "${opencv_lib_SOURCE_DIR}/**/*.lib"
    "${opencv_lib_SOURCE_DIR}/**/*.so*"
)
set(OPENCV_INCLUDE_PATH ${opencv_header_SOURCE_DIR})
message("Depending on OpenCV ${OPENCV_LIB_PATH}")

# Build JNI library
add_library(
    ${PROJECT_NAME}
    src/main/native/cpp/amd_jni.cpp
    src/main/native/cpp/detector.cpp
    src/main/native/include/detector.h
    # src/main/native/cpp/yolo_common.cpp
    # src/main/native/cpp/preprocess.cc
    # src/main/native/cpp/postprocess.cc
)

target_link_libraries(
    ${PROJECT_NAME}
    PUBLIC Threads::Threads ${ONNXRUNTIME_LIB} ${OPENCV_LIB_PATH}
)

target_include_directories(
    ${PROJECT_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/main/native/include
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${JNI_INCLUDE_DIRS}
        ${ONNXRUNTIME_INCLUDE}
        ${OPENCV_INCLUDE_PATH}
)

set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "\\\$ORIGIN/")

include(GNUInstallDirs)
install(TARGETS ${PROJECT_NAME})
install(FILES ${ONNXRUNTIME_LIB} TYPE LIB)

# Simple test program
# add_executable(main_test src/main/native/cpp/main_test.cc)
# target_link_libraries(main_test ${PROJECT_NAME})
# target_include_directories(
#     main_test
#     PRIVATE
#         ${CMAKE_CURRENT_SOURCE_DIR}/src/main/native/include
#         ${OPENCV_INCLUDE_PATH}
# )
