# dependencies

find_package(Threads REQUIRED)

# third-party

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if (EMSCRIPTEN)
else()
    if (NOT GGML_BACKEND_DL)
        add_subdirectory(vdot)
    endif()

    find_package(CUDAToolkit)
    find_package(OpenMP)
    if (CUDAToolkit_FOUND)
        add_executable(cpu_h2d_interference cpu_h2d_interference.cpp)
        target_link_libraries(cpu_h2d_interference PRIVATE CUDA::cudart Threads::Threads)
        if(OpenMP_CXX_FOUND)
            target_link_libraries(cpu_h2d_interference PRIVATE OpenMP::OpenMP_CXX)
        endif()
    endif()
endif()
