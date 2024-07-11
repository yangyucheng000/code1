# Include generated *.pb.h files
file(GLOB_RECURSE CORE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "*.cc"
)

list(APPEND OFFLOADING_SRC ${CORE_SRC_LIST})

set_property(SOURCE ${OFFLOADING_SRC} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_CORE)

add_library(_mindspore_offloading_obj OBJECT ${OFFLOADING_SRC})

add_dependencies(_mindspore_offloading_obj mindspore::cblosc2)

target_link_libraries(_mindspore_offloading_obj PRIVATE pthread)
if(ENABLE_GPU)
    target_link_libraries(_mindspore_offloading_obj PRIVATE ${CUDA_PATH}/lib64/stubs/libnvidia-ml.so)
endif()