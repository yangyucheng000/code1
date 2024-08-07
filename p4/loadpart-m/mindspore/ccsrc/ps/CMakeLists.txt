file(GLOB_RECURSE _PS_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

if(NOT ENABLE_CPU OR WIN32)
    list(REMOVE_ITEM _PS_SRC_FILES "optimizer_info_builder.cc")
    list(REMOVE_ITEM _PS_SRC_FILES "optimizer_info.cc")
    list(REMOVE_ITEM _PS_SRC_FILES "scheduler.cc")
    list(REMOVE_ITEM _PS_SRC_FILES "util.cc")
    list(REMOVE_ITEM _PS_SRC_FILES "embedding_table_shard_metadata.cc")
    list(REMOVE_ITEM _PS_SRC_FILES "core/communicator/http_message_handler.cc")
    list(REMOVE_ITEM _PS_SRC_FILES "core/communicator/http_server.cc")
    list(REMOVE_ITEM _PS_SRC_FILES "core/comm_util.cc")
    list(REMOVE_ITEM _PS_SRC_FILES "core/communicator/tcp_client.cc")
    list(REMOVE_ITEM _PS_SRC_FILES "core/communicator/tcp_message_handler.cc")
    list(REMOVE_ITEM _PS_SRC_FILES "core/communicator/tcp_server.cc")
    list(REMOVE_ITEM _PS_SRC_FILES "core/node.cc")
    list(REMOVE_ITEM _PS_SRC_FILES "core/node_manager.cc")
    list(REMOVE_ITEM _PS_SRC_FILES "ps_cache/ps_cache_manager.cc")
    list(REMOVE_ITEM _PS_SRC_FILES "core/worker_node.cc")
    list(REMOVE_ITEM _PS_SRC_FILES "core/ps_worker_node.cc")
    list(REMOVE_ITEM _PS_SRC_FILES "core/server_node.cc")
    list(REMOVE_ITEM _PS_SRC_FILES "core/ps_server_node.cc")
    list(REMOVE_ITEM _PS_SRC_FILES "core/abstract_node.cc")
    list(REMOVE_ITEM _PS_SRC_FILES "core/scheduler_node.cc")
    list(REMOVE_ITEM _PS_SRC_FILES "core/communicator/http_client.cc")
    list(REMOVE_ITEM _PS_SRC_FILES "worker.cc")
    list(REMOVE_ITEM _PS_SRC_FILES "parameter_server.cc")
    list(REMOVE_ITEM _PS_SRC_FILES "core/communicator/http_request_handler.cc")
    list(REMOVE_ITEM _PS_SRC_FILES "core/communicator/ssl_wrapper.cc")
    list(REMOVE_ITEM _PS_SRC_FILES "core/communicator/ssl_http.cc")
    list(REMOVE_ITEM _PS_SRC_FILES "core/communicator/ssl_client.cc")
    list(REMOVE_ITEM _PS_SRC_FILES "core/leader_scaler.cc")
    list(REMOVE_ITEM _PS_SRC_FILES "core/follower_scaler.cc")
    list(REMOVE_ITEM _PS_SRC_FILES "core/file_configuration.cc")
    list(REMOVE_ITEM _PS_SRC_FILES "core/recovery_base.cc")
    list(REMOVE_ITEM _PS_SRC_FILES "core/node_recovery.cc")
    list(REMOVE_ITEM _PS_SRC_FILES "core/scheduler_recovery.cc")
    list(REMOVE_ITEM _PS_SRC_FILES "core/communicator/communicator_base.cc")
    list(REMOVE_ITEM _PS_SRC_FILES "core/communicator/http_communicator.cc")
    list(REMOVE_ITEM _PS_SRC_FILES "core/communicator/tcp_communicator.cc")
    list(REMOVE_ITEM _PS_SRC_FILES "core/communicator/http_msg_handler.cc")
    list(REMOVE_ITEM _PS_SRC_FILES "core/communicator/tcp_msg_handler.cc")
    list(REMOVE_ITEM _PS_SRC_FILES "core/instance_manager.cc")
endif()

if(NOT ENABLE_D)
    list(REMOVE_ITEM _PS_SRC_FILES "ps_cache/ascend/ascend_ps_cache.cc")
endif()

if(NOT ENABLE_GPU)
    list(REMOVE_ITEM _PS_SRC_FILES "ps_cache/gpu/gpu_ps_cache.cc")
endif()

list(REMOVE_ITEM _PS_SRC_FILES "ps_cache/ps_data/ps_data_prefetch.cc")
list(REMOVE_ITEM _PS_SRC_FILES "ps_cache/ps_data/ps_data_channel.cc")
add_subdirectory(ps_cache)

set_property(SOURCE ${_PS_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_PS)
add_library(_mindspore_ps_obj OBJECT ${_PS_SRC_FILES})
target_link_libraries(_mindspore_ps_obj mindspore::flatbuffers)
