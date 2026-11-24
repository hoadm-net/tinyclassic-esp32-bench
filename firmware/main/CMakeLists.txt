# MODEL_DIR holds model_info.h, adapter.c/.cpp and the tool output.
# bench.firmware copies each configuration into build/<profile>/model and passes that path.
if(NOT MODEL_DIR)
    set(MODEL_DIR "${CMAKE_CURRENT_LIST_DIR}/../model_stub")
endif()

# ESP-IDF first evaluates this file in script mode to collect requirements;
# CONFIGURE_DEPENDS is only valid in the real configure pass.
if(CMAKE_BUILD_EARLY_EXPANSION)
    set(MODEL_SRCS "")
else()
    file(GLOB MODEL_SRCS CONFIGURE_DEPENDS "${MODEL_DIR}/*.c" "${MODEL_DIR}/*.cpp")
endif()

set(model_includes "${MODEL_DIR}")
if(EMLEARN_INCLUDE_DIR)
    list(APPEND model_includes "${EMLEARN_INCLUDE_DIR}")
endif()

idf_component_register(
    SRCS "main.cpp" "link.c" ${MODEL_SRCS}
    INCLUDE_DIRS "."
    PRIV_INCLUDE_DIRS ${model_includes}
    PRIV_REQUIRES esp_driver_uart esp_driver_usb_serial_jtag esp_hw_support freertos heap log)

# The tool output is compiled as is; its warnings are not ours to fix.
set(model_options "-w")
# MODEL_LITPOOLS=ON places literal pools next to the code that uses them, for functions too
# large for the 256 KB reach of the Xtensa L32R instruction. Used only when the default build
# fails with "l32r: literal target out of range".
if(MODEL_LITPOOLS AND CONFIG_IDF_TARGET_ARCH_XTENSA)
    list(APPEND model_options "-mtext-section-literals" "-mauto-litpools")
endif()
set_source_files_properties(${MODEL_SRCS} PROPERTIES COMPILE_OPTIONS "${model_options}")
