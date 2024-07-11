if(ENABLE_GITEE)
    set(REQ_URL "https://codeload.github.com/Blosc/c-blosc2/tar.gz/refs/tags/v2.4.1")
    set(MD5 "c0889859093dc2acdd25055eb1fc9d30")
else()
    set(REQ_URL "https://codeload.github.com/Blosc/c-blosc2/tar.gz/refs/tags/v2.4.1")
    set(MD5 "c0889859093dc2acdd25055eb1fc9d30")
endif()

mindspore_add_pkg(c-blosc2
        VER 2.4.1
        LIBS blosc2
        URL ${REQ_URL}
        MD5 ${MD5}
        CMAKE_OPTION -DCMAKE_BUILD_TYPE:STRING=Release)
        # INSTALL_INCS build/include/*
        # INSTALL_LIBS build/lib/*)

include_directories(${c-blosc2_INC})
add_library(mindspore::cblosc2 ALIAS c-blosc2::blosc2)