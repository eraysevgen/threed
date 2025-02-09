#
# We assume you're using MSVC if you're on WIN32.
#

function(target_compile_settings target)
    target_compile_definitions(${target} PRIVATE
        -DWIN32_LEAN_AND_MEAN)
    if (MSVC)
        target_compile_definitions(${target} PRIVATE
            -D_CRT_SECURE_NO_DEPRECATE
            -D_CRT_SECURE_NO_WARNINGS
            -D_CRT_NONSTDC_NO_WARNING
            -D_SCL_SECURE_NO_WARNINGS
        )
        target_compile_options(${target} PRIVATE
            # Yes, we don't understand GCC pragmas
            /wd4068
            # Windows warns about integer narrowing like crazy and it's
            # annoying.  In most cases the programmer knows what they're
            # doing.  A good static analysis tool would be better than
            # turning this warning off.
            /wd4267
            # Annoying warning about function hiding with virtual
            # inheritance.
            /wd4250
            # MSVC doesn't understand bool -> int conversion
            /wd4805
            # Standard C++-type exception handling.
            /EHsc
            )

        include(ProcessorCount)
        ProcessorCount(N)
        if(NOT N EQUAL 0)
            target_compile_options(${target} PRIVATE "/MP${N}")
        endif()

    endif()
endfunction()

function(library_compile_settings lib type)
    target_compile_settings(${lib})
endfunction()

#
# Windows htonl and similar are in winsock :(
#
set(WINSOCK_LIBRARY ws2_32)