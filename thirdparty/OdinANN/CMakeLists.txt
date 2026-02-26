cmake_minimum_required(VERSION 3.0)
project(odinann)

set(CMAKE_STANDARD 17)
option(USE_AIO "Use AIO instead of liburing" ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_C_COMPILER_LAUNCHER ccache)
# set(CMAKE_CXX_COMPILER_LAUNCHER ccache)

# Check if liburing is installed in a custom location
if(EXISTS ${PROJECT_SOURCE_DIR}/third_party/liburing/install)
    set(LIBURING_CUSTOM_INSTALL TRUE)
    set(LIBURING_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/third_party/liburing/install/include)
    set(LIBURING_LIB_DIR ${PROJECT_SOURCE_DIR}/third_party/liburing/install/lib)
endif()

include_directories(
	${PROJECT_SOURCE_DIR}/include 
	${PROJECT_SOURCE_DIR}/include/tsl/include 
	${LIBURING_INCLUDE_DIR}
)

add_compile_options(
	-fopenmp 
	-fopenmp-simd
	-ffast-math
	-Wall
	-Wextra
	-Wfatal-errors
	-Wno-class-memaccess
	-Wno-unused-variable
	-Wno-unused-parameter
	-Wno-unused-but-set-variable
)
add_compile_options(-g -O3 -march=native -mtune=native -ftree-vectorize -funroll-loops -fno-omit-frame-pointer)

message("Additional definitions: " $ENV{ADDITIONAL_DEFINITIONS})
add_definitions(
	$ENV{ADDITIONAL_DEFINITIONS}
	-DNDEBUG
	# -DIN_PLACE_RECORD_UPDATE # disable design #1
	# -DDIRECT_READ_CC # disable opt #1.1
	# -DBG_IO_THREAD # enable opt #1.2
	# -DDELTA_PRUNING # enable opt #2

	# -DNO_POLLUTE_ORIGINAL

	# optimizations.
	# -DREAD_ONLY_TESTS
	# -DNO_MAPPING # starling: must disable this!
	-DOVERLAP_INIT
	-DDYN_PIPE_WIDTH
	
	# policy.
	# -DSTATIC_POLICY

	# unoptimizations.
	# -DNAIVE_PIPE
)

include_directories(
	${PROJECT_SOURCE_DIR}/third_party/liburing/src/include
)
link_directories(
	${PROJECT_SOURCE_DIR}/third_party/liburing/src
	${LIBURING_LIB_DIR}
)

if(USE_AIO)
	message(STATUS "Using AIO for asynchronous I/O, compatible but slower.")
	link_libraries(aio)
	add_definitions(-DUSE_AIO)
else()
	message(STATUS "Using liburing for asynchronous I/O, faster but requires kernel support.")
	if(LIBURING_CUSTOM_INSTALL)
		link_libraries(${LIBURING_LIB_DIR}/liburing.a)
	else()
		link_libraries(uring)
	endif()
endif()

link_libraries(
	jemalloc
)
# add_compile_options(-fsanitize=address,undefined -lasan -lubsan)
# add_link_options(-fsanitize=address,undefined)
# add_compile_options(-fsanitize=undefined -lubsan)
# add_link_options(-fsanitize=undefined)

# check for AVX512 and AVX2, add the corresponding definitions.
# First, for the compiler.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)

# Then, for the runtime.
set(CPU_SUPPORTS_AVX512 FALSE)
set(CPU_SUPPORTS_AVX2 FALSE)
set(CPU_SUPPORTS_SVE2 FALSE)
try_run(
	RUN_RESULT
	COMPILE_RESULT
	"${CMAKE_BINARY_DIR}"
	SOURCES "${PROJECT_SOURCE_DIR}/tests/test_cpu.cpp"
	COMPILE_OUTPUT_VARIABLE COMPILE_OUT
	RUN_OUTPUT_VARIABLE RUN_OUT
)

function(check_sve2_cpuinfo OUTPUT_VAR)
    if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64")
        set(${OUTPUT_VAR} FALSE PARENT_SCOPE)
        return()
    endif()

    set(HAVE_SVE2 FALSE)
    if(EXISTS "/proc/cpuinfo")
        file(READ "/proc/cpuinfo" CPUINFO)
        if(CPUINFO MATCHES "flags[ \t]*:.*[ \t]sve2([ \t]|$)" OR CPUINFO MATCHES "Features[ \t]*:.*[ \t]sve2([ \t]|$)")
            set(HAVE_SVE2 TRUE)
        endif()
    endif()
    set(${OUTPUT_VAR} ${HAVE_SVE2} PARENT_SCOPE)
endfunction()

check_sve2_cpuinfo(COMPILER_SUPPORTS_SVE2)

set(USE_ARMPL FALSE)
if(COMPILE_RESULT AND RUN_RESULT EQUAL 3) # AVX512 + AVX2
	set(CPU_SUPPORTS_AVX512 TRUE)
	message(STATUS "CPU supports AVX512")
elseif(COMPILE_RESULT AND RUN_RESULT EQUAL 1) # AVX2
	set(CPU_SUPPORTS_AVX2 TRUE)
	message(STATUS "CPU supports AVX2")
elseif(COMPILER_SUPPORTS_SVE2)
	set(CPU_SUPPORTS_SVE2 TRUE)
	set(USE_ARMPL TRUE)
	message(STATUS "CPU supports SVE2")
else()
	message(STATUS "CPU does not support AVX512 or detection failed")
endif()

if(COMPILER_SUPPORTS_AVX512F AND CPU_SUPPORTS_AVX512)
    message(STATUS "Compiler and CPU supports -mavx512f. Enabling AVX512F.")
	add_definitions(-DUSE_AVX512)
elseif(COMPILER_SUPPORTS_AVX2 AND CPU_SUPPORTS_AVX2)
	message(STATUS "Compiler and CPU supports -mavx2. Enabling AVX2.")
	add_definitions(-DUSE_AVX2)
elseif(COMPILER_SUPPORTS_SVE2 AND CPU_SUPPORTS_SVE2)
	message(STATUS "Compiler and CPU supports armv8.2-a+sve2. Enabling SVE2.")
	add_definitions(-DUSE_SVE2)
	add_definitions(-DSVE_MAX_SUPPORT_BITS=128)
	add_compile_options(-march=armv8.2-a+sve2)
	if (USE_ARMPL)
		add_definitions(-DUSE_ARMPL)
		include_directories(/opt/arm/armpl_25.07_gcc/include)
		link_directories(/opt/arm/armpl_25.07_gcc/lib)
		link_libraries(/opt/arm/armpl_25.07_gcc/lib/libarmpl.so)
	endif()
else()
    message(STATUS "Compiler or CPU do not support -mavx512f, -mavx2 and sve2.")
endif()

#OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "No OpenMP support")
endif()

if (NOT USE_ARMPL)	
	find_package(BLAS)
	if(BLAS_FOUND)
		message(STATUS "BLAS found, BLAS_LIBRARIES = ${BLAS_LIBRARIES}")
		link_libraries(${BLAS_LIBRARIES})
	else()
		message(FATAL_ERROR "BLAS is required but was not found.")
	endif()
endif()

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(tests/utils)
# add_subdirectory(python)
