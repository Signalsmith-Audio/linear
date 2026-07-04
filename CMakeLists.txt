cmake_minimum_required(VERSION 3.24)

add_library(signalsmith-linear INTERFACE)
set_target_properties(signalsmith-linear PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/include)

# MSVC needs an extra flag to handle this many templates
if (MSVC)
	add_compile_options(/bigobj)
	#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
endif()

option(SIGNALSMITH_USE_ACCELERATE "use the Accelerate framework for Apple targets (OSX/iOS)" ON)
option(SIGNALSMITH_USE_IPP "use IPP (Intel Performance Primitives)" OFF)
option(SIGNALSMITH_USE_PFFFT "use PFFFT (Pretty Fast FFT)" OFF)
option(SIGNALSMITH_USE_PFFFT_DOUBLE "use PFFFT with double" OFF)
option(SIGNALSMITH_USE_XSIMD "use XSimd (version 8+)" OFF)
option(SIGNALSMITH_USE_XSIMD_DISPATCH "compile multiple architectures (x86/x86/ARM only) and choose at runtime" OFF)

if(APPLE AND SIGNALSMITH_USE_ACCELERATE)
	message(STATUS "Signalsmith Linear: using Accelerate")
	target_link_libraries(signalsmith-linear INTERFACE "-framework Accelerate")
	target_compile_definitions(signalsmith-linear INTERFACE SIGNALSMITH_USE_ACCELERATE ACCELERATE_NEW_LAPACK)
endif()
if(SIGNALSMITH_USE_IPP)
	find_package(IPP REQUIRED)
	message(STATUS "Signalsmith Linear: using IPP")
	target_link_libraries(signalsmith-linear INTERFACE IPP::ippcore IPP::ipps IPP::ippi)
	target_compile_definitions(signalsmith-linear INTERFACE SIGNALSMITH_USE_IPP)
endif()
if(SIGNALSMITH_USE_PFFFT)
	message(STATUS "Signalsmith Linear: using PFFFT (you need to link this yourself)")
	target_compile_definitions(signalsmith-linear INTERFACE SIGNALSMITH_USE_PFFFT)
endif()
if(SIGNALSMITH_USE_PFFFT_DOUBLE)
	message(STATUS "Signalsmith Linear: using PFFFT with double support")
	target_compile_definitions(signalsmith-linear INTERFACE SIGNALSMITH_USE_PFFFT_DOUBLE)
endif()
if(SIGNALSMITH_USE_CMSISDSP)
	message(STATUS "Signalsmith Linear: using CMSIS-DSP")
	FetchContent_Declare(
		CMSISDSP
		GIT_REPOSITORY https://github.com/ARM-software/CMSIS-DSP.git
		GIT_TAG v1.17.0
		GIT_SHALLOW ON
		OVERRIDE_FIND_PACKAGE # Use this if the package isn't found locally
	)
	find_package(CMSISDSP REQUIRED)

	target_link_libraries(signalsmith-linear INTERFACE CMSISDSP)
	target_compile_definitions(signalsmith-linear INTERFACE
		SIGNALSMITH_USE_CMSISDSP
	)
endif()
if(SIGNALSMITH_USE_XSIMD)
	message(STATUS "Signalsmith Linear: using XSimd")
	FetchContent_Declare(
		xsimd
		GIT_REPOSITORY https://github.com/xtensor-stack/xsimd.git
		GIT_TAG 14.0.0
		GIT_SHALLOW ON
		OVERRIDE_FIND_PACKAGE # Use this if the package isn't found locally
	)
	find_package(xsimd REQUIRED)

	target_link_libraries(signalsmith-linear INTERFACE xsimd)
	target_compile_definitions(signalsmith-linear INTERFACE
		SIGNALSMITH_USE_XSIMD
	)

	# This setting is trickier (and optional)
	# It compiles some of core functions for several different architectures
	# Then links them statically, and selects the best one at runtime
	if (SIGNALSMITH_USE_XSIMD_DISPATCH)
		target_compile_definitions(signalsmith-linear INTERFACE SIGNALSMITH_USE_XSIMD_DISPATCH)

		list(APPEND xsimd_arch_names
			sse2 sse4_2 avx avx2 avx512f neon neon64)
		list(APPEND arch_is_arm
			FALSE FALSE FALSE FALSE FALSE TRUE TRUE)
		list(APPEND gcc_arch_flags
			-msse2 -msse4.2 -mavx -mavx2 -mavx512f -mfpu=neon -mfpu=neon-vfpv4)
		# Pretty sure some of these don't exist, but we'll burn that bridge when we come to it
		list(APPEND msvc_arch_flags
			/arch:SSE2 /arch:SSE4.2 /arch:AVX /arch:AVX2 /arch:AVX512 /arch:NEON /arch:NEON64)

		foreach(xsimdName isArm gccFlag msvcFlag IN ZIP_LISTS xsimd_arch_names arch_is_arm gcc_arch_flags msvc_arch_flags)
			set(xsimdTargetName "signalsmith-linear_xsimd_${xsimdName}")
			add_library(${xsimdTargetName} STATIC ${CMAKE_CURRENT_SOURCE_DIR}/platform/linear-xsimd-dispatch.cpp)
			message(STATUS "Signalsmith Linear: creating arch-specific target ${xsimdTargetName}")
			target_link_libraries(${xsimdTargetName} PUBLIC xsimd)
			target_compile_definitions(${xsimdTargetName} PRIVATE
				SIGNALSMITH_USE_XSIMD
				SIGNALSMITH_USE_XSIMD_DISPATCH
				SIGNALSMITH_LINEAR_XSIMD_DISPATCH_ARCH=xsimd::${xsimdName}
			)

			if (MSVC)
				target_compile_options(${xsimdTargetName} PRIVATE ${msvcFlag} /O2) # /fp:fast)
			elseif ((CMAKE_CXX_COMPILER_ID STREQUAL "GNU") OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
				target_compile_options(${xsimdTargetName} PRIVATE ${gccFlag} -O3 -ffast-math)
			elseif (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
				# Dual-architecture builds are a pain for this
				if (isArm)
					target_compile_options(${xsimdTargetName} PRIVATE "SHELL:-Xarch_aarch64 ${gccFlag}" -O3 -ffast-math)
				else()
					target_compile_options(${xsimdTargetName} PRIVATE "SHELL:-Xarch_x86_64 ${gccFlag}" -O3)
				endif()
			else()
				message(FATAL_ERROR "SIGNALSMITH_USE_XSIMD_DISPATCH : unknown compiler (not MSVC, GCC or [Apple]Clang) : ${CMAKE_CXX_COMPILER_ID}")
			endif()

			set_target_properties(${xsimdTargetName} PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
			target_link_libraries(signalsmith-linear INTERFACE ${xsimdTargetName})
		endforeach()
	endif()
endif()
	