add_library(anari_library_photon SHARED)
set_target_properties(anari_library_photon PROPERTIES OUTPUT_NAME "anari_library_photon")

opencode_enable_warnings(anari_library_photon)

target_link_libraries(anari_library_photon PUBLIC Kokkos::kokkos PRIVATE anari::helium)

target_include_directories(anari_library_photon PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
)

target_compile_features(anari_library_photon PUBLIC cxx_std_20)

if(WIN32)
  target_compile_definitions(anari_library_photon PRIVATE _USE_MATH_DEFINES NOMINMAX)
endif()

# Object and parameter queries of the ANARI device, generated from the device
# definition by the code generator of the ANARI SDK.
anari_generate_queries(
  DEVICE_TARGET anari_library_photon
  CPP_NAMESPACE photon_anari
  JSON_DEFINITIONS_FILE ${CMAKE_CURRENT_SOURCE_DIR}/photon/anari/PhotonDefinitions.json
)

# --------------------------------------------------------------------------
# Photon source files — split into Kokkos-compiled and CXX-only groups.
#
# When Kokkos CUDA is enabled (COMPILE_AS_CMAKE_LANGUAGE mode), the files that
# include Kokkos headers are compiled as CUDA via nvcc + MSVC host compiler.
# The ANARI device (helium) MUST remain CXX because CUDA's float4 conflicts
# with linalg::float4 used by the ANARI SDK, and CXX files cannot include
# Kokkos headers of a CUDA-enabled Kokkos. photon/anari/RenderBridge.h is the
# interface between the two groups, it only uses standard C++ types.
# --------------------------------------------------------------------------

# Sources compiled as CUDA when the CUDA backend is enabled.
# nvcc uses MSVC as host compiler (-ccbin cl.exe) so C++ code compiles fine.
set(PHOTON_CUDA_SOURCES
  photon/pt/pathtracer.cpp
  photon/pt/bvh/bvh.cpp
  photon/pt/scene.cpp
  photon/pt/scene/builder.cpp
  photon/pt/environment_map.cpp
  photon/pt/denoiser.cpp
  photon/pt/backend/kokkos_backend.cpp
  photon/pt/backend/embree_backend.cpp
  photon/pt/backend/optix_backend.cpp
  photon/pt/backend/hiprt_backend.cpp
  photon/pt/backend/backend_factory.cpp
  photon/anari/RenderBridge.cpp
  photon/pt/version.cpp
)

if(PHOTON_PBRT_READY)
  list(APPEND PHOTON_CUDA_SOURCES
    photon/pbrt/pbrt_parser.cpp
    photon/pbrt/pbrt_to_photon.cpp
    photon/pbrt/ply_reader.cpp
    photon/pbrt/image_loader.cpp
  )
  target_include_directories(anari_library_photon PRIVATE ${PHOTON_STB_IMAGE_DIR})
  target_compile_definitions(anari_library_photon PUBLIC PHOTON_HAS_PBRT)
endif()

# The ANARI device: plain CXX, no Kokkos headers.
set(PHOTON_CXX_SOURCES
  photon/anari/Frame.cpp
  photon/anari/Geometry.cpp
  photon/anari/Object.cpp
  photon/anari/PhotonDevice.cpp
  photon/anari/PhotonLibrary.cpp
  photon/anari/SceneObjects.cpp
  photon/anari/World.cpp
)

if(OPENCODE_ENABLE_CUDA AND Kokkos_ENABLE_CUDA)
  # Compile the Kokkos sources as CUDA (nvcc with MSVC host compiler)
  set_source_files_properties(${PHOTON_CUDA_SOURCES}
    PROPERTIES LANGUAGE CUDA
  )
  # Extended lambda support required for KOKKOS_LAMBDA in device code
  set_source_files_properties(${PHOTON_CUDA_SOURCES}
    PROPERTIES COMPILE_OPTIONS "--expt-extended-lambda;--expt-relaxed-constexpr"
  )
  set_source_files_properties(${PHOTON_CXX_SOURCES}
    PROPERTIES LANGUAGE CXX
  )
endif()

target_sources(anari_library_photon
  PRIVATE
    ${PHOTON_CUDA_SOURCES}
    ${PHOTON_CXX_SOURCES}
    photon/anari/Frame.h
    photon/anari/Object.h
    photon/anari/PhotonDevice.h
    photon/anari/PhotonGlobalState.h
    photon/anari/RenderBridge.h
    photon/anari/SceneObjects.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/version.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/bvh/bvh.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/bvh/bvh_validate.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/camera.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/geom/mesh_intersector.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/geom/triangle_intersect.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/geom/triangle_mesh.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/material.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/disney_bsdf.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/math.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/tone_mapping.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/math_aabb.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/math_vec2.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/math_vec4.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/math_mat4.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/color.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/pathtracer.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/ray.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/rng.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/sampler.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/sampling.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/work_order.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/tf_volume.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/volume.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/light.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/environment_map.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/denoiser.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/texture.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/scene.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/scene/builder.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/sphere.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/backend/ray_backend.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/backend/kokkos_backend.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/backend/embree_backend.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/backend/optix_backend.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/backend/hiprt_backend.h
    ${PROJECT_SOURCE_DIR}/include/photon/pbrt/pbrt_scene.h
    ${PROJECT_SOURCE_DIR}/include/photon/pbrt/pbrt_parser.h
    ${PROJECT_SOURCE_DIR}/include/photon/pbrt/pbrt_to_photon.h
    ${PROJECT_SOURCE_DIR}/include/photon/pbrt/ply_reader.h
    ${PROJECT_SOURCE_DIR}/include/photon/pbrt/image_loader.h
)

if(PHOTON_EMBREE_READY)
  if(TARGET embree)
    target_link_libraries(anari_library_photon PUBLIC embree)
  else()
    target_include_directories(anari_library_photon PUBLIC ${EMBREE_INCLUDE_DIRS})
    target_link_libraries(anari_library_photon PUBLIC ${EMBREE_LIBRARIES})
  endif()
  target_compile_definitions(anari_library_photon PUBLIC PHOTON_HAS_EMBREE)
endif()

if(PHOTON_OPTIX_READY)
  target_include_directories(anari_library_photon PUBLIC ${PHOTON_OPTIX_INCLUDE_DIR})
  # Only link cuda_driver — Kokkos already links cudart (static).
  # Linking both CUDA::cudart (dynamic) and cudart_static causes LNK2005.
  target_link_libraries(anari_library_photon PUBLIC CUDA::cuda_driver)
  target_compile_definitions(anari_library_photon PUBLIC PHOTON_HAS_OPTIX)

  set(OPTIX_PTX_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/photon/pt/backend/optix/photon_optix.cu")
  set(OPTIX_PTX_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/photon_optix.ptx")
  set(OPTIX_PTX_EMBED  "${CMAKE_CURRENT_BINARY_DIR}/photon_optix_ptx.cpp")

  # CUDAToolkit_INCLUDE_DIRS is a list: CUDA 13 has two directories (include and
  # include/cccl), and each one needs its own -I.
  set(OPTIX_PTX_CUDA_INCLUDES "")
  foreach(_photon_cuda_include IN LISTS CUDAToolkit_INCLUDE_DIRS)
    list(APPEND OPTIX_PTX_CUDA_INCLUDES "-I${_photon_cuda_include}")
  endforeach()

  add_custom_command(
    OUTPUT ${OPTIX_PTX_OUTPUT}
    COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}
      -ptx
      -I${PHOTON_OPTIX_INCLUDE_DIR}
      ${OPTIX_PTX_CUDA_INCLUDES}
      --use_fast_math
      -O3
      ${OPTIX_PTX_SOURCE}
      -o ${OPTIX_PTX_OUTPUT}
    DEPENDS ${OPTIX_PTX_SOURCE}
    COMMENT "Compiling OptiX PTX kernel"
    VERBATIM
  )

  # The PTX is embedded as a byte array: MSVC limits the length of string
  # literals.
  set(OPTIX_EMBED_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/embed_ptx.py")
  file(WRITE ${OPTIX_EMBED_SCRIPT}
"import sys
ptx = open(sys.argv[1], 'rb').read()
with open(sys.argv[2], 'w') as f:
    f.write('namespace photon::pt {\\n')
    f.write('extern const char photon_optix_ptx[] = {\\n')
    for i in range(0, len(ptx), 20):
        f.write(''.join('%d,' % (b if b < 128 else b - 256) for b in ptx[i:i + 20]) + '\\n')
    f.write('0};\\n')
    f.write('extern const unsigned long long photon_optix_ptx_size = sizeof(photon_optix_ptx) - 1;\\n')
    f.write('}\\n')
")

  add_custom_command(
    OUTPUT ${OPTIX_PTX_EMBED}
    COMMAND ${Python3_EXECUTABLE} ${OPTIX_EMBED_SCRIPT} ${OPTIX_PTX_OUTPUT} ${OPTIX_PTX_EMBED}
    DEPENDS ${OPTIX_PTX_OUTPUT}
    COMMENT "Embedding OptiX PTX into C++ source"
    VERBATIM
  )

  target_sources(anari_library_photon PRIVATE ${OPTIX_PTX_EMBED})
  # Embedded PTX is pure C++ (a byte array), must not be compiled as CUDA
  if(OPENCODE_ENABLE_CUDA AND Kokkos_ENABLE_CUDA)
    set_source_files_properties(${OPTIX_PTX_EMBED} PROPERTIES LANGUAGE CXX)
  endif()
endif()

if(PHOTON_HIPRT_READY)
  target_link_libraries(anari_library_photon PUBLIC hiprt::hiprt)
  target_compile_definitions(anari_library_photon PUBLIC PHOTON_HAS_HIPRT)
endif()

if(PHOTON_OIDN_READY)
  target_compile_definitions(anari_library_photon PUBLIC PHOTON_HAS_OIDN)
  if(OpenImageDenoise_FOUND)
    target_link_libraries(anari_library_photon PUBLIC OpenImageDenoise)
  else()
    target_include_directories(anari_library_photon PUBLIC ${PHOTON_OIDN_INCLUDE})
    target_link_libraries(anari_library_photon PUBLIC ${PHOTON_OIDN_LIB})
  endif()
endif()

# --------------------------------------------------------------------------
# Installation: the device library and, on Windows, the DLLs it needs at
# run time go to the same directory.
# --------------------------------------------------------------------------
include(GNUInstallDirs)

install(TARGETS anari_library_photon
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(WIN32 AND PHOTON_RUNTIME_DLLS)
  install(FILES ${PHOTON_RUNTIME_DLLS} DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
