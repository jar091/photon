# When Kokkos CUDA is enabled, app targets compiled as plain CXX cannot
# include Kokkos headers (they require nvcc). These apps are optional
# standalone tools — tsdViewer is built separately from VisRTX.
if(OPENCODE_ENABLE_CUDA AND Kokkos_ENABLE_CUDA)
  message(STATUS "Skipping app/ targets (not compatible with Kokkos CUDA CXX propagation)")
  return()
endif()

add_executable(photon_app main.cpp)

target_link_libraries(photon_app PRIVATE anari_library_photon)


add_executable(photon_anari_smoke anari_smoke.cpp)

target_link_libraries(photon_anari_smoke PRIVATE anari::anari)


add_executable(photon_render photon_render.cpp)

target_link_libraries(photon_render PRIVATE anari_library_photon)

add_executable(photon_debug_ray photon_debug_ray.cpp)

target_link_libraries(photon_debug_ray PRIVATE anari_library_photon)

add_executable(photon_anari_render photon_anari_render.cpp)

target_link_libraries(photon_anari_render PRIVATE anari_library_photon anari::anari)


if(PHOTON_PBRT_READY)
  add_executable(photon_pbrt_render photon_pbrt_render.cpp)
  target_link_libraries(photon_pbrt_render PRIVATE anari_library_photon)

  add_executable(photon_bdpt_render photon_bdpt_render.cpp)
  target_link_libraries(photon_bdpt_render PRIVATE anari_library_photon)
endif()

add_executable(photon_volume_render photon_volume_render.cpp)

target_link_libraries(photon_volume_render PRIVATE anari_library_photon)

if(PHOTON_MPI_READY)
  add_executable(photon_mpi_render photon_mpi_render.cpp)
  target_link_libraries(photon_mpi_render PRIVATE anari_library_photon MPI::MPI_CXX)

  add_executable(photon_scivis_mpi photon_scivis_mpi.cpp)
  target_link_libraries(photon_scivis_mpi PRIVATE anari_library_photon MPI::MPI_CXX)

  if(PHOTON_ICET_READY)
    target_link_libraries(photon_mpi_render PRIVATE IceTCore IceTMPI)
    target_include_directories(photon_mpi_render PRIVATE ${icet_SOURCE_DIR}/src/include ${icet_BINARY_DIR}/src/include)
    target_compile_definitions(photon_mpi_render PRIVATE PHOTON_HAS_ICET)

    target_link_libraries(photon_scivis_mpi PRIVATE IceTCore IceTMPI)
    target_include_directories(photon_scivis_mpi PRIVATE ${icet_SOURCE_DIR}/src/include ${icet_BINARY_DIR}/src/include)
    target_compile_definitions(photon_scivis_mpi PRIVATE PHOTON_HAS_ICET)
  endif()
endif()

add_executable(photon_scene_render photon_scene_render.cpp)

target_link_libraries(photon_scene_render PRIVATE anari_library_photon anari::anari anari::anari_test_scenes anari::helium)

