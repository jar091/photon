#pragma once

// helium
#include "helium/BaseGlobalDeviceState.h"
#include "helium/utility/TimeStamp.h"
// anari
#include <anari/anari_cpp/ext/linalg.h>

#include <atomic>

#include "photon/anari/RenderBridge.h"

namespace photon::anari_device {

using namespace anari::math;

struct PhotonGlobalState : public helium::BaseGlobalDeviceState
{
  PhotonGlobalState(ANARIDevice d) : helium::BaseGlobalDeviceState(d) {}

  // Path tracer of this device (Kokkos is started on first use).
  bridge::Renderer renderer;

  // Time of the last change of any object that is part of a world.
  std::atomic<helium::TimeStamp> sceneLastChanged{0};

  // World whose scene the renderer currently holds and when it was built.
  const void *renderedWorld{nullptr};
  helium::TimeStamp renderedWorldBuilt{0};

  // Frame that rendered last, the accumulated samples belong to it.
  const void *lastFrame{nullptr};

  void markSceneChanged()
  {
    sceneLastChanged = helium::newTimeStamp();
  }
};

#define PHOTON_ANARI_TYPEFOR_SPECIALIZATION(type, anari_type)                  \
  namespace anari {                                                            \
  ANARI_TYPEFOR_SPECIALIZATION(type, anari_type);                              \
  }

#define PHOTON_ANARI_TYPEFOR_DEFINITION(type)                                  \
  namespace anari {                                                            \
  ANARI_TYPEFOR_DEFINITION(type);                                              \
  }

} // namespace photon::anari_device
