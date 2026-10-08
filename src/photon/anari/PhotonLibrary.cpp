#include "photon/anari/PhotonDevice.h"

#include <anari/backend/LibraryImpl.h>

#ifdef _WIN32
#define PHOTON_EXPORT __declspec(dllexport)
#else
#define PHOTON_EXPORT __attribute__((visibility("default")))
#endif

namespace photon_anari {

const char **query_extensions();

} // namespace photon_anari

namespace photon::anari_device {

struct PhotonLibrary : public anari::LibraryImpl
{
  PhotonLibrary(void *lib, ANARIStatusCallback defaultStatusCB, const void *statusCBPtr)
      : anari::LibraryImpl(lib, defaultStatusCB, statusCBPtr)
  {}

  // Every device subtype is the default device.
  ANARIDevice newDevice(const char * /*subtype*/) override
  {
    return (ANARIDevice) new PhotonDevice(this_library());
  }

  const char **getDeviceSubtypes() override
  {
    static const char *subtypes[] = {"default", nullptr};
    return subtypes;
  }

  const char **getDeviceExtensions(const char * /*deviceType*/) override
  {
    return photon_anari::query_extensions();
  }
};

} // namespace photon::anari_device

// Define library entrypoint //////////////////////////////////////////////////

extern "C" PHOTON_EXPORT ANARI_DEFINE_LIBRARY_ENTRYPOINT(photon, handle, scb, scbPtr)
{
  return (ANARILibrary) new photon::anari_device::PhotonLibrary(handle, scb, scbPtr);
}

extern "C" PHOTON_EXPORT ANARIDevice anariNewPhotonDevice(ANARIStatusCallback defaultCallback, const void *userPtr)
{
  return (ANARIDevice) new photon::anari_device::PhotonDevice(defaultCallback, userPtr);
}
