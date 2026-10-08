#include "photon/anari/PhotonDevice.h"

#include "photon/anari/Frame.h"
#include "photon/anari/SceneObjects.h"
#include "photon/pt/version.h"

// std
#include <cstring>
#include <limits>
#include <string>

#include "anari_library_photon_queries.h"

namespace photon::anari_device {

namespace {

std::string_view subtypeView(const char *subtype)
{
  return subtype ? std::string_view(subtype) : std::string_view();
}

} // namespace

// API Objects ////////////////////////////////////////////////////////////////

ANARIArray1D PhotonDevice::newArray1D(
    const void *appMemory, ANARIMemoryDeleter deleter, const void *userData, ANARIDataType type, uint64_t numItems)
{
  helium::Array1DMemoryDescriptor md;
  md.appMemory = appMemory;
  md.deleter = deleter;
  md.deleterPtr = userData;
  md.elementType = type;
  md.numItems = numItems;

  if (anari::isObject(type))
    return (ANARIArray1D) new ObjectArray(deviceState(), md);
  else
    return (ANARIArray1D) new Array1D(deviceState(), md);
}

ANARIArray2D PhotonDevice::newArray2D(const void *appMemory,
    ANARIMemoryDeleter deleter,
    const void *userData,
    ANARIDataType type,
    uint64_t numItems1,
    uint64_t numItems2)
{
  helium::Array2DMemoryDescriptor md;
  md.appMemory = appMemory;
  md.deleter = deleter;
  md.deleterPtr = userData;
  md.elementType = type;
  md.numItems1 = numItems1;
  md.numItems2 = numItems2;

  return (ANARIArray2D) new Array2D(deviceState(), md);
}

ANARIArray3D PhotonDevice::newArray3D(const void *appMemory,
    ANARIMemoryDeleter deleter,
    const void *userData,
    ANARIDataType type,
    uint64_t numItems1,
    uint64_t numItems2,
    uint64_t numItems3)
{
  helium::Array3DMemoryDescriptor md;
  md.appMemory = appMemory;
  md.deleter = deleter;
  md.deleterPtr = userData;
  md.elementType = type;
  md.numItems1 = numItems1;
  md.numItems2 = numItems2;
  md.numItems3 = numItems3;

  return (ANARIArray3D) new Array3D(deviceState(), md);
}

ANARICamera PhotonDevice::newCamera(const char *subtype)
{
  return (ANARICamera) new Camera(deviceState(), subtypeView(subtype));
}

ANARIFrame PhotonDevice::newFrame()
{
  return (ANARIFrame) new Frame(deviceState());
}

ANARIGeometry PhotonDevice::newGeometry(const char *subtype)
{
  return (ANARIGeometry) new Geometry(deviceState(), subtypeView(subtype));
}

ANARIGroup PhotonDevice::newGroup()
{
  return (ANARIGroup) new Group(deviceState());
}

ANARIInstance PhotonDevice::newInstance(const char *subtype)
{
  return (ANARIInstance) new Instance(deviceState(), subtypeView(subtype));
}

ANARILight PhotonDevice::newLight(const char *subtype)
{
  return (ANARILight) new Light(deviceState(), subtypeView(subtype));
}

ANARIMaterial PhotonDevice::newMaterial(const char *subtype)
{
  return (ANARIMaterial) new Material(deviceState(), subtypeView(subtype));
}

ANARIRenderer PhotonDevice::newRenderer(const char *subtype)
{
  return (ANARIRenderer) new Renderer(deviceState(), subtypeView(subtype));
}

ANARISampler PhotonDevice::newSampler(const char *subtype)
{
  return (ANARISampler) new Sampler(deviceState(), subtypeView(subtype));
}

ANARISpatialField PhotonDevice::newSpatialField(const char *subtype)
{
  return (ANARISpatialField) new SpatialField(deviceState(), subtypeView(subtype));
}

ANARISurface PhotonDevice::newSurface()
{
  return (ANARISurface) new Surface(deviceState());
}

ANARIVolume PhotonDevice::newVolume(const char *subtype)
{
  return (ANARIVolume) new Volume(deviceState(), subtypeView(subtype));
}

ANARIWorld PhotonDevice::newWorld()
{
  return (ANARIWorld) new World(deviceState());
}

// Query functions ////////////////////////////////////////////////////////////

const char **PhotonDevice::getObjectSubtypes(ANARIDataType objectType)
{
  return photon_anari::query_object_types(objectType);
}

const void *PhotonDevice::getObjectInfo(
    ANARIDataType objectType, const char *objectSubtype, const char *infoName, ANARIDataType infoType)
{
  return photon_anari::query_object_info(objectType, objectSubtype, infoName, infoType);
}

const void *PhotonDevice::getParameterInfo(ANARIDataType objectType,
    const char *objectSubtype,
    const char *parameterName,
    ANARIDataType parameterType,
    const char *infoName,
    ANARIDataType infoType)
{
  return photon_anari::query_param_info(
      objectType, objectSubtype, parameterName, parameterType, infoName, infoType);
}

// Other PhotonDevice definitions /////////////////////////////////////////////

PhotonDevice::PhotonDevice(ANARIStatusCallback cb, const void *ptr) : helium::BaseDevice(cb, ptr)
{
  m_state = std::make_unique<PhotonGlobalState>(this_device());
  deviceCommitParameters();
}

PhotonDevice::PhotonDevice(ANARILibrary l) : helium::BaseDevice(l)
{
  m_state = std::make_unique<PhotonGlobalState>(this_device());
  deviceCommitParameters();
}

PhotonDevice::~PhotonDevice()
{
  auto &state = *deviceState();
  state.commitBuffer.clear();
  reportMessage(ANARI_SEVERITY_DEBUG, "destroying photon device (%p)", this);
}

int PhotonDevice::deviceGetProperty(const char *name, ANARIDataType type, void *mem, uint64_t size, uint32_t)
{
  static const std::string version_name = std::string("photon ") + photon::version_string();
  std::string_view prop = name;
  if (prop == "extension" && type == ANARI_STRING_LIST) {
    helium::writeToVoidP(mem, photon_anari::query_extensions());
    return 1;
  } else if (prop == "version" && type == ANARI_INT32) {
    helium::writeToVoidP(mem, 1);
    return 1;
  } else if (prop == "version.name.size" && type == ANARI_UINT64) {
    helium::writeToVoidP(mem, uint64_t(version_name.size() + 1));
    return 1;
  } else if (prop == "version.name" && type == ANARI_STRING) {
    std::memset(mem, 0, size);
    std::memcpy(mem, version_name.data(), std::min(uint64_t(version_name.size()), size - 1));
    return 1;
  } else if (prop == "anariVersion.major" && type == ANARI_INT32) {
    helium::writeToVoidP(mem, 1);
    return 1;
  } else if (prop == "anariVersion.minor" && type == ANARI_INT32) {
    helium::writeToVoidP(mem, 1);
    return 1;
  } else if (prop == "geometryMaxIndex" && type == ANARI_UINT64) {
    helium::writeToVoidP(mem, uint64_t(std::numeric_limits<uint32_t>::max()));
    return 1;
  } else if (prop == "photon" && type == ANARI_BOOL) {
    helium::writeToVoidP(mem, true);
    return 1;
  }
  return 0;
}

PhotonGlobalState *PhotonDevice::deviceState() const
{
  return (PhotonGlobalState *)helium::BaseDevice::m_state.get();
}

} // namespace photon::anari_device
