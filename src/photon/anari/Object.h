#pragma once

#include "photon/anari/PhotonGlobalState.h"
// helium
#include "helium/BaseObject.h"
#include "helium/array/Array1D.h"
#include "helium/array/Array2D.h"
#include "helium/array/Array3D.h"
#include "helium/array/ObjectArray.h"
#include "helium/utility/ChangeObserverPtr.h"
// std
#include <string>
#include <string_view>

namespace photon::anari_device {

using Array1D = helium::Array1D;
using Array2D = helium::Array2D;
using Array3D = helium::Array3D;
using ObjectArray = helium::ObjectArray;

struct Object : public helium::BaseObject
{
  Object(ANARIDataType type, PhotonGlobalState *s);
  ~Object() override = default;

  bool getProperty(const std::string_view &name,
      ANARIDataType type,
      void *ptr,
      uint64_t size,
      uint32_t flags) override;

  void commitParameters() override;
  void finalize() override;

  bool isValid() const override;

  PhotonGlobalState *deviceState() const;
};

// Stands in for objects of a type the device does not know at all.
struct UnknownObject : public Object
{
  UnknownObject(ANARIDataType type, PhotonGlobalState *s);
  bool isValid() const override;
};

// Element `i` of an array as a color: missing components are (0, 0, 0, 1),
// fixed point types are normalized and sRGB types are converted to linear.
float4 arrayElementAsFloat4(const helium::Array &array, size_t i);

// Component `c` of element `i` of an index array (UINT32 or UINT64 based).
uint64_t arrayIndexComponent(const helium::Array &array, size_t i, int c);

} // namespace photon::anari_device

PHOTON_ANARI_TYPEFOR_SPECIALIZATION(photon::anari_device::Object *, ANARI_OBJECT);
