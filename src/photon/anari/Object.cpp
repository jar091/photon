#include "photon/anari/Object.h"

#include <cmath>
#include <cstring>

namespace photon::anari_device {

// Object definitions /////////////////////////////////////////////////////////

Object::Object(ANARIDataType type, PhotonGlobalState *s) : helium::BaseObject(type, s)
{
  // Objects have to be valid without any parameter set, and helium does not
  // commit objects whose parameters never changed.
  markParameterChanged();
}

void Object::commitParameters()
{
  // no-op
}

void Object::finalize()
{
  // no-op
}

bool Object::getProperty(const std::string_view &name, ANARIDataType type, void *ptr, uint64_t, uint32_t)
{
  if (name == "valid" && type == ANARI_BOOL) {
    helium::writeToVoidP(ptr, isValid());
    return true;
  }

  return false;
}

bool Object::isValid() const
{
  return true;
}

PhotonGlobalState *Object::deviceState() const
{
  return (PhotonGlobalState *)helium::BaseObject::m_state;
}

// UnknownObject definitions //////////////////////////////////////////////////

UnknownObject::UnknownObject(ANARIDataType type, PhotonGlobalState *s) : Object(type, s) {}

bool UnknownObject::isValid() const
{
  return false;
}

// Array helpers //////////////////////////////////////////////////////////////

namespace {

float srgbToLinear(float s)
{
  return s <= 0.04045f ? s / 12.92f : std::pow((s + 0.055f) / 1.055f, 2.4f);
}

template <typename T>
float4 readComponents(const void *ptr, int components, float scale)
{
  float4 out(0.f, 0.f, 0.f, 1.f);
  T v[4];
  std::memcpy(v, ptr, sizeof(T) * size_t(components));
  for (int c = 0; c < components; ++c)
    out[c] = float(v[c]) * scale;
  return out;
}

} // namespace

float4 arrayElementAsFloat4(const helium::Array &array, size_t i)
{
  const ANARIDataType type = array.elementType();
  const int components = int(anari::componentsOf(type));
  const void *ptr = static_cast<const uint8_t *>(array.data()) + i * anari::sizeOf(type);
  if (components < 1 || components > 4)
    return float4(0.f, 0.f, 0.f, 1.f);

  switch (type) {
  case ANARI_FLOAT32:
  case ANARI_FLOAT32_VEC2:
  case ANARI_FLOAT32_VEC3:
  case ANARI_FLOAT32_VEC4:
    return readComponents<float>(ptr, components, 1.f);
  case ANARI_FLOAT64:
  case ANARI_FLOAT64_VEC2:
  case ANARI_FLOAT64_VEC3:
  case ANARI_FLOAT64_VEC4:
    return readComponents<double>(ptr, components, 1.f);
  case ANARI_UFIXED8:
  case ANARI_UFIXED8_VEC2:
  case ANARI_UFIXED8_VEC3:
  case ANARI_UFIXED8_VEC4:
    return readComponents<uint8_t>(ptr, components, 1.f / 255.f);
  case ANARI_UFIXED8_R_SRGB:
  case ANARI_UFIXED8_RA_SRGB:
  case ANARI_UFIXED8_RGB_SRGB:
  case ANARI_UFIXED8_RGBA_SRGB: {
    float4 c = readComponents<uint8_t>(ptr, components, 1.f / 255.f);
    // The last component of RA and RGBA is a linear alpha.
    const int colors = (components == 2 || components == 4) ? components - 1 : components;
    for (int k = 0; k < colors; ++k)
      c[k] = srgbToLinear(c[k]);
    if (components == 2) {
      c.w = c.y;
      c.y = 0.f;
    }
    return c;
  }
  case ANARI_UFIXED16:
  case ANARI_UFIXED16_VEC2:
  case ANARI_UFIXED16_VEC3:
  case ANARI_UFIXED16_VEC4:
    return readComponents<uint16_t>(ptr, components, 1.f / 65535.f);
  case ANARI_UFIXED32:
  case ANARI_UFIXED32_VEC2:
  case ANARI_UFIXED32_VEC3:
  case ANARI_UFIXED32_VEC4:
    return readComponents<uint32_t>(ptr, components, 1.f / 4294967295.f);
  case ANARI_FIXED8:
  case ANARI_FIXED8_VEC2:
  case ANARI_FIXED8_VEC3:
  case ANARI_FIXED8_VEC4:
    return readComponents<int8_t>(ptr, components, 1.f / 127.f);
  case ANARI_FIXED16:
  case ANARI_FIXED16_VEC2:
  case ANARI_FIXED16_VEC3:
  case ANARI_FIXED16_VEC4:
    return readComponents<int16_t>(ptr, components, 1.f / 32767.f);
  case ANARI_FIXED32:
  case ANARI_FIXED32_VEC2:
  case ANARI_FIXED32_VEC3:
  case ANARI_FIXED32_VEC4:
    return readComponents<int32_t>(ptr, components, 1.f / 2147483647.f);
  case ANARI_UINT8:
  case ANARI_UINT8_VEC2:
  case ANARI_UINT8_VEC3:
  case ANARI_UINT8_VEC4:
    return readComponents<uint8_t>(ptr, components, 1.f);
  case ANARI_INT8:
  case ANARI_INT8_VEC2:
  case ANARI_INT8_VEC3:
  case ANARI_INT8_VEC4:
    return readComponents<int8_t>(ptr, components, 1.f);
  case ANARI_UINT16:
  case ANARI_UINT16_VEC2:
  case ANARI_UINT16_VEC3:
  case ANARI_UINT16_VEC4:
    return readComponents<uint16_t>(ptr, components, 1.f);
  case ANARI_INT16:
  case ANARI_INT16_VEC2:
  case ANARI_INT16_VEC3:
  case ANARI_INT16_VEC4:
    return readComponents<int16_t>(ptr, components, 1.f);
  case ANARI_UINT32:
  case ANARI_UINT32_VEC2:
  case ANARI_UINT32_VEC3:
  case ANARI_UINT32_VEC4:
    return readComponents<uint32_t>(ptr, components, 1.f);
  case ANARI_INT32:
  case ANARI_INT32_VEC2:
  case ANARI_INT32_VEC3:
  case ANARI_INT32_VEC4:
    return readComponents<int32_t>(ptr, components, 1.f);
  default:
    break;
  }
  return float4(0.f, 0.f, 0.f, 1.f);
}

uint64_t arrayIndexComponent(const helium::Array &array, size_t i, int c)
{
  const ANARIDataType type = array.elementType();
  const size_t components = anari::componentsOf(type);
  const uint8_t *ptr = static_cast<const uint8_t *>(array.data()) + i * anari::sizeOf(type);
  switch (type) {
  case ANARI_UINT64:
  case ANARI_UINT64_VEC2:
  case ANARI_UINT64_VEC3:
  case ANARI_UINT64_VEC4:
  case ANARI_INT64:
  case ANARI_INT64_VEC2:
  case ANARI_INT64_VEC3:
  case ANARI_INT64_VEC4: {
    uint64_t v = 0;
    if (size_t(c) < components)
      std::memcpy(&v, ptr + sizeof(uint64_t) * size_t(c), sizeof(v));
    return v;
  }
  default: {
    uint32_t v = 0;
    if (size_t(c) < components && anari::sizeOf(type) == components * sizeof(uint32_t))
      std::memcpy(&v, ptr + sizeof(uint32_t) * size_t(c), sizeof(v));
    return v;
  }
  }
}

} // namespace photon::anari_device

PHOTON_ANARI_TYPEFOR_DEFINITION(photon::anari_device::Object *);
