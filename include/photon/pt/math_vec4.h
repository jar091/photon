#pragma once

#include <Kokkos_Core.hpp>

#include "photon/pt/math.h"

namespace photon::pt {

struct Vec4
{
  f32 x{0.f};
  f32 y{0.f};
  f32 z{0.f};
  f32 w{0.f};

  KOKKOS_FUNCTION Vec4() = default;
  KOKKOS_FUNCTION Vec4(f32 xx, f32 yy, f32 zz, f32 ww) : x(xx), y(yy), z(zz), w(ww) {}
  KOKKOS_FUNCTION Vec4(const Vec3 &v, f32 ww) : x(v.x), y(v.y), z(v.z), w(ww) {}

  KOKKOS_FUNCTION Vec3 xyz() const
  {
    return {x, y, z};
  }
};

KOKKOS_FUNCTION inline Vec4 operator+(const Vec4 &a, const Vec4 &b)
{
  return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

KOKKOS_FUNCTION inline Vec4 operator*(const Vec4 &a, f32 s)
{
  return {a.x * s, a.y * s, a.z * s, a.w * s};
}

// Row-major 4x4 matrix times column vector.
KOKKOS_FUNCTION inline Vec4 mul_mat4(const f32 *m, const Vec4 &v)
{
  return {m[0] * v.x + m[1] * v.y + m[2] * v.z + m[3] * v.w,
      m[4] * v.x + m[5] * v.y + m[6] * v.z + m[7] * v.w,
      m[8] * v.x + m[9] * v.y + m[10] * v.z + m[11] * v.w,
      m[12] * v.x + m[13] * v.y + m[14] * v.z + m[15] * v.w};
}

} // namespace photon::pt
