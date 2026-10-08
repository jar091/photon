#pragma once

#include <Kokkos_Core.hpp>

#include "photon/pt/material.h"
#include "photon/pt/math.h"
#include "photon/pt/math_vec4.h"

namespace photon::pt {

// Attribute slots a material parameter or a sampler can read from.
constexpr i32 ATTRIBUTE_0 = 0;
constexpr i32 ATTRIBUTE_1 = 1;
constexpr i32 ATTRIBUTE_2 = 2;
constexpr i32 ATTRIBUTE_3 = 3;
constexpr i32 ATTRIBUTE_COLOR = 4;
constexpr i32 ATTRIBUTE_MESH_COUNT = 5; // slots stored with the mesh
constexpr i32 ATTRIBUTE_WORLD_POSITION = 5;
constexpr i32 ATTRIBUTE_WORLD_NORMAL = 6;
constexpr i32 ATTRIBUTE_COUNT = 7;

// Per-vertex attributes of the scene mesh (indexed like the positions). All
// attributes share one view, `offset[slot]` is the first element of a slot or
// -1 when no geometry provides it.
struct MeshAttributes
{
  Kokkos::View<Vec4 *> values;
  i32 offset[ATTRIBUTE_MESH_COUNT]{-1, -1, -1, -1, -1};

  KOKKOS_FUNCTION bool has(i32 slot) const
  {
    return slot >= 0 && slot < ATTRIBUTE_MESH_COUNT && offset[slot] >= 0;
  }
};

enum class SamplerType : u32 { Image1D = 0, Image2D = 1, Image3D = 2, Transform = 3 };
enum class WrapMode : u32 { ClampToEdge = 0, Repeat = 1, MirrorRepeat = 2 };

struct SamplerInfo
{
  SamplerType type{SamplerType::Image2D};
  i32 in_attribute{ATTRIBUTE_0};
  u32 linear{1};
  WrapMode wrap[3]{WrapMode::ClampToEdge, WrapMode::ClampToEdge, WrapMode::ClampToEdge};
  u32 offset{0}; // first texel in SamplerTable::texels
  u32 size[3]{1, 1, 1};
  // Row-major 4x4 matrices.
  f32 in_transform[16]{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  f32 in_offset[4]{0, 0, 0, 0};
  f32 out_transform[16]{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  f32 out_offset[4]{0, 0, 0, 0};
};

// Image and transform samplers of the scene. Texels are RGBA, images are stored
// row by row starting with the row at coordinate 0.
struct SamplerTable
{
  Kokkos::View<SamplerInfo *> infos;
  Kokkos::View<Vec4 *> texels;
  u32 count{0};

  KOKKOS_FUNCTION static i32 wrap_index(i32 i, i32 n, WrapMode mode)
  {
    if (mode == WrapMode::Repeat)
      return ((i % n) + n) % n;
    if (mode == WrapMode::MirrorRepeat) {
      const i32 j = ((i % (2 * n)) + 2 * n) % (2 * n);
      return j < n ? j : 2 * n - 1 - j;
    }
    return i < 0 ? 0 : (i >= n ? n - 1 : i);
  }

  KOKKOS_FUNCTION Vec4 sample(i32 id, const Vec4 &attribute) const
  {
    if (id < 0 || u32(id) >= count)
      return {0.f, 0.f, 0.f, 1.f};

    const SamplerInfo &si = infos(u32(id));
    Vec4 c = mul_mat4(si.in_transform, attribute);
    c.x += si.in_offset[0];
    c.y += si.in_offset[1];
    c.z += si.in_offset[2];
    c.w += si.in_offset[3];

    Vec4 result = c;

    if (si.type != SamplerType::Transform) {
      const int dims = si.type == SamplerType::Image1D ? 1 : (si.type == SamplerType::Image2D ? 2 : 3);
      const f32 coord[3] = {c.x, c.y, c.z};
      i32 i0[3] = {0, 0, 0};
      i32 i1[3] = {0, 0, 0};
      f32 frac[3] = {0.f, 0.f, 0.f};

      for (int d = 0; d < dims; ++d) {
        const i32 n = i32(si.size[d]);
        f32 x = coord[d];
        // Keep the texel computation finite for degenerate coordinates.
        if (!(x > -1e6f && x < 1e6f))
          x = 0.f;
        if (si.linear) {
          const f32 f = x * f32(n) - 0.5f;
          const f32 fl = Kokkos::floor(f);
          frac[d] = f - fl;
          i0[d] = wrap_index(i32(fl), n, si.wrap[d]);
          i1[d] = wrap_index(i32(fl) + 1, n, si.wrap[d]);
        } else {
          i0[d] = wrap_index(i32(Kokkos::floor(x * f32(n))), n, si.wrap[d]);
          i1[d] = i0[d];
        }
      }

      const u32 sx = si.size[0];
      const u32 sxy = si.size[0] * si.size[1];
      const u32 base = si.offset;
      Vec4 acc{0.f, 0.f, 0.f, 0.f};
      const int nz = dims > 2 ? 2 : 1;
      const int ny = dims > 1 ? 2 : 1;
      for (int kz = 0; kz < nz; ++kz) {
        const f32 wz = dims > 2 ? (kz ? frac[2] : 1.f - frac[2]) : 1.f;
        const u32 z = u32(kz ? i1[2] : i0[2]);
        for (int ky = 0; ky < ny; ++ky) {
          const f32 wy = dims > 1 ? (ky ? frac[1] : 1.f - frac[1]) : 1.f;
          const u32 y = u32(ky ? i1[1] : i0[1]);
          const Vec4 a = texels(base + z * sxy + y * sx + u32(i0[0]));
          const Vec4 b = texels(base + z * sxy + y * sx + u32(i1[0]));
          acc = acc + (a * (1.f - frac[0]) + b * frac[0]) * (wy * wz);
        }
      }
      result = acc;
    }

    Vec4 out = mul_mat4(si.out_transform, result);
    out.x += si.out_offset[0];
    out.y += si.out_offset[1];
    out.z += si.out_offset[2];
    out.w += si.out_offset[3];
    return out;
  }
};

}
