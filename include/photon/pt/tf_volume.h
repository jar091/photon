#pragma once

#include <Kokkos_Core.hpp>

#include "photon/pt/math.h"
#include "photon/pt/math_vec4.h"
#include "photon/pt/ray.h"
#include "photon/pt/rng.h"

namespace photon::pt {

// Scalar field on a regular grid (values at the grid points) which is mapped to
// color and extinction by a transfer function table.
struct TfVolume
{
  // World to local (grid object) space, row-major 3x4.
  f32 world_to_local[12]{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
  Vec3 origin{0.f, 0.f, 0.f};
  Vec3 spacing{1.f, 1.f, 1.f};
  u32 dims[3]{1, 1, 1};
  u32 data_offset{0}; // first value in VolumeSet::values, x runs fastest
  u32 tf_offset{0};   // first entry in VolumeSet::tf
  u32 tf_size{0};     // number of table entries, they span [value_lo, value_hi]
  f32 value_lo{0.f};
  f32 value_hi{1.f};
  f32 density_scale{1.f}; // extinction = table alpha * density_scale
  f32 majorant{0.f};      // upper bound of the extinction
  u32 linear{1};
  // Coarse grid over the volume with an upper bound of the extinction per
  // cell (VolumeSet::majorants, x runs fastest): empty space is skipped.
  u32 macro_offset{0};
  u32 macro_dims[3]{1, 1, 1};
};

struct VolumeSet
{
  Kokkos::View<TfVolume *> volumes;
  Kokkos::View<f32 *> values;
  Kokkos::View<Vec4 *> tf;
  Kokkos::View<f32 *> majorants;
  u32 count{0};

  KOKKOS_FUNCTION static Vec3 to_local_point(const TfVolume &v, const Vec3 &p)
  {
    const f32 *m = v.world_to_local;
    return {m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3], m[4] * p.x + m[5] * p.y + m[6] * p.z + m[7],
        m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11]};
  }

  KOKKOS_FUNCTION static Vec3 to_local_vector(const TfVolume &v, const Vec3 &d)
  {
    const f32 *m = v.world_to_local;
    return {m[0] * d.x + m[1] * d.y + m[2] * d.z, m[4] * d.x + m[5] * d.y + m[6] * d.z,
        m[8] * d.x + m[9] * d.y + m[10] * d.z};
  }

  // Clips [t0, t1] of the local space ray to the grid bounds.
  KOKKOS_FUNCTION static bool clip_to_bounds(const TfVolume &v, const Vec3 &org, const Vec3 &dir, f32 &t0, f32 &t1)
  {
    const f32 o[3] = {org.x, org.y, org.z};
    const f32 d[3] = {dir.x, dir.y, dir.z};
    const f32 lo[3] = {v.origin.x, v.origin.y, v.origin.z};
    const f32 sp[3] = {v.spacing.x, v.spacing.y, v.spacing.z};
    for (int a = 0; a < 3; ++a) {
      const f32 hi = lo[a] + sp[a] * f32(v.dims[a] > 1 ? v.dims[a] - 1 : 1);
      if (d[a] == 0.f) {
        if (o[a] < lo[a] || o[a] > hi)
          return false;
        continue;
      }
      const f32 inv = 1.f / d[a];
      const f32 ta = (lo[a] - o[a]) * inv;
      const f32 tb = (hi - o[a]) * inv;
      const f32 tn = ta < tb ? ta : tb;
      const f32 tf = ta < tb ? tb : ta;
      t0 = tn > t0 ? tn : t0;
      t1 = tf < t1 ? tf : t1;
      if (t1 <= t0)
        return false;
    }
    return true;
  }

  KOKKOS_FUNCTION f32 field_value(const TfVolume &v, const Vec3 &p_local) const
  {
    const f32 g[3] = {(p_local.x - v.origin.x) / v.spacing.x, (p_local.y - v.origin.y) / v.spacing.y,
        (p_local.z - v.origin.z) / v.spacing.z};
    int i0[3], i1[3];
    f32 fr[3];
    for (int a = 0; a < 3; ++a) {
      const int n = int(v.dims[a]);
      f32 x = g[a];
      x = x < 0.f ? 0.f : (x > f32(n - 1) ? f32(n - 1) : x);
      if (v.linear) {
        int i = int(x);
        if (i > n - 2)
          i = n - 2;
        if (i < 0)
          i = 0;
        i0[a] = i;
        i1[a] = i + 1 < n ? i + 1 : i;
        fr[a] = x - f32(i);
      } else {
        int i = int(x + 0.5f);
        i = i > n - 1 ? n - 1 : i;
        i0[a] = i1[a] = i;
        fr[a] = 0.f;
      }
    }
    const u32 sx = v.dims[0];
    const u32 sxy = v.dims[0] * v.dims[1];
    const u32 base = v.data_offset;
    auto at = [&](int x, int y, int z) { return values(base + u32(z) * sxy + u32(y) * sx + u32(x)); };
    if (!v.linear)
      return at(i0[0], i0[1], i0[2]);
    const f32 c00 = at(i0[0], i0[1], i0[2]) * (1.f - fr[0]) + at(i1[0], i0[1], i0[2]) * fr[0];
    const f32 c10 = at(i0[0], i1[1], i0[2]) * (1.f - fr[0]) + at(i1[0], i1[1], i0[2]) * fr[0];
    const f32 c01 = at(i0[0], i0[1], i1[2]) * (1.f - fr[0]) + at(i1[0], i0[1], i1[2]) * fr[0];
    const f32 c11 = at(i0[0], i1[1], i1[2]) * (1.f - fr[0]) + at(i1[0], i1[1], i1[2]) * fr[0];
    const f32 c0 = c00 * (1.f - fr[1]) + c10 * fr[1];
    const f32 c1 = c01 * (1.f - fr[1]) + c11 * fr[1];
    return c0 * (1.f - fr[2]) + c1 * fr[2];
  }

  // Color (rgb) and extinction (w) at a local space point.
  KOKKOS_FUNCTION Vec4 classify(const TfVolume &v, const Vec3 &p_local) const
  {
    const f32 value = field_value(v, p_local);
    const f32 range = v.value_hi - v.value_lo;
    f32 x = range != 0.f ? (value - v.value_lo) / range : 0.f;
    x = x < 0.f ? 0.f : (x > 1.f ? 1.f : x);
    // The first and the last table entry sit exactly on the ends of the range.
    const f32 f = x * f32(v.tf_size - 1);
    u32 i = u32(f);
    if (i > v.tf_size - 2)
      i = v.tf_size >= 2 ? v.tf_size - 2 : 0;
    const f32 fr = v.tf_size >= 2 ? f - f32(i) : 0.f;
    const Vec4 a = tf(v.tf_offset + i);
    const Vec4 b = tf(v.tf_offset + (v.tf_size >= 2 ? i + 1 : i));
    Vec4 c = a * (1.f - fr) + b * fr;
    c.w = c.w * v.density_scale;
    return c;
  }

  // Tracks the local space ray through the macro cells of a volume, using the
  // extinction bound of each cell. With `scatter` it returns the distance of
  // the first scattering event in (t0, t1) or a negative value (delta tracking)
  // and sets `albedo`; otherwise it multiplies `transmittance` by an estimate
  // of the transmittance of the segment (ratio tracking) and returns -1.
  KOKKOS_FUNCTION f32 track(const TfVolume &v, const Vec3 &org, const Vec3 &dir, f32 t0, f32 t1, Rng &rng,
      bool scatter, Vec3 &albedo, f32 &transmittance) const
  {
    const f32 o[3] = {org.x, org.y, org.z};
    const f32 d[3] = {dir.x, dir.y, dir.z};
    const f32 lo[3] = {v.origin.x, v.origin.y, v.origin.z};
    const f32 sp[3] = {v.spacing.x, v.spacing.y, v.spacing.z};

    // Macro cell of the entry point and the distances to the next cell borders.
    int cell[3];
    int step[3];
    f32 t_next[3];
    f32 t_delta[3];
    for (int a = 0; a < 3; ++a) {
      const int m = int(v.macro_dims[a]);
      const f32 extent = sp[a] * f32(v.dims[a] > 1 ? v.dims[a] - 1 : 1);
      const f32 cells_per_unit = f32(m) / extent;
      const f32 g = (o[a] + d[a] * t0 - lo[a]) * cells_per_unit;
      int c = int(g);
      c = c < 0 ? 0 : (c > m - 1 ? m - 1 : c);
      cell[a] = c;
      const f32 speed = d[a] * cells_per_unit; // cells per unit of t
      if (speed > 0.f) {
        step[a] = 1;
        t_delta[a] = 1.f / speed;
        t_next[a] = t0 + (f32(c + 1) - g) * t_delta[a];
      } else if (speed < 0.f) {
        step[a] = -1;
        t_delta[a] = -1.f / speed;
        t_next[a] = t0 + (g - f32(c)) * t_delta[a];
      } else {
        step[a] = 0;
        t_delta[a] = 1e30f;
        t_next[a] = 1e30f;
      }
    }

    f32 t = t0;
    int lookups = 0;
    for (int visited = 0; visited < 4096; ++visited) {
      const int axis = t_next[0] < t_next[1] ? (t_next[0] < t_next[2] ? 0 : 2) : (t_next[1] < t_next[2] ? 1 : 2);
      const f32 t_exit = t_next[axis] < t1 ? t_next[axis] : t1;
      const f32 bound = majorants(
          v.macro_offset + (u32(cell[2]) * v.macro_dims[1] + u32(cell[1])) * v.macro_dims[0] + u32(cell[0]));

      if (bound > 0.f) {
        const f32 inv_bound = 1.f / bound;
        while (lookups < 8192) {
          t -= Kokkos::log(1.f - rng.next_f32()) * inv_bound;
          if (t >= t_exit)
            break;
          ++lookups;
          const Vec4 c = classify(v, org + dir * t);
          if (scatter) {
            if (rng.next_f32() * bound < c.w) {
              albedo = c.xyz();
              return t;
            }
          } else {
            transmittance *= 1.f - c.w * inv_bound;
            // Russian roulette ends rays through dense media early.
            if (transmittance < 0.05f) {
              if (!(transmittance > 0.f) || rng.next_f32() * 0.05f >= transmittance) {
                transmittance = 0.f;
                return -1.f;
              }
              transmittance = 0.05f;
            }
          }
        }
      }

      // Free flight sampling is memoryless: it restarts at the cell border.
      t = t_exit;
      if (t_exit >= t1 || lookups >= 8192)
        break;
      cell[axis] += step[axis];
      if (cell[axis] < 0 || cell[axis] >= int(v.macro_dims[axis]))
        break;
      t_next[axis] += t_delta[axis];
    }
    return -1.f;
  }

  // Delta tracking through all volumes: distance of the first scattering event
  // inside (tmin, tmax), or a negative value when the ray passes unscattered.
  // `albedo` is the scattering albedo at the event.
  KOKKOS_FUNCTION f32 sample_scatter(const Ray &ray, f32 tmin, f32 tmax, Rng &rng, Vec3 &albedo) const
  {
    f32 t_event = -1.f;
    f32 t_limit = tmax;
    f32 unused = 1.f;
    for (u32 vi = 0; vi < count; ++vi) {
      const TfVolume &v = volumes(vi);
      if (v.majorant <= 0.f || v.tf_size == 0)
        continue;
      const Vec3 org = to_local_point(v, ray.org);
      const Vec3 dir = to_local_vector(v, ray.dir);
      f32 t0 = tmin;
      f32 t1 = t_limit;
      if (!clip_to_bounds(v, org, dir, t0, t1))
        continue;
      Vec3 event_albedo{0.f, 0.f, 0.f};
      const f32 t = track(v, org, dir, t0, t1, rng, true, event_albedo, unused);
      if (t >= 0.f) {
        t_event = t;
        t_limit = t;
        albedo = event_albedo;
      }
    }
    return t_event;
  }

  // Ratio tracking estimate of the transmittance along (tmin, tmax).
  KOKKOS_FUNCTION f32 transmittance(const Ray &ray, f32 tmin, f32 tmax, Rng &rng) const
  {
    f32 tr = 1.f;
    Vec3 unused{0.f, 0.f, 0.f};
    for (u32 vi = 0; vi < count; ++vi) {
      const TfVolume &v = volumes(vi);
      if (v.majorant <= 0.f || v.tf_size == 0)
        continue;
      const Vec3 org = to_local_point(v, ray.org);
      const Vec3 dir = to_local_vector(v, ray.dir);
      f32 t0 = tmin;
      f32 t1 = tmax;
      if (!clip_to_bounds(v, org, dir, t0, t1))
        continue;
      track(v, org, dir, t0, t1, rng, false, unused, tr);
      if (tr <= 0.f)
        return 0.f;
    }
    return tr;
  }
};

}
