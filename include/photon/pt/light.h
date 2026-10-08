#pragma once

#include <Kokkos_Core.hpp>

#include "photon/pt/math.h"
#include "photon/pt/rng.h"
#include "photon/pt/sampling.h"

namespace photon::pt {

enum class LightType : u32 {
  Point = 0,
  Directional = 1,
  Spot = 2,
  Area = 3,
  Environment = 4
};

struct Light {
  LightType type{LightType::Point};
  Vec3 position{0.f, 0.f, 0.f};
  Vec3 direction{0.f, -1.f, 0.f};
  Vec3 color{1.f, 1.f, 1.f};
  f32 intensity{1.f};
  f32 spot_angle{0.5f};   // half opening angle of the cone
  f32 spot_falloff{0.1f}; // width of the smooth rim inside the cone
  u32 mesh_prim_begin{0};
  u32 mesh_prim_count{0};
  f32 area{0.f};
  Vec3 edge1{0.f, 0.f, 0.f};
  Vec3 edge2{0.f, 0.f, 0.f};
  // Point/Spot: radius of the emitting sphere. Directional: angular radius of
  // the emitting disk (radians). 0 gives a delta light.
  f32 radius{0.f};
};

struct LightSample {
  Vec3 wi;
  Vec3 Li;
  f32 pdf{0.f};
  f32 dist{0.f};
  bool is_delta{false};
};

KOKKOS_FUNCTION inline f32 clamp(f32 x, f32 a, f32 b)
{
  return x < a ? a : (x > b ? b : x);
}

KOKKOS_FUNCTION inline f32 smoothstep(f32 edge0, f32 edge1, f32 x)
{
  const f32 t = clamp((x - edge0) / (edge1 - edge0), 0.f, 1.f);
  return t * t * (3.f - 2.f * t);
}

// Point on a sphere light as seen from `hit_pos`: a point of the disk through
// the center which faces the shading point.
KOKKOS_FUNCTION inline Vec3 sample_sphere_light_point(const Light &light, const Vec3 &hit_pos, Rng &rng)
{
  if (light.radius <= 0.f)
    return light.position;
  const Vec3 axis = normalize(light.position - hit_pos);
  Vec3 t{}, b{};
  onb_from_normal(axis, t, b);
  const Vec2 d = sample_concentric_disk(rng);
  return light.position + (t * d.x + b * d.y) * light.radius;
}

// Samples a direction towards the light. The returned lights cannot be hit by
// rays (they have no geometry), so the estimate needs no MIS weight: `pdf` is
// the density of the sampled direction and `is_delta` is always true for them.
KOKKOS_FUNCTION inline LightSample sample_light(const Light &light, const Vec3 &hit_pos, Rng &rng)
{
  constexpr f32 INF = 1e30f;

  if (light.type == LightType::Point) {
    const Vec3 center_to = light.position - hit_pos;
    const f32 center_dist2 = dot(center_to, center_to);
    const Vec3 to = sample_sphere_light_point(light, hit_pos, rng) - hit_pos;
    const f32 dist = length(to);
    const Vec3 wi = dist > 0.f ? (to * (1.f / dist)) : Vec3{0.f, 1.f, 0.f};
    const f32 invDist2 = center_dist2 > 0.f ? (1.f / center_dist2) : 0.f;
    return {wi, light.color * (light.intensity * invDist2), 1.f, dist, true};
  }

  if (light.type == LightType::Directional) {
    Vec3 wi = -normalize(light.direction);
    if (light.radius > 0.f) {
      // Uniform direction inside the cone of the angular radius.
      const f32 cos_max = Kokkos::cos(light.radius);
      const f32 cos_theta = 1.f - rng.next_f32() * (1.f - cos_max);
      const f32 sin_theta = Kokkos::sqrt(Kokkos::fmax(0.f, 1.f - cos_theta * cos_theta));
      const f32 phi = TWO_PI * rng.next_f32();
      Vec3 t{}, b{};
      onb_from_normal(wi, t, b);
      wi = normalize(t * (sin_theta * Kokkos::cos(phi)) + b * (sin_theta * Kokkos::sin(phi)) + wi * cos_theta);
    }
    return {wi, light.color * light.intensity, 1.f, INF, true};
  }

  if (light.type == LightType::Spot) {
    const Vec3 center_to = light.position - hit_pos;
    const f32 center_dist2 = dot(center_to, center_to);
    const Vec3 to = sample_sphere_light_point(light, hit_pos, rng) - hit_pos;
    const f32 dist = length(to);
    const Vec3 wi = dist > 0.f ? (to * (1.f / dist)) : Vec3{0.f, 1.f, 0.f};

    const Vec3 spotDir = normalize(light.direction);
    const Vec3 fromLight = -normalize(center_to);
    const f32 cosAngle = dot(fromLight, spotDir);
    const f32 cosInner = Kokkos::cos(Kokkos::fmax(light.spot_angle - light.spot_falloff, 0.f));
    const f32 cosOuter = Kokkos::cos(light.spot_angle);

    f32 falloff = 0.f;
    if (cosAngle >= cosInner)
      falloff = 1.f;
    else if (cosAngle <= cosOuter)
      falloff = 0.f;
    else {
      falloff = smoothstep(cosOuter, cosInner, cosAngle);
    }

    const f32 invDist2 = center_dist2 > 0.f ? (1.f / center_dist2) : 0.f;
    return {wi, light.color * (light.intensity * invDist2 * falloff), 1.f, dist, true};
  }

  if (light.type == LightType::Area) {
    const f32 u = rng.next_f32();
    const f32 v = rng.next_f32();
    const Vec3 sample_pos = light.position + light.edge1 * u + light.edge2 * v;
    const Vec3 light_normal = normalize(cross(light.edge1, light.edge2));
    const Vec3 to = sample_pos - hit_pos;
    const f32 dist = length(to);
    if (dist < 1e-6f)
      return {Vec3{0.f, 1.f, 0.f}, Vec3{0.f, 0.f, 0.f}, 0.f, 0.f, true};
    const Vec3 wi = to * (1.f / dist);
    const f32 cos_light = -dot(wi, light_normal);
    if (cos_light <= 0.f)
      return {Vec3{0.f, 1.f, 0.f}, Vec3{0.f, 0.f, 0.f}, 0.f, 0.f, true};
    const f32 light_area = light.area > 0.f ? light.area : length(cross(light.edge1, light.edge2));
    const f32 pdf_area = 1.f / light_area;
    const f32 pdf_solid = pdf_area * (dist * dist) / cos_light;
    const Vec3 Le = light.color * light.intensity;
    return {wi, Le, pdf_solid, dist * 0.999f, true};
  }

  return {Vec3{0.f, 1.f, 0.f}, Vec3{0.f, 0.f, 0.f}, 0.f, 0.f, true};
}

KOKKOS_FUNCTION inline f32 light_pdf(const Light &light, const Vec3 &hit_pos, const Vec3 &wi)
{
  if (light.type == LightType::Point || light.type == LightType::Directional || light.type == LightType::Spot)
    return 0.f;
  if (light.type == LightType::Area) {
    const Vec3 light_normal = normalize(cross(light.edge1, light.edge2));
    const f32 cos_light = -dot(wi, light_normal);
    if (cos_light <= 0.f)
      return 0.f;
    const f32 light_area = light.area > 0.f ? light.area : length(cross(light.edge1, light.edge2));
    const f32 to_light = length(hit_pos - light.position);
    return (1.f / light_area) * (to_light * to_light) / cos_light;
  }
  return 0.f;
}

}
