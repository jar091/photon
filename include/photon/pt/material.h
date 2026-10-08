#pragma once

#include <Kokkos_Core.hpp>

#include "photon/pt/math.h"

namespace photon::pt {

using i32 = int;

// Where the value of a material parameter comes from. `Constant` uses the value
// stored in the Material itself; attributes and samplers are resolved per hit
// (see MeshAttributes and SamplerTable).
enum class ParamSourceKind : i32 { Constant = 0, Attribute = 1, Sampler = 2 };

struct ParamSource {
  ParamSourceKind kind{ParamSourceKind::Constant};
  i32 index{0}; // attribute slot or sampler id
};

// How Material::alpha is interpreted.
enum class AlphaMode : i32 {
  Blend = 0,  // alpha is the probability to hit the surface (stochastic transparency)
  Opaque = 1, // alpha is ignored
  Mask = 2    // surface is hit when alpha >= alpha_cutoff
};

// Material::flags bits.
constexpr i32 MATERIAL_FLAG_EMITTER_ONLY = 1;     // pure emitter, paths end on it
constexpr i32 MATERIAL_FLAG_HIDDEN_TO_CAMERA = 2; // invisible to camera rays
constexpr i32 MATERIAL_FLAG_EMIT_BOTH_SIDES = 4;  // emission also leaves the back face
constexpr i32 MATERIAL_FLAG_REFRACTIVE = 8;       // transmission refracts with `ior`
constexpr i32 MATERIAL_FLAG_EMIT_BACK_ONLY = 16;  // emission leaves the back face only

struct Material {
  Vec3 base_color{0.8f, 0.8f, 0.8f};
  f32 metallic{0.f};
  f32 roughness{0.5f};
  f32 ior{1.5f};
  f32 transmission{0.f};
  // Disney specular: 0.5 gives the Fresnel reflectance of `ior`, 0 disables the
  // dielectric specular lobe.
  f32 specular{0.5f};
  f32 clearcoat{0.f};
  f32 clearcoat_roughness{0.03f};
  f32 sheen{0.f};
  f32 sheen_tint{0.5f};
  f32 specular_tint{0.f};
  f32 anisotropic{0.f};
  Vec3 emission{0.f, 0.f, 0.f};
  f32 emission_strength{0.f};
  f32 subsurface{0.f};
  Vec3 subsurface_color{1.f, 1.f, 1.f};
  f32 subsurface_radius{1.f};
  f32 alpha{1.f};
  i32 thin{0};
  i32 base_color_tex{-1};
  i32 normal_tex{-1};
  i32 roughness_tex{-1};
  i32 metallic_tex{-1};
  i32 emission_tex{-1};
  i32 alpha_tex{-1};

  // Per-parameter sources (attributes / samplers), used by the ANARI device.
  ParamSource base_color_src{};
  ParamSource alpha_src{};
  ParamSource metallic_src{};
  ParamSource roughness_src{};
  ParamSource emission_src{};
  ParamSource transmission_src{};
  AlphaMode alpha_mode{AlphaMode::Blend};
  f32 alpha_cutoff{0.5f};
  i32 flags{0};
  // 1: the alpha of the base color source multiplies `alpha`.
  i32 base_color_alpha{0};
  // 1: `roughness` from a source is perceptual and gets squared.
  i32 roughness_src_squared{0};
};

KOKKOS_FUNCTION inline Vec3 material_emission(const Material &mat)
{
  return mat.emission * mat.emission_strength;
}

KOKKOS_FUNCTION inline bool material_is_emissive(const Material &mat)
{
  return mat.emission_strength > 0.f &&
      (mat.emission.x > 0.f || mat.emission.y > 0.f || mat.emission.z > 0.f);
}

// True when the material can emit light, also when the emission comes from an
// attribute or a sampler.
KOKKOS_FUNCTION inline bool material_may_emit(const Material &mat)
{
  if (mat.emission_strength <= 0.f)
    return false;
  return mat.emission_src.kind != ParamSourceKind::Constant || mat.emission_tex >= 0
      || mat.emission.x > 0.f || mat.emission.y > 0.f || mat.emission.z > 0.f;
}

}
