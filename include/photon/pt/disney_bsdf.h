#pragma once

#include <Kokkos_Core.hpp>

#include "photon/pt/material.h"
#include "photon/pt/math.h"
#include "photon/pt/rng.h"
#include "photon/pt/sampling.h"

namespace photon::pt {

struct BsdfSample {
  Vec3 wi;
  Vec3 f;
  f32 pdf{0.f};
  bool is_specular{false};
};

KOKKOS_FUNCTION inline f32 saturate(f32 x)
{
  return Kokkos::fmin(1.f, Kokkos::fmax(0.f, x));
}

KOKKOS_FUNCTION inline f32 dielectric_f0(f32 ior)
{
  const f32 a = (1.f - ior) / (1.f + ior);
  return a * a;
}

KOKKOS_FUNCTION inline Vec3 fresnel_schlick(f32 cosTheta, Vec3 F0)
{
  const f32 m = 1.f - saturate(cosTheta);
  const f32 m2 = m * m;
  const f32 m5 = m2 * m2 * m;
  return F0 + (Vec3{1.f, 1.f, 1.f} - F0) * m5;
}

KOKKOS_FUNCTION inline Vec3 fresnel_schlick(f32 cosTheta, Vec3 F0, Vec3 F90)
{
  const f32 m = 1.f - saturate(cosTheta);
  const f32 m2 = m * m;
  const f32 m5 = m2 * m2 * m;
  return F0 + (F90 - F0) * m5;
}

// Exact Fresnel reflectance of a dielectric boundary, eta = eta_i / eta_t.
KOKKOS_FUNCTION inline f32 fresnel_dielectric(f32 cos_i, f32 eta)
{
  cos_i = saturate(cos_i);
  const f32 sin2_t = eta * eta * (1.f - cos_i * cos_i);
  if (sin2_t >= 1.f)
    return 1.f;
  const f32 cos_t = Kokkos::sqrt(1.f - sin2_t);
  const f32 r_par = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
  const f32 r_perp = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
  return 0.5f * (r_par * r_par + r_perp * r_perp);
}

KOKKOS_FUNCTION inline f32 D_ggx(f32 NoH, f32 a)
{
  const f32 a2 = a * a;
  const f32 d = NoH * NoH * (a2 - 1.f) + 1.f;
  return a2 / (PI * d * d);
}

KOKKOS_FUNCTION inline f32 D_ggx_aniso(f32 NoH, f32 XoH, f32 YoH, f32 ax, f32 ay)
{
  const f32 d = (XoH * XoH) / (ax * ax) + (YoH * YoH) / (ay * ay) + NoH * NoH;
  return 1.f / (PI * ax * ay * d * d);
}

KOKKOS_FUNCTION inline void aniso_roughness(f32 roughness, f32 anisotropic, f32 &ax, f32 &ay)
{
  const f32 aspect = Kokkos::sqrt(1.f - 0.9f * anisotropic);
  ax = Kokkos::fmax(roughness / aspect, 0.001f);
  ay = Kokkos::fmax(roughness * aspect, 0.001f);
}

KOKKOS_FUNCTION inline f32 G1_smith(f32 NoV, f32 a)
{
  const f32 a2 = a * a;
  const f32 denom = NoV + Kokkos::sqrt(a2 + (1.f - a2) * NoV * NoV);
  if (denom <= 0.f)
    return 0.f;
  return (2.f * NoV) / denom;
}

KOKKOS_FUNCTION inline f32 G_smith(f32 NoV, f32 NoL, f32 a)
{
  return G1_smith(NoV, a) * G1_smith(NoL, a);
}

KOKKOS_FUNCTION inline f32 ggx_pdf_reflection(f32 NoH, f32 VoH, f32 a)
{
  const f32 D = D_ggx(NoH, a);
  const f32 denom = 4.f * Kokkos::fmax(VoH, 1e-7f);
  return D * NoH / denom;
}

// Scale of the dielectric specular lobe: Disney `specular` 0.5 is the Fresnel
// reflectance of `ior`, 0 removes the lobe.
KOKKOS_FUNCTION inline f32 disney_specular_scale(const Material &mat)
{
  return Kokkos::fmax(2.f * mat.specular, 0.f);
}

// Normal incidence (F0) and grazing (F90) reflectance of the specular lobe. The
// transmissive fraction of a dielectric is handled by the transmission lobe.
KOKKOS_FUNCTION inline void disney_specular_f0_f90(const Material &mat, Vec3 &F0, Vec3 &F90)
{
  const f32 scale = disney_specular_scale(mat);
  const f32 opaque = 1.f - saturate(mat.transmission);
  const f32 f0d = Kokkos::fmin(dielectric_f0(mat.ior) * scale, 1.f) * opaque;
  Vec3 F0_dielectric = Vec3{f0d, f0d, f0d};
  if (mat.specular_tint > 0.f) {
    const f32 lum = 0.2126f * mat.base_color.x + 0.7152f * mat.base_color.y + 0.0722f * mat.base_color.z;
    const Vec3 c_tint = lum > 0.f ? mat.base_color * (1.f / lum) : Vec3{1.f, 1.f, 1.f};
    F0_dielectric = lerp(F0_dielectric, c_tint * f0d, mat.specular_tint);
  }
  F0 = lerp(F0_dielectric, mat.base_color, mat.metallic);
  const f32 f90d = Kokkos::fmin(scale, 1.f) * opaque;
  F90 = lerp(Vec3{f90d, f90d, f90d}, Vec3{1.f, 1.f, 1.f}, mat.metallic);
}

// Fraction of the energy that is left for the diffuse lobe below the dielectric
// specular lobe.
KOKKOS_FUNCTION inline f32 disney_diffuse_scale(const Material &mat, f32 NoV)
{
  const f32 scale = disney_specular_scale(mat);
  const f32 f0d = Kokkos::fmin(dielectric_f0(mat.ior) * scale, 1.f);
  const f32 f90d = Kokkos::fmin(scale, 1.f);
  const f32 m = 1.f - saturate(NoV);
  const f32 m2 = m * m;
  return 1.f - (f0d + (f90d - f0d) * m2 * m2 * m);
}

// The specular lobe of a (nearly) perfectly smooth surface is a delta lobe: it is
// only produced by disney_bsdf_sample() and excluded from eval() and pdf().
KOKKOS_FUNCTION inline bool disney_specular_is_delta(const Material &mat)
{
  return mat.roughness < 0.001f;
}

// Selection weights of the lobes for a given outgoing direction. They only steer
// the sampling, the estimate is unbiased for any positive weights.
KOKKOS_FUNCTION inline void disney_lobe_weights(const Material &mat, f32 NoV, f32 &w_diffuse,
    f32 &w_specular, f32 &w_transmission, f32 &w_clearcoat)
{
  const f32 opaque = 1.f - saturate(mat.transmission);
  const f32 dielectric = 1.f - saturate(mat.metallic);

  const f32 diffuse_energy = dielectric * opaque * disney_diffuse_scale(mat, NoV);
  const f32 base_lum =
      0.2126f * mat.base_color.x + 0.7152f * mat.base_color.y + 0.0722f * mat.base_color.z;
  w_diffuse = diffuse_energy > 0.f ? diffuse_energy * Kokkos::fmax(base_lum, 0.05f) : 0.f;
  if (mat.sheen > 0.f && dielectric * opaque > 0.f)
    w_diffuse += 0.1f * mat.sheen;

  Vec3 F0{}, F90{};
  disney_specular_f0_f90(mat, F0, F90);
  const Vec3 F = fresnel_schlick(NoV, F0, F90);
  const f32 spec = 0.2126f * F.x + 0.7152f * F.y + 0.0722f * F.z;
  w_specular = spec > 0.f ? Kokkos::fmax(spec, 0.05f) : 0.f;

  w_transmission = saturate(mat.transmission) * dielectric;

  w_clearcoat = 0.f;
  if (mat.clearcoat > 0.f) {
    const f32 m = 1.f - saturate(NoV);
    const f32 m2 = m * m;
    w_clearcoat = Kokkos::fmax(mat.clearcoat * 0.25f * (0.04f + 0.96f * m2 * m2 * m), 0.02f);
  }
}

KOKKOS_FUNCTION inline Vec3 eval_diffuse(const Material &mat, f32 NoL, f32 NoV)
{
  const f32 w = (1.f - mat.metallic) * (1.f - mat.transmission) * disney_diffuse_scale(mat, NoV);
  if (w <= 0.f)
    return Vec3{0.f, 0.f, 0.f};

  if (mat.subsurface > 0.f) {
    const Vec3 diffuse = mat.base_color * (w * INV_PI * NoL);
    const f32 Fss90 = 0.5f;
    const f32 Fd90 = Fss90;
    const f32 m = 1.f - NoL;
    const f32 Fss = 1.f + (Fd90 - 1.f) * m * m * m * m * m;
    const f32 ss = 1.25f * (Fss * (1.f / (NoL + 0.001f) - 0.5f) + 0.5f);
    const Vec3 ss_color = lerp(mat.base_color, mat.subsurface_color, mat.subsurface);
    const Vec3 subsurface_term = ss_color * (w * INV_PI * NoL * ss);
    return lerp(diffuse, subsurface_term, mat.subsurface);
  }

  return mat.base_color * (w * INV_PI * NoL);
}

// Specular reflection times the cosine of the incident direction:
// F * D * G / (4 * NoV * NoL) * NoL.
KOKKOS_FUNCTION inline Vec3 eval_specular_reflection(const Material &mat,
    const Vec3 &wo, const Vec3 &wi, const Vec3 &n, f32 a)
{
  const f32 NoV = saturate(dot(n, wo));
  const f32 NoL = saturate(dot(n, wi));
  if (NoV <= 0.f || NoL <= 0.f)
    return Vec3{0.f, 0.f, 0.f};

  const Vec3 h = normalize(wi + wo);
  const f32 NoH = saturate(dot(n, h));
  const f32 VoH = saturate(dot(wo, h));

  Vec3 F0{}, F90{};
  disney_specular_f0_f90(mat, F0, F90);
  const Vec3 F = fresnel_schlick(VoH, F0, F90);

  f32 D, G_val;
  if (mat.anisotropic > 0.f) {
    f32 ax, ay;
    aniso_roughness(a, mat.anisotropic, ax, ay);
    Vec3 tangent{}, bitangent{};
    onb_from_normal(n, tangent, bitangent);
    const f32 XoH = dot(tangent, h);
    const f32 YoH = dot(bitangent, h);
    D = D_ggx_aniso(NoH, XoH, YoH, ax, ay);
    G_val = G_smith(NoV, NoL, ax) * G_smith(NoV, NoL, ay);
    G_val = Kokkos::sqrt(G_val);
  } else {
    D = D_ggx(NoH, a);
    G_val = G_smith(NoV, NoL, a);
  }

  const f32 denom = 4.f * Kokkos::fmax(NoV, 1e-7f);
  return F * (D * G_val / denom);
}

KOKKOS_FUNCTION inline Vec3 eval_sheen(const Material &mat, const Vec3 &wo,
    const Vec3 &wi, const Vec3 &n)
{
  if (mat.sheen <= 0.f || mat.metallic >= 1.f)
    return Vec3{0.f, 0.f, 0.f};

  const Vec3 h = normalize(wi + wo);
  const f32 HoL = Kokkos::fabs(dot(h, wi));

  const f32 lum = 0.2126f * mat.base_color.x + 0.7152f * mat.base_color.y + 0.0722f * mat.base_color.z;
  const Vec3 c_tint = lum > 0.f ? mat.base_color * (1.f / lum) : Vec3{1.f, 1.f, 1.f};
  const Vec3 c_sheen = lerp(Vec3{1.f, 1.f, 1.f}, c_tint, mat.sheen_tint);

  const f32 m = 1.f - HoL;
  const f32 m2 = m * m;
  const f32 m5 = m2 * m2 * m;

  const f32 NoL = saturate(dot(n, wi));
  const f32 w = (1.f - mat.metallic) * (1.f - mat.transmission);
  return c_sheen * (mat.sheen * m5 * w * NoL);
}

KOKKOS_FUNCTION inline Vec3 eval_clearcoat(const Material &mat, const Vec3 &wo,
    const Vec3 &wi, const Vec3 &n)
{
  if (mat.clearcoat <= 0.f)
    return Vec3{0.f, 0.f, 0.f};

  const f32 NoV = saturate(dot(n, wo));
  const f32 NoL = saturate(dot(n, wi));
  if (NoV <= 0.f || NoL <= 0.f)
    return Vec3{0.f, 0.f, 0.f};

  const f32 a = Kokkos::fmax(mat.clearcoat_roughness, 0.001f);

  const Vec3 h = normalize(wi + wo);
  const f32 NoH = saturate(dot(n, h));
  const f32 VoH = saturate(dot(wo, h));

  const f32 F0 = dielectric_f0(1.5f);
  const Vec3 F = fresnel_schlick(VoH, Vec3{F0, F0, F0});
  const f32 D = D_ggx(NoH, a);
  const f32 G = G_smith(NoV, NoL, a);

  const f32 denom = 4.f * Kokkos::fmax(NoV, 1e-7f);
  const f32 w = mat.clearcoat * 0.25f;
  return F * (w * D * G / denom);
}

// BSDF times the cosine of the incident direction for the reflection lobes. Delta
// lobes (smooth specular reflection, transmission) are not included, they are only
// produced by disney_bsdf_sample().
KOKKOS_FUNCTION inline Vec3 disney_bsdf_eval(const Material &mat, const Vec3 &wo,
    const Vec3 &wi, const Vec3 &n, const Vec3 &shading_n)
{
  if (dot(wo, n) <= 0.f || dot(wi, n) <= 0.f)
    return Vec3{0.f, 0.f, 0.f};

  const f32 NoL = saturate(dot(shading_n, wi));
  const f32 NoV = saturate(dot(shading_n, wo));
  if (NoL <= 0.f || NoV <= 0.f)
    return Vec3{0.f, 0.f, 0.f};

  const f32 a = Kokkos::fmax(mat.roughness, 0.001f);

  Vec3 f = eval_diffuse(mat, NoL, NoV);
  if (!disney_specular_is_delta(mat))
    f += eval_specular_reflection(mat, wo, wi, shading_n, a);
  f += eval_sheen(mat, wo, wi, shading_n);
  f += eval_clearcoat(mat, wo, wi, shading_n);
  return f;
}

// Density of disney_bsdf_sample() for the reflection lobes (solid angle measure).
KOKKOS_FUNCTION inline f32 disney_bsdf_pdf(const Material &mat, const Vec3 &wo,
    const Vec3 &wi, const Vec3 &n, const Vec3 &shading_n)
{
  if (dot(wo, n) <= 0.f || dot(wi, n) <= 0.f)
    return 0.f;

  const f32 NoL = saturate(dot(shading_n, wi));
  const f32 NoV = saturate(dot(shading_n, wo));
  if (NoL <= 0.f || NoV <= 0.f)
    return 0.f;

  const f32 a = Kokkos::fmax(mat.roughness, 0.001f);

  f32 w_diffuse, w_specular, w_transmission, w_clearcoat;
  disney_lobe_weights(mat, NoV, w_diffuse, w_specular, w_transmission, w_clearcoat);

  const f32 sum_w = w_diffuse + w_specular + w_transmission + w_clearcoat;
  if (sum_w <= 0.f)
    return 0.f;

  const f32 p_diffuse = w_diffuse / sum_w;
  const f32 p_specular = w_specular / sum_w;
  const f32 p_clearcoat = w_clearcoat / sum_w;

  f32 pdf = 0.f;

  if (p_diffuse > 0.f)
    pdf += p_diffuse * (NoL * INV_PI);

  if (p_specular > 0.f && !disney_specular_is_delta(mat)) {
    const Vec3 h = normalize(wi + wo);
    const f32 NoH = saturate(dot(shading_n, h));
    const f32 VoH = saturate(dot(wo, h));
    pdf += p_specular * ggx_pdf_reflection(NoH, VoH, a);
  }

  if (p_clearcoat > 0.f) {
    const Vec3 h = normalize(wi + wo);
    const f32 NoH = saturate(dot(shading_n, h));
    const f32 VoH = saturate(dot(wo, h));
    const f32 a_cc = Kokkos::fmax(mat.clearcoat_roughness, 0.001f);
    pdf += p_clearcoat * ggx_pdf_reflection(NoH, VoH, a_cc);
  }

  return Kokkos::fmax(pdf, 0.f);
}

KOKKOS_FUNCTION inline Vec3 sample_ggx_half_vector(const Vec3 &n, f32 a, Rng &rng)
{
  Vec3 tangent{}, bitangent{};
  onb_from_normal(n, tangent, bitangent);

  const f32 r1 = rng.next_f32();
  const f32 r2 = rng.next_f32();
  const f32 a2 = a * a;

  const f32 cos_theta = Kokkos::sqrt((1.f - r1) / (1.f + (a2 - 1.f) * r1));
  const f32 sin_theta = Kokkos::sqrt(Kokkos::fmax(0.f, 1.f - cos_theta * cos_theta));
  const f32 phi = 2.f * PI * r2;

  const f32 x = sin_theta * Kokkos::cos(phi);
  const f32 y = sin_theta * Kokkos::sin(phi);
  const f32 z = cos_theta;

  return normalize(tangent * x + bitangent * y + n * z);
}

// Samples an incident direction. For non-specular samples `f` is the BSDF times
// the cosine and `pdf` the density of all reflection lobes, so f / pdf is the
// path weight. Specular samples (is_specular) come from delta lobes: `pdf` is 1
// and `f` is the path weight itself.
KOKKOS_FUNCTION inline BsdfSample disney_bsdf_sample(const Material &mat,
    const Vec3 &wo, const Vec3 &n, const Vec3 &shading_n, Rng &rng)
{
  BsdfSample out{};

  const bool front_face = dot(wo, n) > 0.f;
  if (!front_face && mat.transmission <= 0.f)
    return out;

  const f32 rough = Kokkos::fmax(mat.roughness, 0.001f);
  const f32 NoV = saturate(Kokkos::fabs(dot(shading_n, wo)));

  f32 w_diffuse, w_specular, w_transmission, w_clearcoat;
  // Energy of the transmission lobe relative to the whole BSDF.
  f32 transmission_energy = 1.f;
  if (!front_face) {
    // Leaving the medium: only the dielectric boundary is left.
    w_diffuse = 0.f;
    w_specular = 0.f;
    w_transmission = 1.f;
    w_clearcoat = 0.f;
  } else {
    disney_lobe_weights(mat, NoV, w_diffuse, w_specular, w_transmission, w_clearcoat);
    transmission_energy = w_transmission;
  }

  const f32 sum_w = w_diffuse + w_specular + w_transmission + w_clearcoat;
  if (sum_w <= 0.f)
    return out;

  const f32 p_diffuse = w_diffuse / sum_w;
  const f32 p_specular = w_specular / sum_w;
  const f32 p_transmission = w_transmission / sum_w;
  const f32 p_clearcoat = w_clearcoat / sum_w;

  const f32 xi = rng.next_f32();

  if (xi < p_diffuse) {
    auto s = sample_cosine_hemisphere(shading_n, rng);
    out.wi = s.direction;
    out.f = disney_bsdf_eval(mat, wo, out.wi, n, shading_n);
    out.pdf = disney_bsdf_pdf(mat, wo, out.wi, n, shading_n);
    return out;
  }

  if (xi < p_diffuse + p_specular) {
    if (disney_specular_is_delta(mat)) {
      out.wi = reflect(-wo, shading_n);
      if (dot(out.wi, n) <= 0.f)
        out.wi = reflect(-wo, n);
      if (dot(out.wi, n) <= 0.f)
        return BsdfSample{};

      Vec3 F0{}, F90{};
      disney_specular_f0_f90(mat, F0, F90);
      const Vec3 F = fresnel_schlick(saturate(dot(wo, shading_n)), F0, F90);
      out.f = F * (1.f / p_specular);
      out.pdf = 1.f;
      out.is_specular = true;
      return out;
    }

    const Vec3 h = sample_ggx_half_vector(shading_n, rough, rng);
    out.wi = reflect(-wo, h);
    if (dot(out.wi, n) <= 0.f)
      return BsdfSample{};

    out.f = disney_bsdf_eval(mat, wo, out.wi, n, shading_n);
    out.pdf = disney_bsdf_pdf(mat, wo, out.wi, n, shading_n);
    return out;
  }

  if (xi < p_diffuse + p_specular + p_transmission && mat.transmission > 0.f) {
    // Dielectric boundary, sampled as a delta lobe around a (possibly rough)
    // microfacet normal: Fresnel picks between reflection and transmission.
    const Vec3 nn = front_face ? n : n * -1.f;
    Vec3 sn_face = front_face ? shading_n : shading_n * -1.f;
    if (dot(sn_face, wo) <= 0.f)
      sn_face = nn;

    Vec3 h = sn_face;
    if (mat.roughness >= 0.001f) {
      h = sample_ggx_half_vector(sn_face, rough, rng);
      if (dot(h, wo) <= 0.f)
        h = sn_face;
    }

    const bool refractive = (mat.flags & MATERIAL_FLAG_REFRACTIVE) != 0 && mat.thin == 0;
    const f32 ior = Kokkos::fmax(mat.ior, 1.0001f);
    const f32 eta = refractive ? (front_face ? (1.f / ior) : ior) : 1.f;
    const f32 cos_i = saturate(dot(wo, h));

    f32 F;
    if (refractive) {
      F = fresnel_dielectric(cos_i, eta);
    } else {
      const f32 f0 = dielectric_f0(mat.ior);
      F = f0 + (1.f - f0) * Kokkos::pow(1.f - cos_i, 5.f);
    }

    const f32 lobe_weight = transmission_energy / p_transmission;
    out.pdf = 1.f;
    out.is_specular = true;

    if (rng.next_f32() < F) {
      out.wi = reflect(-wo, h);
      if (dot(out.wi, nn) <= 0.f)
        out.wi = reflect(-wo, nn);
      if (dot(out.wi, nn) <= 0.f)
        return BsdfSample{};
      out.f = Vec3{1.f, 1.f, 1.f} * lobe_weight;
      return out;
    }

    if (refractive) {
      out.wi = normalize(refract(-wo, h, eta));
      if (dot(out.wi, nn) >= 0.f)
        out.wi = normalize(refract(-wo, sn_face, eta));
      if (dot(out.wi, nn) >= 0.f)
        return BsdfSample{};
    } else {
      out.wi = normalize(-wo);
    }
    out.f = mat.base_color * lobe_weight;
    return out;
  }

  if (p_clearcoat > 0.f) {
    const f32 a_cc = Kokkos::fmax(mat.clearcoat_roughness, 0.001f);
    const Vec3 h = sample_ggx_half_vector(shading_n, a_cc, rng);
    out.wi = reflect(-wo, h);
    if (dot(out.wi, n) <= 0.f)
      return BsdfSample{};
    out.f = disney_bsdf_eval(mat, wo, out.wi, n, shading_n);
    out.pdf = disney_bsdf_pdf(mat, wo, out.wi, n, shading_n);
    return out;
  }

  return out;
}

}
