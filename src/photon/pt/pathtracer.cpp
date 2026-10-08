#include "photon/pt/pathtracer.h"

#include <Kokkos_Core.hpp>

#include <utility>
#include <vector>

#include "photon/pt/disney_bsdf.h"
#include "photon/pt/environment_map.h"
#include "photon/pt/light.h"
#include "photon/pt/material.h"
#include "photon/pt/rng.h"
#include "photon/pt/sampling.h"
#include "photon/pt/work_order.h"

namespace photon::pt {

namespace {

KOKKOS_FUNCTION inline u32 hash_u32(u32 x)
{
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

// Decorrelated random sequence for a pixel, a sample and a use (dimension).
KOKKOS_FUNCTION inline Rng make_rng(u32 pixel, u32 sample, u32 dimension)
{
  return Rng(hash_u32(pixel + hash_u32(sample + hash_u32(dimension + 0x9e3779b9U))));
}

KOKKOS_FUNCTION inline f32 luminance_of(const Vec3 &c)
{
  return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

// Scene data used by the shading kernels.
struct ShadeScene
{
  Kokkos::View<Material *> materials;
  TextureAtlas textures;
  MeshAttributes attributes;
  SamplerTable samplers;

  Kokkos::View<Vec3 *> positions;
  Kokkos::View<u32 *> indices;
  Kokkos::View<u32 *> material_ids;

  Kokkos::View<Light *> lights;
  u32 light_count{0};

  Kokkos::View<u32 *> emissive_ids;
  Kokkos::View<f32 *> emissive_cdf;
  Kokkos::View<f32 *> prim_light_pdf;
  u32 emissive_count{0};

  EnvironmentMap env;
  bool has_env{false};
  bool env_visible{true};

  VolumeSet volumes;

  // Number of light sampling strategies one is picked from uniformly.
  u32 strategy_count{0};
};

// A point on a triangle of the scene mesh.
struct SurfaceQuery
{
  u32 prim{0};
  u32 material_id{0};
  f32 b1{0.f};
  f32 b2{0.f};
  Vec3 p;
  Vec3 shading_normal;
  Vec2 uv;
  Vec3 color;
  bool has_color{false};
};

KOKKOS_FUNCTION inline Vec4 fetch_attribute(const ShadeScene &sc, i32 slot, const SurfaceQuery &q)
{
  if (slot == ATTRIBUTE_WORLD_POSITION)
    return {q.p, 1.f};
  if (slot == ATTRIBUTE_WORLD_NORMAL)
    return {q.shading_normal, 1.f};
  if (!sc.attributes.has(slot))
    return {0.f, 0.f, 0.f, 1.f};

  const u32 base = u32(sc.attributes.offset[slot]);
  const Vec4 a0 = sc.attributes.values(base + sc.indices(q.prim * 3 + 0));
  const Vec4 a1 = sc.attributes.values(base + sc.indices(q.prim * 3 + 1));
  const Vec4 a2 = sc.attributes.values(base + sc.indices(q.prim * 3 + 2));
  return a0 * (1.f - q.b1 - q.b2) + a1 * q.b1 + a2 * q.b2;
}

KOKKOS_FUNCTION inline Vec4 eval_source(const ShadeScene &sc, const ParamSource &src, const SurfaceQuery &q)
{
  if (src.kind == ParamSourceKind::Attribute)
    return fetch_attribute(sc, src.index, q);
  if (src.index < 0 || u32(src.index) >= sc.samplers.count)
    return {0.f, 0.f, 0.f, 1.f};
  const i32 slot = sc.samplers.infos(u32(src.index)).in_attribute;
  return sc.samplers.sample(src.index, fetch_attribute(sc, slot, q));
}

KOKKOS_FUNCTION inline f32 clamp01f(f32 x)
{
  return x < 0.f ? 0.f : (x > 1.f ? 1.f : x);
}

// Material at a surface point with all textures, attributes and samplers
// applied. `alpha` is the probability that a ray interacts with the surface.
KOKKOS_FUNCTION inline Material resolve_material(const ShadeScene &sc, const SurfaceQuery &q, f32 &alpha)
{
  Material mat = sc.materials.extent(0) ? sc.materials(q.material_id) : Material{};
  const bool has_textures = sc.textures.count > 0;

  alpha = mat.alpha;
  if (mat.alpha_tex >= 0 && has_textures)
    alpha = sc.textures.sample(mat.alpha_tex, q.uv.x, q.uv.y).x;

  if (mat.base_color_tex >= 0 && has_textures) {
    mat.base_color = sc.textures.sample(mat.base_color_tex, q.uv.x, q.uv.y);
  } else if (q.has_color) {
    mat.base_color = q.color;
  }

  if (mat.roughness_tex >= 0 && has_textures)
    mat.roughness = sc.textures.sample(mat.roughness_tex, q.uv.x, q.uv.y).x;

  if (mat.metallic_tex >= 0 && has_textures)
    mat.metallic = sc.textures.sample(mat.metallic_tex, q.uv.x, q.uv.y).x;

  if (mat.base_color_src.kind != ParamSourceKind::Constant) {
    const Vec4 c = eval_source(sc, mat.base_color_src, q);
    mat.base_color = {Kokkos::fmax(c.x, 0.f), Kokkos::fmax(c.y, 0.f), Kokkos::fmax(c.z, 0.f)};
    if (mat.base_color_alpha)
      alpha *= c.w;
  }

  if (mat.alpha_src.kind != ParamSourceKind::Constant)
    alpha *= eval_source(sc, mat.alpha_src, q).x;

  if (mat.metallic_src.kind != ParamSourceKind::Constant)
    mat.metallic = eval_source(sc, mat.metallic_src, q).x;

  if (mat.roughness_src.kind != ParamSourceKind::Constant) {
    const f32 r = clamp01f(eval_source(sc, mat.roughness_src, q).x);
    mat.roughness = mat.roughness_src_squared ? r * r : r;
  }

  if (mat.transmission_src.kind != ParamSourceKind::Constant)
    mat.transmission = eval_source(sc, mat.transmission_src, q).x;

  if (mat.emission_src.kind != ParamSourceKind::Constant) {
    const Vec4 e = eval_source(sc, mat.emission_src, q);
    mat.emission = {Kokkos::fmax(e.x, 0.f), Kokkos::fmax(e.y, 0.f), Kokkos::fmax(e.z, 0.f)};
  }

  mat.metallic = clamp01f(mat.metallic);
  mat.roughness = clamp01f(mat.roughness);
  mat.transmission = clamp01f(mat.transmission);

  if (mat.alpha_mode == AlphaMode::Opaque)
    alpha = 1.f;
  else if (mat.alpha_mode == AlphaMode::Mask)
    alpha = alpha >= mat.alpha_cutoff ? 1.f : 0.f;
  else
    alpha = clamp01f(alpha);

  return mat;
}

// True when the material emits towards a direction on the given side.
KOKKOS_FUNCTION inline bool emits_on_side(const Material &mat, bool front_side)
{
  if (mat.flags & MATERIAL_FLAG_EMIT_BOTH_SIDES)
    return true;
  const bool back_only = (mat.flags & MATERIAL_FLAG_EMIT_BACK_ONLY) != 0;
  return front_side ? !back_only : back_only;
}

struct DirectSample
{
  Vec3 wi;
  Vec3 radiance;
  f32 pdf{0.f};     // density of `wi`, includes the selection of the strategy
  f32 dist{0.f};
  bool mis{false};  // the light can also be found by sampling the BSDF
  bool valid{false};
};

// Picks one light sampling strategy uniformly (an analytic light, the emissive
// triangles or the environment map) and samples a direction towards it.
KOKKOS_FUNCTION inline DirectSample sample_direct(const ShadeScene &sc, const Vec3 &p, Rng &rng)
{
  DirectSample ds;
  if (sc.strategy_count == 0)
    return ds;

  const f32 pick_pdf = 1.f / f32(sc.strategy_count);
  u32 pick = u32(rng.next_f32() * f32(sc.strategy_count));
  if (pick >= sc.strategy_count)
    pick = sc.strategy_count - 1;

  if (pick < sc.light_count) {
    const LightSample ls = sample_light(sc.lights(pick), p, rng);
    if (!(ls.pdf > 0.f))
      return ds;
    ds.wi = ls.wi;
    ds.radiance = ls.Li;
    ds.pdf = ls.pdf * pick_pdf;
    ds.dist = ls.dist;
    ds.mis = false;
    ds.valid = true;
    return ds;
  }

  if (sc.emissive_count > 0 && pick == sc.light_count) {
    // Triangle by the selection CDF, then a uniform point on it.
    const f32 xi = rng.next_f32();
    u32 lo = 0;
    u32 hi = sc.emissive_count - 1;
    while (lo < hi) {
      const u32 mid = (lo + hi) / 2;
      if (sc.emissive_cdf(mid) <= xi)
        lo = mid + 1;
      else
        hi = mid;
    }
    const u32 prim = sc.emissive_ids(lo);
    const f32 pdf_area = sc.prim_light_pdf(prim);
    if (!(pdf_area > 0.f))
      return ds;

    const Vec3 v0 = sc.positions(sc.indices(prim * 3 + 0));
    const Vec3 v1 = sc.positions(sc.indices(prim * 3 + 1));
    const Vec3 v2 = sc.positions(sc.indices(prim * 3 + 2));

    const f32 su = Kokkos::sqrt(rng.next_f32());
    const f32 r2 = rng.next_f32();
    const f32 b0 = 1.f - su;
    const f32 b1 = r2 * su;
    const f32 b2 = 1.f - b0 - b1;
    const Vec3 light_p = v0 * b0 + v1 * b1 + v2 * b2;

    const Vec3 to_light = light_p - p;
    const f32 dist_sq = dot(to_light, to_light);
    if (!(dist_sq > 0.f))
      return ds;
    const f32 dist = Kokkos::sqrt(dist_sq);
    const Vec3 wi = to_light * (1.f / dist);
    const Vec3 light_n = normalize(cross(v1 - v0, v2 - v0));
    const f32 cos_light = -dot(wi, light_n);
    if (Kokkos::fabs(cos_light) < 1e-6f)
      return ds;

    SurfaceQuery q;
    q.prim = prim;
    q.material_id = sc.material_ids.extent(0) > 0 ? sc.material_ids(prim) : 0u;
    q.b1 = b1;
    q.b2 = b2;
    q.p = light_p;
    q.shading_normal = light_n;
    f32 alpha = 1.f;
    const Material emit_mat = resolve_material(sc, q, alpha);
    if (!emits_on_side(emit_mat, cos_light > 0.f))
      return ds;

    ds.wi = wi;
    ds.radiance = material_emission(emit_mat) * alpha;
    ds.pdf = pdf_area * dist_sq / Kokkos::fabs(cos_light) * pick_pdf;
    ds.dist = dist;
    ds.mis = true;
    ds.valid = true;
    return ds;
  }

  if (sc.has_env) {
    f32 env_pdf = 0.f;
    const Vec3 wi = sc.env.sample_direction(rng, env_pdf);
    if (!(env_pdf > 0.f))
      return ds;
    ds.wi = wi;
    ds.radiance = sc.env.evaluate(wi);
    ds.pdf = env_pdf * pick_pdf;
    ds.dist = 1e30f;
    ds.mis = true;
    ds.valid = true;
  }
  return ds;
}

// Bilinear lookup in the background image at the image plane position (s, t).
KOKKOS_FUNCTION inline Vec4 sample_background_image(
    const Kokkos::View<Vec4 **, Kokkos::LayoutRight> &image, f32 s, f32 t)
{
  const int h = int(image.extent(0));
  const int w = int(image.extent(1));
  const f32 fx = s * f32(w) - 0.5f;
  const f32 fy = t * f32(h) - 0.5f;
  const f32 flx = Kokkos::floor(fx);
  const f32 fly = Kokkos::floor(fy);
  const f32 tx = fx - flx;
  const f32 ty = fy - fly;
  auto clampi = [](int i, int n) { return i < 0 ? 0 : (i >= n ? n - 1 : i); };
  const int x0 = clampi(int(flx), w);
  const int x1 = clampi(int(flx) + 1, w);
  const int y0 = clampi(int(fly), h);
  const int y1 = clampi(int(fly) + 1, h);
  const Vec4 a = image(y0, x0) * (1.f - tx) + image(y0, x1) * tx;
  const Vec4 b = image(y1, x0) * (1.f - tx) + image(y1, x1) * tx;
  return a * (1.f - ty) + b * ty;
}

} // namespace

void PathTracer::set_scene(const Scene &scene)
{
  m_scene = scene;
  build_light_tables();
}

void PathTracer::set_backend(std::unique_ptr<RayBackend> backend)
{
  m_backend_owned = std::move(backend);
  m_backend = m_backend_owned.get();
}

void PathTracer::set_backend_ref(RayBackend &backend)
{
  m_backend_owned.reset();
  m_backend = &backend;
}

void PathTracer::build_light_tables()
{
  const Scene &scene = m_scene;
  const bool has_emissive = scene.emissive_count > 0 && scene.emissive_prim_ids.extent(0) >= scene.emissive_count;

  // Analytic lights. Area lights that stand for emissive triangles are left
  // out when the triangles are sampled themselves.
  m_nee_light_count = 0;
  m_nee_lights = Kokkos::View<Light *>();
  if (scene.light_count > 0 && scene.lights.extent(0) >= scene.light_count) {
    auto lights_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, scene.lights);
    std::vector<Light> kept;
    for (u32 i = 0; i < scene.light_count; ++i) {
      const Light &l = lights_h(i);
      if (has_emissive && l.type == LightType::Area && l.mesh_prim_count > 0)
        continue;
      kept.push_back(l);
    }
    if (!kept.empty()) {
      m_nee_light_count = u32(kept.size());
      m_nee_lights = Kokkos::View<Light *>("pt_nee_lights", kept.size());
      auto kept_h = Kokkos::create_mirror_view(m_nee_lights);
      for (size_t i = 0; i < kept.size(); ++i)
        kept_h(i) = kept[i];
      Kokkos::deep_copy(m_nee_lights, kept_h);
    }
  }

  // Emissive triangles: selection CDF by area times emission and the resulting
  // area density of a point on each primitive.
  m_emissive_count = 0;
  m_emissive_cdf = Kokkos::View<f32 *>();
  m_prim_light_pdf = Kokkos::View<f32 *>();
  if (has_emissive) {
    const u32 n = scene.emissive_count;
    auto ids_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, scene.emissive_prim_ids);
    auto areas_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, scene.emissive_prim_areas);
    auto mats_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, scene.materials);
    const bool has_mat_ids = scene.mesh.material_ids.extent(0) > 0;
    auto mat_ids_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, scene.mesh.material_ids);

    std::vector<double> weights(n, 0.0);
    double total = 0.0;
    for (u32 i = 0; i < n; ++i) {
      const u32 prim = ids_h(i);
      const u32 mat_id = has_mat_ids ? mat_ids_h(prim) : 0u;
      f32 power = 1.f;
      if (mat_id < mats_h.extent(0)) {
        const Material &m = mats_h(mat_id);
        const bool constant = m.emission_src.kind == ParamSourceKind::Constant;
        power = constant ? luminance_of(material_emission(m)) : m.emission_strength;
      }
      weights[i] = double(areas_h(i)) * double(power > 1e-6f ? power : 1e-6f);
      if (!(areas_h(i) > 0.f))
        weights[i] = 0.0;
      total += weights[i];
    }

    if (total > 0.0) {
      m_emissive_count = n;
      m_emissive_cdf = Kokkos::View<f32 *>("pt_emissive_cdf", n);
      m_prim_light_pdf = Kokkos::View<f32 *>("pt_prim_light_pdf", scene.mesh.triangle_count());
      auto cdf_h = Kokkos::create_mirror_view(m_emissive_cdf);
      auto pdf_h = Kokkos::create_mirror_view(m_prim_light_pdf);
      Kokkos::deep_copy(pdf_h, 0.f);
      double running = 0.0;
      for (u32 i = 0; i < n; ++i) {
        running += weights[i];
        cdf_h(i) = f32(running / total);
        if (weights[i] > 0.0 && ids_h(i) < pdf_h.extent(0))
          pdf_h(ids_h(i)) = f32(weights[i] / total / double(areas_h(i)));
      }
      cdf_h(n - 1) = 1.f;
      Kokkos::deep_copy(m_emissive_cdf, cdf_h);
      Kokkos::deep_copy(m_prim_light_pdf, pdf_h);
    }
  }

  // Ray offset relative to the size of the scene.
  m_ray_epsilon = 1e-3f;
  if (scene.ray_epsilon > 0.f) {
    m_ray_epsilon = scene.ray_epsilon;
  } else if (scene.bvh.nodes.extent(0) > 0 && scene.mesh.triangle_count() > 0) {
    auto root = Kokkos::subview(scene.bvh.nodes, std::make_pair(size_t(scene.bvh.root), size_t(scene.bvh.root) + 1));
    auto root_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, root);
    const Aabb b = root_h(0).bounds;
    if (b.hi.x >= b.lo.x) {
      const f32 diag = length(b.hi - b.lo);
      const f32 eps = 1e-4f * diag;
      m_ray_epsilon = eps < 1e-6f ? 1e-6f : (eps > 1e-2f ? 1e-2f : eps);
    }
  }
}

void PathTracer::ensure_views(u32 pixel_count, u32 w, u32 h)
{
  if (pixel_count == m_cached_pixel_count && w == m_cached_w && h == m_cached_h)
    return;

  m_cached_pixel_count = pixel_count;
  m_cached_w = w;
  m_cached_h = h;

  // Output 2D views
  m_out_color = Kokkos::View<Vec3 **, Kokkos::LayoutRight>("pt_color", h, w);
  m_out_alpha = Kokkos::View<f32 **, Kokkos::LayoutRight>("pt_alpha", h, w);
  m_out_depth = Kokkos::View<f32 **, Kokkos::LayoutRight>("pt_depth", h, w);
  m_out_normal = Kokkos::View<Vec3 **, Kokkos::LayoutRight>("pt_normal", h, w);
  m_out_albedo = Kokkos::View<Vec3 **, Kokkos::LayoutRight>("pt_albedo", h, w);

  // Accumulators
  m_accum = Kokkos::View<Vec3 *>("pt_accum", pixel_count);
  m_accum_sq = Kokkos::View<Vec3 *>("pt_accum_sq", pixel_count);
  m_accum_alpha = Kokkos::View<f32 *>("pt_accum_alpha", pixel_count);
  m_pixel_samples = Kokkos::View<u32 *>("pt_pixel_spp", pixel_count);

  // AOVs
  m_aov_albedo = Kokkos::View<Vec3 *>("pt_aov_albedo", pixel_count);
  m_aov_normal = Kokkos::View<Vec3 *>("pt_aov_normal", pixel_count);
  m_aov_depth = Kokkos::View<f32 *>("pt_aov_depth", pixel_count);
  m_aov_written = Kokkos::View<u32 *>("pt_aov_written", pixel_count);

  // Rays
  m_ray_origins = Kokkos::View<Vec3 *>("pt_ray_o", pixel_count);
  m_ray_dirs = Kokkos::View<Vec3 *>("pt_ray_d", pixel_count);
  m_ray_tmin = Kokkos::View<f32 *>("pt_ray_tmin", pixel_count);
  m_ray_tmax = Kokkos::View<f32 *>("pt_ray_tmax", pixel_count);

  // Hits
  m_hits = Kokkos::View<HitResult *>("pt_hits", pixel_count);

  // Path state
  m_throughput = Kokkos::View<Vec3 *>("pt_throughput", pixel_count);
  m_active = Kokkos::View<u32 *>("pt_active", pixel_count);
  m_prev_pdf = Kokkos::View<f32 *>("pt_prev_pdf", pixel_count);
  m_path_depth = Kokkos::View<u32 *>("pt_path_depth", pixel_count);
  m_primary = Kokkos::View<u32 *>("pt_primary", pixel_count);
  m_path_dist = Kokkos::View<f32 *>("pt_path_dist", pixel_count);

  // Shadow rays
  m_shadow_origins = Kokkos::View<Vec3 *>("pt_shadow_o", pixel_count);
  m_shadow_dirs = Kokkos::View<Vec3 *>("pt_shadow_d", pixel_count);
  m_shadow_tmin = Kokkos::View<f32 *>("pt_shadow_tmin", pixel_count);
  m_shadow_tmax = Kokkos::View<f32 *>("pt_shadow_tmax", pixel_count);

  // NEE state
  m_nee_contrib = Kokkos::View<Vec3 *>("pt_nee", pixel_count);
  m_nee_active = Kokkos::View<u32 *>("pt_nee_active", pixel_count);
  m_occluded = Kokkos::View<u32 *>("pt_occluded", pixel_count);
  m_sample_start = Kokkos::View<Vec3 *>("pt_sample_start", pixel_count);
}

RenderResult PathTracer::render()
{
  const u32 w = params.width;
  const u32 h = params.height;
  const u32 spp = params.samples_per_pixel;
  const u32 max_depth = params.max_depth > 0 ? params.max_depth : 1u;
  const u32 sample_offset = params.sample_offset;
  const Vec3 bg_color = params.background_color;
  const f32 bg_alpha = params.background_alpha;
  const auto bg_image = params.background_image;
  const bool has_bg_image = bg_image.extent(0) > 0 && bg_image.extent(1) > 0;
  const Vec3 ambient = params.ambient_color * params.ambient_radiance;
  const bool adaptive = params.adaptive_sampling;
  const f32 throughput_clamp = params.throughput_clamp;
  const bool bottom_up = params.bottom_up;
  const f32 region_x0 = params.image_region[0];
  const f32 region_y0 = params.image_region[1];
  const f32 region_x1 = params.image_region[2];
  const f32 region_y1 = params.image_region[3];

  const u32 pixel_count = w * h;
  ensure_views(pixel_count, w, h);

  RenderResult out;
  out.color = m_out_color;
  out.alpha = m_out_alpha;
  out.depth = m_out_depth;
  out.normal = m_out_normal;
  out.albedo = m_out_albedo;

  if (!m_backend) {
    Kokkos::deep_copy(out.color, Vec3{0.f, 0.f, 0.f});
    Kokkos::deep_copy(out.alpha, 0.f);
    Kokkos::deep_copy(out.depth, 0.f);
    Kokkos::deep_copy(out.normal, Vec3{0.f, 0.f, 0.f});
    Kokkos::deep_copy(out.albedo, Vec3{0.f, 0.f, 0.f});
    return out;
  }

  const Camera cam = camera;

  ShadeScene sc;
  sc.materials = m_scene.materials;
  sc.textures = m_scene.textures;
  sc.attributes = m_scene.attributes;
  sc.samplers = m_scene.samplers;
  sc.positions = m_scene.mesh.positions;
  sc.indices = m_scene.mesh.indices;
  sc.material_ids = m_scene.mesh.material_ids;
  sc.lights = m_nee_lights;
  sc.light_count = m_nee_light_count;
  sc.emissive_ids = m_scene.emissive_prim_ids;
  sc.emissive_cdf = m_emissive_cdf;
  sc.prim_light_pdf = m_prim_light_pdf;
  sc.emissive_count = m_emissive_count;
  sc.has_env = m_scene.env_map.has_value() && m_scene.env_map->width > 0 && m_scene.env_map->height > 0;
  if (sc.has_env)
    sc.env = m_scene.env_map.value();
  sc.env_visible = m_scene.env_visible;
  sc.volumes = m_scene.volumes;
  sc.strategy_count = sc.light_count + (sc.emissive_count > 0 ? 1u : 0u) + (sc.has_env ? 1u : 0u);

  const bool has_volumes = sc.volumes.count > 0;
  const f32 eps = m_ray_epsilon;

  auto &accum = m_accum;
  auto &accum_sq = m_accum_sq;
  auto &accum_alpha = m_accum_alpha;
  auto &pixel_samples = m_pixel_samples;

  auto &aov_albedo = m_aov_albedo;
  auto &aov_normal = m_aov_normal;
  auto &aov_depth = m_aov_depth;
  auto &aov_written = m_aov_written;

  RayBatch rays;
  rays.origins = m_ray_origins;
  rays.directions = m_ray_dirs;
  rays.tmin = m_ray_tmin;
  rays.tmax = m_ray_tmax;
  rays.count = pixel_count;

  HitBatch hits;
  hits.hits = m_hits;
  hits.count = pixel_count;

  auto &throughput = m_throughput;
  auto &active = m_active;
  auto &prev_pdf = m_prev_pdf;
  auto &path_depth = m_path_depth;
  auto &primary = m_primary;
  auto &path_dist = m_path_dist;

  RayBatch shadow_rays;
  shadow_rays.origins = m_shadow_origins;
  shadow_rays.directions = m_shadow_dirs;
  shadow_rays.tmin = m_shadow_tmin;
  shadow_rays.tmax = m_shadow_tmax;
  shadow_rays.count = pixel_count;

  auto &nee_contrib = m_nee_contrib;
  auto &nee_active = m_nee_active;
  auto &occluded = m_occluded;
  auto &sample_start = m_sample_start;

  const f32 adaptive_threshold = 0.01f;
  const u32 adaptive_min_samples = 16u;

  // One more iteration than scattering events finds the lights reached by the
  // last sampled direction; a few extra ones let paths cross transparent
  // surfaces without losing depth.
  const u32 max_iterations = max_depth + 1u + 8u;

  // The shading kernels visit the pixels in a permuted order that balances
  // the load of the host threads (see WorkOrder).
  const WorkOrder order(pixel_count);

  for (u32 s = 0; s < spp; ++s) {
    const u32 sample_index = s + sample_offset;

    Kokkos::parallel_for(
        "pt_init_sample",
        Kokkos::RangePolicy<>(0, int(pixel_count)),
        KOKKOS_LAMBDA(const int idx) {
          const u32 x = u32(idx) % w;
          const u32 y = u32(idx) / w;

          // Zero per-frame accumulators on first sample (replaces Kokkos::deep_copy)
          if (s == 0u) {
            accum(idx) = Vec3{0.f, 0.f, 0.f};
            accum_sq(idx) = Vec3{0.f, 0.f, 0.f};
            accum_alpha(idx) = 0.f;
            pixel_samples(idx) = 0u;
            aov_albedo(idx) = Vec3{0.f, 0.f, 0.f};
            aov_normal(idx) = Vec3{0.f, 0.f, 0.f};
            aov_depth(idx) = 1e30f;
            aov_written(idx) = 0u;
          }

          if (adaptive && s >= adaptive_min_samples) {
            const u32 ns = pixel_samples(idx);
            if (ns >= adaptive_min_samples) {
              const Vec3 mean = accum(idx) * (1.f / f32(ns));
              const Vec3 sq = accum_sq(idx) * (1.f / f32(ns));
              const Vec3 var = sq - Vec3{mean.x * mean.x, mean.y * mean.y, mean.z * mean.z};
              const f32 max_var = Kokkos::fmax(var.x, Kokkos::fmax(var.y, var.z));
              const f32 lum = Kokkos::fmax(0.2126f * mean.x + 0.7152f * mean.y + 0.0722f * mean.z, 1e-4f);
              if (max_var / (lum * lum) < adaptive_threshold) {
                active(idx) = 0u;
                rays.tmin(idx) = 1.f;
                rays.tmax(idx) = 0.f;
                return;
              }
            }
          }

          Rng rng = make_rng(u32(idx), sample_index, 0u);
          f32 u = (f32(x) + rng.next_f32()) / f32(w);
          f32 v = (f32(y) + rng.next_f32()) / f32(h);
          if (!bottom_up)
            v = 1.f - v;
          u = region_x0 + (region_x1 - region_x0) * u;
          v = region_y0 + (region_y1 - region_y0) * v;

          const Ray r = cam.ray(u, v, rng);
          rays.origins(idx) = r.org;
          rays.directions(idx) = r.dir;
          rays.tmin(idx) = 0.f;
          rays.tmax(idx) = 1e30f;

          throughput(idx) = Vec3{1.f, 1.f, 1.f};
          active(idx) = 1u;
          prev_pdf(idx) = -1.f;
          path_depth(idx) = 0u;
          primary(idx) = 1u;
          path_dist(idx) = 0.f;
          sample_start(idx) = accum(idx);
          pixel_samples(idx) = pixel_samples(idx) + 1u;

          // Empty shadow ray (tmax < tmin) until next event estimation fills it.
          shadow_rays.origins(idx) = r.org;
          shadow_rays.directions(idx) = r.dir;
          shadow_rays.tmin(idx) = 1.f;
          shadow_rays.tmax(idx) = 0.f;
        });

    for (u32 iteration = 0; iteration < max_iterations; ++iteration) {
      m_backend->trace_closest(rays, hits);

      Kokkos::parallel_for(
          "pt_shade",
          Kokkos::RangePolicy<>(0, int(order.range())),
          KOKKOS_LAMBDA(const int work_index) {
            const int idx = order.item(work_index);
            if (idx < 0)
              return;

            // Zero per-bounce NEE state unconditionally (replaces Kokkos::deep_copy)
            nee_contrib(idx) = Vec3{0.f, 0.f, 0.f};
            nee_active(idx) = 0u;

            if (active(idx) == 0u)
              return;

            const HitResult hr = hits.hits(idx);
            const Vec3 ray_org = rays.origins(idx);
            const Vec3 ray_dir = rays.directions(idx);
            const u32 depth = path_depth(idx);

            Rng rng = make_rng(u32(idx), sample_index, 1u + iteration);

            // Scattering inside a volume in front of the surface hit.
            if (has_volumes) {
              Vec3 albedo{0.f, 0.f, 0.f};
              const f32 t_end = hr.hit ? hr.t : 1e30f;
              const f32 t_scatter =
                  sc.volumes.sample_scatter(Ray{ray_org, ray_dir}, rays.tmin(idx), t_end, rng, albedo);
              if (t_scatter >= 0.f) {
                const Vec3 p = ray_org + ray_dir * t_scatter;

                if (primary(idx) != 0u) {
                  primary(idx) = 0u;
                  accum_alpha(idx) = accum_alpha(idx) + 1.f;
                  if (aov_written(idx) == 0u) {
                    aov_written(idx) = 1u;
                    aov_depth(idx) = path_dist(idx) + t_scatter;
                    aov_normal(idx) = -ray_dir;
                    aov_albedo(idx) = albedo;
                  }
                }

                if (depth >= max_depth) {
                  active(idx) = 0u;
                  return;
                }

                const Vec3 tp = throughput(idx) * albedo;
                const f32 phase = 0.25f * INV_PI;

                const DirectSample ds = sample_direct(sc, p, rng);
                if (ds.valid) {
                  const f32 w_mis = ds.mis ? power_heuristic(1, ds.pdf, 1, phase) : 1.f;
                  nee_contrib(idx) = tp * ds.radiance * (phase * w_mis / ds.pdf);
                  nee_active(idx) = 1u;
                  shadow_rays.origins(idx) = p;
                  shadow_rays.directions(idx) = ds.wi;
                  shadow_rays.tmin(idx) = 0.f;
                  shadow_rays.tmax(idx) = ds.dist < 1e29f ? ds.dist * (1.f - 1e-4f) : 1e30f;
                }

                // Russian roulette, as for surfaces below.
                Vec3 tp_next = tp;
                if (depth >= 2) {
                  const f32 max_tp = Kokkos::fmax(tp.x, Kokkos::fmax(tp.y, tp.z));
                  const f32 survive = Kokkos::fmin(Kokkos::fmax(max_tp, 0.05f), 0.95f);
                  if (rng.next_f32() >= survive) {
                    active(idx) = 0u;
                    return;
                  }
                  tp_next = tp * (1.f / survive);
                }

                throughput(idx) = tp_next;
                prev_pdf(idx) = phase;
                path_depth(idx) = depth + 1u;
                rays.origins(idx) = p;
                rays.directions(idx) = sample_uniform_sphere(rng);
                rays.tmin(idx) = 0.f;
                rays.tmax(idx) = 1e30f;
                return;
              }
            }

            if (!hr.hit) {
              Vec3 L{0.f, 0.f, 0.f};
              if (primary(idx) != 0u) {
                if (sc.has_env && sc.env_visible) {
                  L = sc.env.evaluate(ray_dir);
                  accum_alpha(idx) = accum_alpha(idx) + 1.f;
                } else if (has_bg_image) {
                  const u32 x = u32(idx) % w;
                  const u32 y = u32(idx) / w;
                  const f32 bs = (f32(x) + 0.5f) / f32(w);
                  const f32 bt = (f32(y) + 0.5f) / f32(h);
                  const Vec4 bg = sample_background_image(bg_image, bs, bottom_up ? bt : 1.f - bt);
                  L = bg.xyz() * bg.w;
                  accum_alpha(idx) = accum_alpha(idx) + bg.w;
                } else {
                  L = bg_color * bg_alpha;
                  accum_alpha(idx) = accum_alpha(idx) + bg_alpha;
                }
              } else {
                L = ambient;
                if (sc.has_env) {
                  f32 w_mis = 1.f;
                  const f32 bsdf_pdf = prev_pdf(idx);
                  if (bsdf_pdf > 0.f) {
                    const f32 light_pdf = sc.env.pdf(ray_dir) / f32(sc.strategy_count);
                    w_mis = power_heuristic(1, bsdf_pdf, 1, light_pdf);
                  }
                  L += sc.env.evaluate(ray_dir) * w_mis;
                }
              }

              accum(idx) = accum(idx) + throughput(idx) * L;
              active(idx) = 0u;
              return;
            }

            const Vec3 p = hr.position;
            const Vec3 wo = -ray_dir;

            // Vertices without a normal (zero vector) are shaded flat; the shading
            // normal is kept on the side of the geometric normal.
            Vec3 hit_sn = hr.shading_normal;
            if (!(dot(hit_sn, hit_sn) > 0.25f))
              hit_sn = hr.normal;
            else if (dot(hit_sn, hr.normal) < 0.f)
              hit_sn = hit_sn * -1.f;

            SurfaceQuery q;
            q.prim = hr.prim_id;
            q.material_id = hr.material_id;
            q.b1 = hr.bary.x;
            q.b2 = hr.bary.y;
            q.p = p;
            q.shading_normal = hit_sn;
            q.uv = hr.uv;
            q.color = hr.interpolated_color;
            q.has_color = hr.has_interpolated_color;

            f32 alpha_val = 1.f;
            Material mat = resolve_material(sc, q, alpha_val);

            const bool hidden = (mat.flags & MATERIAL_FLAG_HIDDEN_TO_CAMERA) != 0 && primary(idx) != 0u;

            // Stochastic transparency: the ray continues behind the surface.
            if (hidden || (alpha_val < 1.f && rng.next_f32() >= alpha_val)) {
              rays.origins(idx) = p;
              rays.tmin(idx) = eps;
              rays.tmax(idx) = 1e30f;
              path_dist(idx) = path_dist(idx) + hr.t;
              // Shadow rays stop at transparent surfaces, so lights behind them
              // are only found by this ray.
              prev_pdf(idx) = -1.f;
              return;
            }

            Vec3 n = hr.normal;
            Vec3 sn = hit_sn;
            const bool front_side = dot(n, wo) >= 0.f;
            const bool is_transmissive = mat.transmission > 0.f;

            if (!is_transmissive && !front_side) {
              n = n * -1.f;
              sn = sn * -1.f;
            }

            if (mat.normal_tex >= 0 && sc.textures.count > 0) {
              Vec3 nm = sc.textures.sample(mat.normal_tex, hr.uv.x, hr.uv.y);
              nm = nm * 2.f - Vec3{1.f, 1.f, 1.f};
              Vec3 tangent{}, bitangent{};
              onb_from_normal(sn, tangent, bitangent);
              sn = normalize(tangent * nm.x + bitangent * nm.y + sn * nm.z);
            }

            if (primary(idx) != 0u) {
              primary(idx) = 0u;
              accum_alpha(idx) = accum_alpha(idx) + 1.f;
              if (aov_written(idx) == 0u) {
                aov_written(idx) = 1u;
                aov_depth(idx) = path_dist(idx) + hr.t;
                aov_normal(idx) = normalize(sn);
                aov_albedo(idx) = mat.base_color;
              }
            }

            if (material_may_emit(mat) && emits_on_side(mat, front_side)) {
              f32 w_mis = 1.f;
              const f32 bsdf_pdf = prev_pdf(idx);
              if (bsdf_pdf > 0.f && sc.emissive_count > 0) {
                const f32 pdf_area = sc.prim_light_pdf(hr.prim_id);
                const f32 cos_light = Kokkos::fabs(dot(hr.normal, wo));
                if (pdf_area > 0.f && cos_light > 0.f) {
                  const f32 light_pdf = pdf_area * hr.t * hr.t / cos_light / f32(sc.strategy_count);
                  w_mis = power_heuristic(1, bsdf_pdf, 1, light_pdf);
                }
              }
              accum(idx) = accum(idx) + throughput(idx) * material_emission(mat) * w_mis;
            }

            if ((mat.flags & MATERIAL_FLAG_EMITTER_ONLY) != 0 || depth >= max_depth) {
              active(idx) = 0u;
              return;
            }

            // Next event estimation.
            {
              const DirectSample ds = sample_direct(sc, p, rng);
              if (ds.valid) {
                const Vec3 f = disney_bsdf_eval(mat, wo, ds.wi, n, sn);
                if (f.x > 0.f || f.y > 0.f || f.z > 0.f) {
                  f32 w_mis = 1.f;
                  if (ds.mis)
                    w_mis = power_heuristic(1, ds.pdf, 1, disney_bsdf_pdf(mat, wo, ds.wi, n, sn));
                  nee_contrib(idx) = throughput(idx) * f * ds.radiance * (w_mis / ds.pdf);
                  nee_active(idx) = 1u;

                  // The shadow ray starts above the surface and ends just before
                  // the sampled point of the light (which can be geometry too).
                  const Vec3 shadow_org = p + n * eps;
                  shadow_rays.origins(idx) = shadow_org;
                  shadow_rays.tmin(idx) = 0.f;
                  if (ds.dist < 1e29f) {
                    const Vec3 to_light = p + ds.wi * ds.dist - shadow_org;
                    const f32 shadow_dist = length(to_light);
                    shadow_rays.directions(idx) = shadow_dist > 0.f ? to_light * (1.f / shadow_dist) : ds.wi;
                    shadow_rays.tmax(idx) = shadow_dist * (1.f - 1e-3f);
                  } else {
                    shadow_rays.directions(idx) = ds.wi;
                    shadow_rays.tmax(idx) = 1e30f;
                  }
                }
              }
            }

            const BsdfSample bs = disney_bsdf_sample(mat, wo, n, sn, rng);
            if (bs.pdf <= 0.f || (bs.f.x <= 0.f && bs.f.y <= 0.f && bs.f.z <= 0.f)) {
              active(idx) = 0u;
              return;
            }

            Vec3 tp = throughput(idx) * (bs.f * (1.f / bs.pdf));

            if (depth >= 2) {
              const f32 max_tp = Kokkos::fmax(tp.x, Kokkos::fmax(tp.y, tp.z));
              const f32 survive = Kokkos::fmin(Kokkos::fmax(max_tp, 0.05f), 0.95f);
              if (rng.next_f32() >= survive) {
                active(idx) = 0u;
                return;
              }
              tp = tp * (1.f / survive);
            }

            if (throughput_clamp > 0.f) {
              const f32 m = Kokkos::fmax(tp.x, Kokkos::fmax(tp.y, tp.z));
              if (m > throughput_clamp)
                tp = tp * (throughput_clamp / m);
            }

            if (!(tp.x == tp.x && tp.y == tp.y && tp.z == tp.z)) {
              active(idx) = 0u;
              return;
            }

            throughput(idx) = tp;
            prev_pdf(idx) = bs.is_specular ? -1.f : bs.pdf;
            path_depth(idx) = depth + 1u;

            const Vec3 wi = normalize(bs.wi);
            // Start on the side of the surface the new direction leaves from.
            const f32 side = dot(wi, hr.normal) < 0.f ? -1.f : 1.f;
            rays.origins(idx) = p + hr.normal * (side * eps);
            rays.directions(idx) = wi;
            rays.tmin(idx) = 0.f;
            rays.tmax(idx) = 1e30f;
          });

      m_backend->trace_occluded(shadow_rays, occluded);

      u32 active_count = 0;
      Kokkos::parallel_reduce(
          "pt_nee_accum_pack",
          Kokkos::RangePolicy<>(0, int(order.range())),
          KOKKOS_LAMBDA(const int work_index, u32 &count) {
            const int idx = order.item(work_index);
            if (idx < 0)
              return;
            if (nee_active(idx) != 0u && occluded(idx) == 0u) {
              Vec3 contrib = nee_contrib(idx);
              if (has_volumes) {
                Rng rng = make_rng(u32(idx), sample_index, 0x10000u + iteration);
                const Ray shadow{shadow_rays.origins(idx), shadow_rays.directions(idx)};
                contrib = contrib
                    * sc.volumes.transmittance(shadow, shadow_rays.tmin(idx), shadow_rays.tmax(idx), rng);
              }
              if (contrib.x == contrib.x && contrib.y == contrib.y && contrib.z == contrib.z)
                accum(idx) = accum(idx) + contrib;
            }
            // Unused shadow rays must not be traced again (tmax < tmin).
            shadow_rays.tmin(idx) = 1.f;
            shadow_rays.tmax(idx) = 0.f;
            if (active(idx) != 0u) {
              count += 1u;
            } else {
              rays.tmin(idx) = 1.f;
              rays.tmax(idx) = 0.f;
            }
          },
          active_count);

      if (active_count == 0)
        break;
    }

    if (adaptive) {
      Kokkos::parallel_for(
          "pt_update_variance",
          Kokkos::RangePolicy<>(0, int(pixel_count)),
          KOKKOS_LAMBDA(const int idx) {
            const Vec3 sample_val = accum(idx) - sample_start(idx);
            accum_sq(idx) = accum_sq(idx) + Vec3{
              sample_val.x * sample_val.x,
              sample_val.y * sample_val.y,
              sample_val.z * sample_val.z};
          });
    }
  }

  Kokkos::parallel_for(
      "pt_resolve",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {int(h), int(w)}),
      KOKKOS_LAMBDA(const int y, const int x) {
        const int idx = y * int(w) + x;
        const u32 ns = pixel_samples(idx);
        const f32 inv = ns > 0 ? (1.f / f32(ns)) : 0.f;
        out.color(y, x) = accum(idx) * inv;
        out.alpha(y, x) = accum_alpha(idx) * inv;
        out.depth(y, x) = aov_depth(idx);
        out.normal(y, x) = aov_normal(idx);
        out.albedo(y, x) = aov_albedo(idx);
      });

  Kokkos::fence();
  return out;
}

} // namespace photon::pt
