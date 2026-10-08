#ifdef PHOTON_HAS_EMBREE

#include "photon/pt/backend/embree_backend.h"

#include "photon/pt/geom/triangle_intersect.h"
#include "photon/pt/scene.h"
#include "photon/pt/work_order.h"

#include <cstdio>
#include <stdexcept>
#include <vector>

namespace photon::pt {
namespace {

void embree_error_handler(void *, const RTCError code, const char *str)
{
  std::fprintf(stderr, "[photon][embree] error %d: %s\n", int(code), str ? str : "(null)");
}

// Host view of the data of `view`. When Kokkos keeps its views in host memory
// this is the view itself, otherwise the data is copied to `staging`.
template <typename HostView, typename AnyView>
HostView host_view_of(const AnyView &view, HostView &staging, bool copy_data)
{
  if constexpr (Kokkos::SpaceAccessibility<Kokkos::HostSpace, typename AnyView::memory_space>::assignable) {
    return view;
  } else {
    if (staging.extent(0) != view.extent(0))
      staging = HostView("emb_staging", view.extent(0));
    if (copy_data)
      Kokkos::deep_copy(staging, view);
    return staging;
  }
}

// Copies results from the staging view back, unless host_view_of() returned
// the view itself.
template <typename HostView, typename AnyView>
void copy_back(const AnyView &view, const HostView &host)
{
  if constexpr (!Kokkos::SpaceAccessibility<Kokkos::HostSpace, typename AnyView::memory_space>::assignable)
    Kokkos::deep_copy(view, host);
}

inline Vec3 normalize_safe(const Vec3 v)
{
  const f32 len2 = dot(v, v);
  if (len2 <= 0.f)
    return {0.f, 0.f, 0.f};
  const f32 inv_len = 1.f / std::sqrt(len2);
  return v * inv_len;
}

}

EmbreeBackend::EmbreeBackend()
{
  m_device = rtcNewDevice(nullptr);
  if (!m_device) {
    throw std::runtime_error("rtcNewDevice failed");
  }
  rtcSetDeviceErrorFunction(m_device, embree_error_handler, nullptr);
}

EmbreeBackend::~EmbreeBackend()
{
  if (m_scene) {
    rtcReleaseScene(m_scene);
    m_scene = nullptr;
  }
  if (m_device) {
    rtcReleaseDevice(m_device);
    m_device = nullptr;
  }
}

void EmbreeBackend::build_accel(const Scene &scene)
{
  const auto &mesh = scene.mesh;

  m_positions_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.positions);
  m_indices_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.indices);

  m_has_normals = mesh.has_normals();
  if (m_has_normals)
    m_normals_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.normals);

  m_has_texcoords = mesh.has_texcoords();
  if (m_has_texcoords)
    m_texcoords_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.texcoords);

  m_has_material_ids = mesh.has_material_ids();
  if (m_has_material_ids)
    m_material_ids_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.material_ids);

  m_has_vertex_colors = mesh.has_vertex_colors();
  if (m_has_vertex_colors)
    m_vertex_colors_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.vertex_colors);

  if (m_scene) {
    rtcReleaseScene(m_scene);
    m_scene = nullptr;
  }

  m_scene = rtcNewScene(m_device);

  RTCGeometry geom = rtcNewGeometry(m_device, RTC_GEOMETRY_TYPE_TRIANGLE);

  const auto &pos_h = m_positions_h;
  const auto &idx_h = m_indices_h;

  const size_t vertex_count = pos_h.extent(0);
  const size_t index_count = idx_h.extent(0);

  auto *vb = static_cast<float *>(rtcSetNewGeometryBuffer(
      geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(float) * 3, vertex_count));

  for (size_t i = 0; i < vertex_count; ++i) {
    vb[i * 3 + 0] = pos_h(i).x;
    vb[i * 3 + 1] = pos_h(i).y;
    vb[i * 3 + 2] = pos_h(i).z;
  }

  auto *ib = static_cast<unsigned *>(rtcSetNewGeometryBuffer(
      geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(unsigned) * 3, index_count / 3));

  for (size_t i = 0; i < index_count / 3; ++i) {
    ib[i * 3 + 0] = idx_h(i * 3 + 0);
    ib[i * 3 + 1] = idx_h(i * 3 + 1);
    ib[i * 3 + 2] = idx_h(i * 3 + 2);
  }

  rtcCommitGeometry(geom);
  rtcAttachGeometry(m_scene, geom);
  rtcReleaseGeometry(geom);

  rtcCommitScene(m_scene);
}

void EmbreeBackend::ensure_buffers(u32 n)
{
  // The staging views are (re)allocated by host_view_of() when they are needed.
  m_buf_size = n;
}

void EmbreeBackend::trace_closest(const RayBatch &rays, HitBatch &hits)
{
  if (!m_scene) {
    throw std::runtime_error("EmbreeBackend::trace_closest called before build_accel");
  }

  const u32 n = rays.count;
  if (hits.count != n) {
    throw std::runtime_error("HitBatch.count must match RayBatch.count");
  }

  // Local copies: the kernel must not capture `this`.
  const RTCScene scene = m_scene;
  const auto org_h = host_view_of(rays.origins, m_ray_org_h, true);
  const auto dir_h = host_view_of(rays.directions, m_ray_dir_h, true);
  const auto tmin_h = host_view_of(rays.tmin, m_ray_tmin_h, true);
  const auto tmax_h = host_view_of(rays.tmax, m_ray_tmax_h, true);
  const auto hit_h = host_view_of(hits.hits, m_hit_h, false);
  const auto positions_h = m_positions_h;
  const auto indices_h = m_indices_h;
  const auto normals_h = m_normals_h;
  const auto texcoords_h = m_texcoords_h;
  const auto material_ids_h = m_material_ids_h;
  const auto vertex_colors_h = m_vertex_colors_h;
  const bool has_normals = m_has_normals;
  const bool has_texcoords = m_has_texcoords;
  const bool has_material_ids = m_has_material_ids;
  const bool has_vertex_colors = m_has_vertex_colors;

  // Embree is thread safe: the rays are traced on all host threads, in an
  // order that balances their load (the active rays are neighbours).
  const WorkOrder order(n);
  Kokkos::parallel_for(
      "embree_trace_closest",
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, int(order.range())),
      [=](const int work_index) {
        const int i = order.item(work_index);
        if (i < 0)
          return;
        const f32 tnear = tmin_h(i);
        const f32 tfar = tmax_h(i);
        if (!(tfar > tnear)) {
          // Inactive ray: only the flag is written.
          hit_h(i).hit = false;
          return;
        }
        HitResult hr{};

        RTCRayHit rh{};
        rh.ray.org_x = org_h(i).x;
        rh.ray.org_y = org_h(i).y;
        rh.ray.org_z = org_h(i).z;
        rh.ray.dir_x = dir_h(i).x;
        rh.ray.dir_y = dir_h(i).y;
        rh.ray.dir_z = dir_h(i).z;
        rh.ray.tnear = tnear;
        rh.ray.tfar = tfar;
        rh.ray.mask = 0xFFFFFFFFu;
        rh.ray.flags = 0;

        rh.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        rh.hit.primID = RTC_INVALID_GEOMETRY_ID;

        rtcIntersect1(scene, &rh, nullptr);

        hr.hit = (rh.hit.geomID != RTC_INVALID_GEOMETRY_ID);
        if (hr.hit) {
          hr.t = rh.ray.tfar;
          hr.position = org_h(i) + dir_h(i) * hr.t;
          hr.prim_id = rh.hit.primID;
          hr.geom_id = rh.hit.geomID;
          hr.inst_id = 0u;

          const u32 pid = hr.prim_id;
          const f32 bu = rh.hit.u;
          const f32 bv = rh.hit.v;
          const f32 bw = 1.f - bu - bv;
          hr.bary = {bu, bv};

          const u32 i0 = indices_h(pid * 3 + 0);
          const u32 i1 = indices_h(pid * 3 + 1);
          const u32 i2 = indices_h(pid * 3 + 2);

          const Vec3 p0 = positions_h(i0);
          const Vec3 p1 = positions_h(i1);
          const Vec3 p2 = positions_h(i2);

          hr.normal = normalize_safe(cross(p1 - p0, p2 - p0));

          if (has_normals) {
            hr.shading_normal = normalize_safe(normals_h(i0) * bw + normals_h(i1) * bu + normals_h(i2) * bv);
            if (dot(hr.shading_normal, hr.normal) < 0.f)
              hr.shading_normal = hr.shading_normal * -1.f;
          } else {
            hr.shading_normal = hr.normal;
          }

          if (has_texcoords) {
            const Vec2 t0 = texcoords_h(i0), t1 = texcoords_h(i1), t2 = texcoords_h(i2);
            hr.uv = {t0.x * bw + t1.x * bu + t2.x * bv, t0.y * bw + t1.y * bu + t2.y * bv};
          } else {
            hr.uv = {bu, bv};
          }

          hr.material_id = has_material_ids ? material_ids_h(pid) : 0u;

          if (has_vertex_colors && vertex_colors_h(i0).x >= 0.f) {
            hr.interpolated_color =
                vertex_colors_h(i0) * bw + vertex_colors_h(i1) * bu + vertex_colors_h(i2) * bv;
            hr.has_interpolated_color = true;
          }
        }

        hit_h(i) = hr;
      });

  copy_back(hits.hits, hit_h);
}

void EmbreeBackend::trace_occluded(const RayBatch &rays, Kokkos::View<u32 *> occluded)
{
  if (!m_scene) {
    throw std::runtime_error("EmbreeBackend::trace_occluded called before build_accel");
  }

  const u32 n = rays.count;

  const RTCScene scene = m_scene;
  const auto org_h = host_view_of(rays.origins, m_ray_org_h, true);
  const auto dir_h = host_view_of(rays.directions, m_ray_dir_h, true);
  const auto tmin_h = host_view_of(rays.tmin, m_ray_tmin_h, true);
  const auto tmax_h = host_view_of(rays.tmax, m_ray_tmax_h, true);
  const auto occ_h = host_view_of(occluded, m_occ_h, false);

  const WorkOrder order(n);
  Kokkos::parallel_for(
      "embree_trace_occluded",
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, int(order.range())),
      [=](const int work_index) {
        const int i = order.item(work_index);
        if (i < 0)
          return;
        const f32 tnear = tmin_h(i);
        const f32 tfar = tmax_h(i);
        if (!(tfar > tnear)) {
          occ_h(i) = 0u;
          return;
        }

        RTCRay ray{};
        ray.org_x = org_h(i).x;
        ray.org_y = org_h(i).y;
        ray.org_z = org_h(i).z;
        ray.dir_x = dir_h(i).x;
        ray.dir_y = dir_h(i).y;
        ray.dir_z = dir_h(i).z;
        ray.tnear = tnear;
        ray.tfar = tfar;
        ray.mask = 0xFFFFFFFFu;
        ray.flags = 0;

        rtcOccluded1(scene, &ray, nullptr);
        occ_h(i) = (ray.tfar < 0.f) ? 1u : 0u;
      });

  copy_back(occluded, occ_h);
}

}

#endif
