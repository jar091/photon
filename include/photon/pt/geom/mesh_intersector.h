#pragma once

#include <Kokkos_Core.hpp>

#include "photon/pt/bvh/bvh.h"
#include "photon/pt/geom/triangle_intersect.h"
#include "photon/pt/math_vec2.h"
#include "photon/pt/ray.h"

namespace photon::pt {

struct MeshHit
{
  f32 t{0.f};
  Vec3 p;
  Vec3 n;
  Vec3 shading_normal;
  Vec2 uv{0.f, 0.f};
  Vec2 bary{0.f, 0.f};
  Vec3 interpolated_color{0.f, 0.f, 0.f};
  u32 prim_id{0};
  u32 geom_id{0};
  u32 material_id{0};
  bool hit{false};
  bool has_interpolated_color{false};
};

// Distance at which the ray enters the box, or a negative value when it misses
// the box inside [tmin, tmax].
KOKKOS_FUNCTION inline f32 aabb_entry_distance(const Aabb &box, const Vec3 &org, const Vec3 &inv_dir, f32 tmin,
    f32 tmax)
{
  const f32 tx0 = (box.lo.x - org.x) * inv_dir.x;
  const f32 tx1 = (box.hi.x - org.x) * inv_dir.x;
  const f32 ty0 = (box.lo.y - org.y) * inv_dir.y;
  const f32 ty1 = (box.hi.y - org.y) * inv_dir.y;
  const f32 tz0 = (box.lo.z - org.z) * inv_dir.z;
  const f32 tz1 = (box.hi.z - org.z) * inv_dir.z;

  const f32 tnear = Kokkos::fmax(Kokkos::fmax(Kokkos::fmin(tx0, tx1), Kokkos::fmin(ty0, ty1)),
      Kokkos::fmax(Kokkos::fmin(tz0, tz1), tmin));
  const f32 tfar = Kokkos::fmin(Kokkos::fmin(Kokkos::fmax(tx0, tx1), Kokkos::fmax(ty0, ty1)),
      Kokkos::fmin(Kokkos::fmax(tz0, tz1), tmax));
  return tnear <= tfar ? tnear : -1.f;
}

KOKKOS_FUNCTION inline Vec3 safe_inverse_direction(const Vec3 &d)
{
  // A tiny non-zero value keeps the slab test free of 0 * inf.
  const f32 eps = 1e-20f;
  auto inv = [eps](f32 x) { return 1.f / (Kokkos::fabs(x) > eps ? x : (x < 0.f ? -eps : eps)); };
  return {inv(d.x), inv(d.y), inv(d.z)};
}

// Closest hit of the ray with the mesh. With `any_hit` the traversal stops at
// the first hit found (occlusion rays), only `hit` is meaningful then.
KOKKOS_FUNCTION inline MeshHit intersect_mesh_bvh(const TriangleMesh &mesh, const Bvh &bvh, const Ray &ray, f32 tmin,
    f32 tmax, bool any_hit = false)
{
  MeshHit best;
  best.t = tmax;

  if (!(tmax > tmin) || bvh.nodes.extent(0) == 0)
    return best;

  const Vec3 inv_dir = safe_inverse_direction(ray.dir);

  f32 best_u = 0.f;
  f32 best_v = 0.f;

  u32 stack[64];
  int sp = 0;
  stack[sp++] = bvh.root;

  while (sp > 0) {
    const u32 node_id = stack[--sp];
    const BvhNode node = bvh.nodes(node_id);

    if (aabb_entry_distance(node.bounds, ray.org, inv_dir, tmin, best.t) < 0.f)
      continue;

    if (node.is_leaf) {
      for (u32 i = 0; i < node.count; ++i) {
        const u32 prim_id = bvh.prim_ids(node.begin + i);
        const u32 i0 = mesh.indices(prim_id * 3 + 0);
        const u32 i1 = mesh.indices(prim_id * 3 + 1);
        const u32 i2 = mesh.indices(prim_id * 3 + 2);
        const Vec3 v0 = mesh.positions(i0);
        const Vec3 v1 = mesh.positions(i1);
        const Vec3 v2 = mesh.positions(i2);

        const TriHit h = intersect_triangle(ray, v0, v1, v2, tmin, best.t);
        if (!h.hit)
          continue;

        best.t = h.t;
        best.prim_id = prim_id;
        best.hit = true;
        best_u = h.u;
        best_v = h.v;
        if (any_hit)
          return best;
      }
    } else {
      if (sp + 2 <= 64) {
        // Visit the nearer child first.
        const f32 dl = aabb_entry_distance(bvh.nodes(node.left).bounds, ray.org, inv_dir, tmin, best.t);
        const f32 dr = aabb_entry_distance(bvh.nodes(node.right).bounds, ray.org, inv_dir, tmin, best.t);
        if (dl >= 0.f && dr >= 0.f) {
          if (dl < dr) {
            stack[sp++] = node.right;
            stack[sp++] = node.left;
          } else {
            stack[sp++] = node.left;
            stack[sp++] = node.right;
          }
        } else if (dl >= 0.f) {
          stack[sp++] = node.left;
        } else if (dr >= 0.f) {
          stack[sp++] = node.right;
        }
      }
    }
  }

  if (!best.hit)
    return best;

  const u32 prim_id = best.prim_id;
  const u32 i0 = mesh.indices(prim_id * 3 + 0);
  const u32 i1 = mesh.indices(prim_id * 3 + 1);
  const u32 i2 = mesh.indices(prim_id * 3 + 2);
  const Vec3 v0 = mesh.positions(i0);
  const Vec3 v1 = mesh.positions(i1);
  const Vec3 v2 = mesh.positions(i2);

  best.p = ray.at(best.t);
  const Vec3 nn = normalize(cross(v1 - v0, v2 - v0));
  best.n = nn;
  best.shading_normal = nn;
  best.uv = Vec2{0.f, 0.f};
  best.bary = Vec2{best_u, best_v};
  best.geom_id = 0;
  best.material_id = 0;

  const f32 w0 = 1.f - best_u - best_v;
  const f32 w1 = best_u;
  const f32 w2 = best_v;

  if (mesh.has_normals()) {
    const Vec3 n0 = mesh.normals(i0);
    const Vec3 n1 = mesh.normals(i1);
    const Vec3 n2 = mesh.normals(i2);
    best.shading_normal = normalize(n0 * w0 + n1 * w1 + n2 * w2);
    if (dot(best.shading_normal, nn) < 0.f)
      best.shading_normal = best.shading_normal * -1.f;
  }

  if (mesh.has_texcoords()) {
    const Vec2 uv0 = mesh.texcoords(i0);
    const Vec2 uv1 = mesh.texcoords(i1);
    const Vec2 uv2 = mesh.texcoords(i2);
    best.uv = Vec2{uv0.x * w0 + uv1.x * w1 + uv2.x * w2, uv0.y * w0 + uv1.y * w1 + uv2.y * w2};
  }

  if (mesh.has_vertex_colors()) {
    const Vec3 vc0 = mesh.vertex_colors(i0);
    if (vc0.x >= 0.f) {
      const Vec3 vc1 = mesh.vertex_colors(i1);
      const Vec3 vc2 = mesh.vertex_colors(i2);
      best.interpolated_color = vc0 * w0 + vc1 * w1 + vc2 * w2;
      best.has_interpolated_color = true;
    }
  }

  if (mesh.has_material_ids()) {
    best.material_id = mesh.material_ids(prim_id);
  }

  return best;
}

} // namespace photon::pt
