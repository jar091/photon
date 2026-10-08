#include "photon/anari/SceneObjects.h"

#include <algorithm>
#include <cmath>

namespace photon::anari_device {

namespace {

constexpr float kPi = 3.14159265358979323846f;

const char *kAttributeNames[bridge::kMeshAttributeCount] = {
    "attribute0", "attribute1", "attribute2", "attribute3", "color"};

std::array<helium::ChangeObserverPtr<Array1D>, bridge::kMeshAttributeCount> makeObservers(helium::BaseObject *self)
{
  return {helium::ChangeObserverPtr<Array1D>(self),
      helium::ChangeObserverPtr<Array1D>(self),
      helium::ChangeObserverPtr<Array1D>(self),
      helium::ChangeObserverPtr<Array1D>(self),
      helium::ChangeObserverPtr<Array1D>(self)};
}

bool isFloat3Array(const Array1D *array)
{
  return array && array->elementType() == ANARI_FLOAT32_VEC3 && array->size() > 0;
}

// Adds the triangle with the winding that makes its normal point along `outward`.
void pushTriangle(GeometryMesh &mesh, uint32_t a, uint32_t b, uint32_t c, const float3 &outward)
{
  const float3 n = linalg::cross(mesh.positions[b] - mesh.positions[a], mesh.positions[c] - mesh.positions[a]);
  if (linalg::dot(n, outward) < 0.f)
    std::swap(b, c);
  mesh.indices.push_back(a);
  mesh.indices.push_back(b);
  mesh.indices.push_back(c);
}

// Unit sphere with `slices` segments around the axis and `stacks` along it.
struct SphereTemplate
{
  std::vector<float3> directions;
  std::vector<uint32_t> indices;

  SphereTemplate(uint32_t slices, uint32_t stacks)
  {
    directions.push_back(float3(0.f, 0.f, 1.f));
    for (uint32_t i = 1; i < stacks; ++i) {
      const float phi = kPi * float(i) / float(stacks);
      for (uint32_t j = 0; j < slices; ++j) {
        const float theta = 2.f * kPi * float(j) / float(slices);
        directions.push_back(
            float3(std::sin(phi) * std::cos(theta), std::sin(phi) * std::sin(theta), std::cos(phi)));
      }
    }
    directions.push_back(float3(0.f, 0.f, -1.f));

    const uint32_t bottom = uint32_t(directions.size()) - 1;
    auto add = [&](uint32_t a, uint32_t b, uint32_t c) {
      // Counter-clockwise when seen from the outside.
      const float3 n = linalg::cross(directions[b] - directions[a], directions[c] - directions[a]);
      if (linalg::dot(n, directions[a] + directions[b] + directions[c]) < 0.f)
        std::swap(b, c);
      indices.push_back(a);
      indices.push_back(b);
      indices.push_back(c);
    };
    for (uint32_t j = 0; j < slices; ++j) {
      const uint32_t j1 = (j + 1) % slices;
      add(0, 1 + j, 1 + j1);
      const uint32_t last = 1 + (stacks - 2) * slices;
      add(bottom, last + j, last + j1);
    }
    for (uint32_t i = 0; i + 2 < stacks; ++i) {
      for (uint32_t j = 0; j < slices; ++j) {
        const uint32_t j1 = (j + 1) % slices;
        const uint32_t a = 1 + i * slices + j;
        const uint32_t b = 1 + i * slices + j1;
        const uint32_t c = 1 + (i + 1) * slices + j;
        const uint32_t d = 1 + (i + 1) * slices + j1;
        add(a, c, b);
        add(b, c, d);
      }
    }
  }
};

} // namespace

Geometry::Geometry(PhotonGlobalState *s, std::string_view subtype)
    : Object(ANARI_GEOMETRY, s),
      m_subtype(subtype),
      m_index(this),
      m_vertexPosition(this),
      m_vertexNormal(this),
      m_vertexRadius(this),
      m_primitiveRadius(this),
      m_faceVaryingNormal(this),
      m_vertexAttributes(makeObservers(this)),
      m_primitiveAttributes(makeObservers(this)),
      m_faceVaryingAttributes(makeObservers(this)),
      m_isovalueArray(this)
{
  if (subtype == "triangle")
    m_type = Type::Triangle;
  else if (subtype == "quad")
    m_type = Type::Quad;
  else if (subtype == "sphere")
    m_type = Type::Sphere;
  else if (subtype == "cylinder")
    m_type = Type::Cylinder;
  else if (subtype == "cone")
    m_type = Type::Cone;
  else if (subtype == "curve")
    m_type = Type::Curve;
  else if (subtype == "isosurface")
    m_type = Type::Isosurface;
  else {
    reportMessage(ANARI_SEVERITY_WARNING,
        "photon does not support geometry subtype '%s', the geometry is not rendered",
        m_subtype.c_str());
  }
}

void Geometry::commitParameters()
{
  m_index = getParamObject<Array1D>("primitive.index");
  m_vertexPosition = getParamObject<Array1D>("vertex.position");
  m_vertexNormal = getParamObject<Array1D>("vertex.normal");
  m_vertexRadius = getParamObject<Array1D>("vertex.radius");
  m_primitiveRadius = getParamObject<Array1D>("primitive.radius");
  m_faceVaryingNormal = getParamObject<Array1D>("faceVarying.normal");
  for (int slot = 0; slot < bridge::kMeshAttributeCount; ++slot) {
    const std::string name = kAttributeNames[slot];
    m_vertexAttributes[slot] = getParamObject<Array1D>("vertex." + name);
    m_primitiveAttributes[slot] = getParamObject<Array1D>("primitive." + name);
    m_faceVaryingAttributes[slot] = getParamObject<Array1D>("faceVarying." + name);
  }
  m_radius = getParam<float>("radius", 1.f);
  const std::string caps = getParamString("caps", "none");
  m_capFirst = caps == "first" || caps == "both";
  m_capSecond = caps == "second" || caps == "both";

  if (m_type == Type::Isosurface) {
    m_field = getParamObject<SpatialField>("field");
    m_isovalueArray = getParamObject<Array1D>("isovalue");
    m_hasIsovalue = getParam("isovalue", ANARI_FLOAT32, &m_isovalue);
    for (int slot = 0; slot < bridge::kMeshAttributeCount; ++slot) {
      m_uniformAttributes[slot] = float4(0.f, 0.f, 0.f, 1.f);
      m_hasUniformAttribute[slot] = getParam(kAttributeNames[slot], ANARI_FLOAT32_VEC4, &m_uniformAttributes[slot]);
    }
  }
}

void Geometry::finalize()
{
  m_meshDirty = true;
  if (m_type == Type::Isosurface) {
    if (!m_field)
      reportMessage(ANARI_SEVERITY_WARNING, "missing required parameter 'field' on isosurface geometry");
    else if (!m_field->isValid())
      reportMessage(ANARI_SEVERITY_WARNING, "isosurface geometry needs a structuredRegular field");
    if (!m_hasIsovalue && !m_isovalueArray)
      reportMessage(ANARI_SEVERITY_WARNING, "missing required parameter 'isovalue' on isosurface geometry");
  } else if (m_type != Type::Unknown) {
    if (!m_vertexPosition) {
      reportMessage(ANARI_SEVERITY_WARNING,
          "missing required parameter 'vertex.position' on %s geometry",
          m_subtype.c_str());
    } else if (m_vertexPosition->elementType() != ANARI_FLOAT32_VEC3) {
      reportMessage(ANARI_SEVERITY_WARNING,
          "'vertex.position' on %s geometry must be an array of FLOAT32_VEC3",
          m_subtype.c_str());
    }
  }
  deviceState()->markSceneChanged();
}

const GeometryMesh &Geometry::mesh()
{
  // The field of an isosurface can change without the geometry being
  // committed again.
  if (m_type == Type::Isosurface) {
    m_mesh = GeometryMesh();
    if (m_field && m_field->isValid())
      buildIsosurface(m_mesh);
    return m_mesh;
  }

  if (m_meshDirty) {
    m_mesh = GeometryMesh();
    if (isFloat3Array(m_vertexPosition.get())) {
      switch (m_type) {
      case Type::Triangle:
        buildTriangles(m_mesh, 3);
        break;
      case Type::Quad:
        buildTriangles(m_mesh, 4);
        break;
      case Type::Sphere:
        buildSpheres(m_mesh);
        break;
      case Type::Cylinder:
      case Type::Cone:
      case Type::Curve:
        buildCylinders(m_mesh);
        break;
      default:
        break;
      }
    }
    m_meshDirty = false;
  }
  return m_mesh;
}

void Geometry::buildTriangles(GeometryMesh &mesh, int verticesPerFace) const
{
  const float3 *positions = m_vertexPosition->beginAs<float3>();
  const size_t vertexCount = m_vertexPosition->size();

  const Array1D *index = m_index.get();
  if (index && int(anari::componentsOf(index->elementType())) != verticesPerFace) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "'primitive.index' on %s geometry has the wrong element type, it is ignored",
        m_subtype.c_str());
    index = nullptr;
  }
  const size_t faceCount = index ? index->size() : vertexCount / size_t(verticesPerFace);

  auto vertexOf = [&](size_t face, int corner) -> size_t {
    return index ? size_t(arrayIndexComponent(*index, face, corner))
                 : face * size_t(verticesPerFace) + size_t(corner);
  };

  const Array1D *vertexNormal = isFloat3Array(m_vertexNormal.get()) && m_vertexNormal->size() >= vertexCount
      ? m_vertexNormal.get()
      : nullptr;
  const Array1D *faceNormal =
      isFloat3Array(m_faceVaryingNormal.get())
              && m_faceVaryingNormal->size() >= faceCount * size_t(verticesPerFace)
          ? m_faceVaryingNormal.get()
          : nullptr;

  const Array1D *vertexAttribute[bridge::kMeshAttributeCount];
  const Array1D *primitiveAttribute[bridge::kMeshAttributeCount];
  const Array1D *faceAttribute[bridge::kMeshAttributeCount];
  bool expand = faceNormal != nullptr;
  for (int slot = 0; slot < bridge::kMeshAttributeCount; ++slot) {
    const Array1D *v = m_vertexAttributes[slot].get();
    const Array1D *p = m_primitiveAttributes[slot].get();
    const Array1D *f = m_faceVaryingAttributes[slot].get();
    vertexAttribute[slot] = v && v->size() >= vertexCount ? v : nullptr;
    primitiveAttribute[slot] = p && p->size() >= faceCount ? p : nullptr;
    faceAttribute[slot] = f && f->size() >= faceCount * size_t(verticesPerFace) ? f : nullptr;
    if (primitiveAttribute[slot] || faceAttribute[slot])
      expand = true;
  }

  if (!expand) {
    // Shared vertices.
    mesh.positions.assign(positions, positions + vertexCount);
    if (vertexNormal)
      mesh.normals.assign(vertexNormal->beginAs<float3>(), vertexNormal->beginAs<float3>() + vertexCount);
    for (int slot = 0; slot < bridge::kMeshAttributeCount; ++slot) {
      if (!vertexAttribute[slot])
        continue;
      mesh.attributes[slot].resize(vertexCount);
      for (size_t i = 0; i < vertexCount; ++i)
        mesh.attributes[slot][i] = arrayElementAsFloat4(*vertexAttribute[slot], i);
    }
    mesh.indices.reserve(faceCount * size_t(verticesPerFace == 4 ? 6 : 3));
    for (size_t face = 0; face < faceCount; ++face) {
      size_t v[4] = {0, 0, 0, 0};
      bool valid = true;
      for (int k = 0; k < verticesPerFace; ++k) {
        v[k] = vertexOf(face, k);
        valid = valid && v[k] < vertexCount;
      }
      if (!valid)
        continue;
      mesh.indices.push_back(uint32_t(v[0]));
      mesh.indices.push_back(uint32_t(v[1]));
      mesh.indices.push_back(uint32_t(v[2]));
      if (verticesPerFace == 4) {
        mesh.indices.push_back(uint32_t(v[0]));
        mesh.indices.push_back(uint32_t(v[2]));
        mesh.indices.push_back(uint32_t(v[3]));
      }
    }
    return;
  }

  // Per-primitive or per-corner data: every face gets its own vertices.
  const bool hasNormals = vertexNormal || faceNormal;
  for (int slot = 0; slot < bridge::kMeshAttributeCount; ++slot) {
    if (vertexAttribute[slot] || primitiveAttribute[slot] || faceAttribute[slot])
      mesh.attributes[slot].reserve(faceCount * size_t(verticesPerFace));
  }
  for (size_t face = 0; face < faceCount; ++face) {
    size_t v[4] = {0, 0, 0, 0};
    bool valid = true;
    for (int k = 0; k < verticesPerFace; ++k) {
      v[k] = vertexOf(face, k);
      valid = valid && v[k] < vertexCount;
    }
    if (!valid)
      continue;

    const uint32_t base = uint32_t(mesh.positions.size());
    for (int k = 0; k < verticesPerFace; ++k) {
      const size_t corner = face * size_t(verticesPerFace) + size_t(k);
      mesh.positions.push_back(positions[v[k]]);
      if (hasNormals) {
        if (faceNormal)
          mesh.normals.push_back(faceNormal->beginAs<float3>()[corner]);
        else
          mesh.normals.push_back(vertexNormal->beginAs<float3>()[v[k]]);
      }
      for (int slot = 0; slot < bridge::kMeshAttributeCount; ++slot) {
        if (faceAttribute[slot])
          mesh.attributes[slot].push_back(arrayElementAsFloat4(*faceAttribute[slot], corner));
        else if (vertexAttribute[slot])
          mesh.attributes[slot].push_back(arrayElementAsFloat4(*vertexAttribute[slot], v[k]));
        else if (primitiveAttribute[slot])
          mesh.attributes[slot].push_back(arrayElementAsFloat4(*primitiveAttribute[slot], face));
      }
    }
    mesh.indices.push_back(base);
    mesh.indices.push_back(base + 1);
    mesh.indices.push_back(base + 2);
    if (verticesPerFace == 4) {
      mesh.indices.push_back(base);
      mesh.indices.push_back(base + 2);
      mesh.indices.push_back(base + 3);
    }
  }
}

void Geometry::buildSpheres(GeometryMesh &mesh) const
{
  const float3 *positions = m_vertexPosition->beginAs<float3>();
  const size_t vertexCount = m_vertexPosition->size();
  const Array1D *index = m_index.get();
  const size_t count = index ? index->size() : vertexCount;
  if (count == 0)
    return;

  const Array1D *radii =
      m_vertexRadius && m_vertexRadius->elementType() == ANARI_FLOAT32 && m_vertexRadius->size() >= vertexCount
          ? m_vertexRadius.get()
          : nullptr;

  // Photon has no sphere primitive: spheres are tessellated, with less detail
  // the more spheres there are.
  uint32_t slices = 48, stacks = 24;
  if (count > 400000) {
    slices = 4;
    stacks = 2;
  } else if (count > 120000) {
    slices = 8;
    stacks = 4;
  } else if (count > 30000) {
    slices = 10;
    stacks = 5;
  } else if (count > 4000) {
    slices = 16;
    stacks = 8;
  } else if (count > 256) {
    slices = 24;
    stacks = 12;
  }
  const SphereTemplate sphere(slices, stacks);
  const size_t vertsPerSphere = sphere.directions.size();
  if (count * vertsPerSphere > size_t(0xFFFFFFF0u)) {
    reportMessage(ANARI_SEVERITY_WARNING, "too many spheres (%zu), the geometry is not rendered", count);
    return;
  }

  bool hasAttribute[bridge::kMeshAttributeCount];
  for (int slot = 0; slot < bridge::kMeshAttributeCount; ++slot) {
    hasAttribute[slot] = m_vertexAttributes[slot] || m_primitiveAttributes[slot];
    if (hasAttribute[slot])
      mesh.attributes[slot].reserve(count * vertsPerSphere);
  }
  mesh.positions.reserve(count * vertsPerSphere);
  mesh.normals.reserve(count * vertsPerSphere);
  mesh.indices.reserve(count * sphere.indices.size());

  for (size_t i = 0; i < count; ++i) {
    const size_t v = index ? size_t(arrayIndexComponent(*index, i, 0)) : i;
    if (v >= vertexCount)
      continue;
    const float3 center = positions[v];
    const float radius = radii ? *radii->valueAt<float>(v) : m_radius;
    if (!(radius > 0.f))
      continue;

    const uint32_t base = uint32_t(mesh.positions.size());
    for (const float3 &d : sphere.directions) {
      mesh.positions.push_back(center + d * radius);
      mesh.normals.push_back(d);
    }
    for (uint32_t idx : sphere.indices)
      mesh.indices.push_back(base + idx);

    for (int slot = 0; slot < bridge::kMeshAttributeCount; ++slot) {
      if (!hasAttribute[slot])
        continue;
      float4 value(0.f, 0.f, 0.f, 1.f);
      const Array1D *va = m_vertexAttributes[slot].get();
      const Array1D *pa = m_primitiveAttributes[slot].get();
      if (va && v < va->size())
        value = arrayElementAsFloat4(*va, v);
      else if (pa && i < pa->size())
        value = arrayElementAsFloat4(*pa, i);
      mesh.attributes[slot].insert(mesh.attributes[slot].end(), vertsPerSphere, value);
    }
  }
}

void Geometry::buildCylinders(GeometryMesh &mesh) const
{
  const float3 *positions = m_vertexPosition->beginAs<float3>();
  const size_t vertexCount = m_vertexPosition->size();
  const bool curve = m_type == Type::Curve;

  const Array1D *index = m_index.get();
  if (index && int(anari::componentsOf(index->elementType())) != (curve ? 1 : 2)) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "'primitive.index' on %s geometry has the wrong element type, it is ignored",
        m_subtype.c_str());
    index = nullptr;
  }
  const size_t count = index ? index->size() : (curve ? (vertexCount > 0 ? vertexCount - 1 : 0) : vertexCount / 2);
  if (count == 0)
    return;

  const Array1D *vertexRadius =
      m_vertexRadius && m_vertexRadius->elementType() == ANARI_FLOAT32 && m_vertexRadius->size() >= vertexCount
          ? m_vertexRadius.get()
          : nullptr;
  const Array1D *primitiveRadius =
      m_primitiveRadius && m_primitiveRadius->elementType() == ANARI_FLOAT32 && m_primitiveRadius->size() >= count
          ? m_primitiveRadius.get()
          : nullptr;

  // Tessellated as (truncated) cones, with less detail the more there are.
  const uint32_t segments =
      count > 200000 ? 5 : (count > 20000 ? 8 : (count > 4000 ? 16 : (count > 256 ? 32 : 64)));
  // The segments of curves are closed by spheres at their vertices (below).
  const bool capFirst = !curve && m_capFirst;
  const bool capSecond = !curve && m_capSecond;
  std::vector<uint8_t> curveVertexUsed(curve ? vertexCount : 0, 0);

  bool hasAttribute[bridge::kMeshAttributeCount];
  for (int slot = 0; slot < bridge::kMeshAttributeCount; ++slot)
    hasAttribute[slot] = m_vertexAttributes[slot] || m_primitiveAttributes[slot];

  auto attributeAt = [&](int slot, size_t vertex, size_t primitive) {
    const Array1D *va = m_vertexAttributes[slot].get();
    const Array1D *pa = m_primitiveAttributes[slot].get();
    if (va && vertex < va->size())
      return arrayElementAsFloat4(*va, vertex);
    if (pa && primitive < pa->size())
      return arrayElementAsFloat4(*pa, primitive);
    return float4(0.f, 0.f, 0.f, 1.f);
  };

  for (size_t prim = 0; prim < count; ++prim) {
    size_t v0, v1;
    if (curve) {
      v0 = index ? size_t(arrayIndexComponent(*index, prim, 0)) : prim;
      v1 = v0 + 1;
    } else {
      v0 = index ? size_t(arrayIndexComponent(*index, prim, 0)) : 2 * prim;
      v1 = index ? size_t(arrayIndexComponent(*index, prim, 1)) : 2 * prim + 1;
    }
    if (v0 >= vertexCount || v1 >= vertexCount)
      continue;

    float r0 = m_radius, r1 = m_radius;
    if (m_type == Type::Cylinder) {
      if (primitiveRadius)
        r0 = r1 = *primitiveRadius->valueAt<float>(prim);
    } else if (vertexRadius) {
      r0 = *vertexRadius->valueAt<float>(v0);
      r1 = *vertexRadius->valueAt<float>(v1);
    }
    r0 = std::max(r0, 0.f);
    r1 = std::max(r1, 0.f);
    if (!(r0 > 0.f || r1 > 0.f))
      continue;

    if (curve) {
      curveVertexUsed[v0] = 1;
      curveVertexUsed[v1] = 1;
    }

    const float3 p0 = positions[v0];
    const float3 p1 = positions[v1];
    const float3 axis = p1 - p0;
    const float height = linalg::length(axis);
    if (!(height > 0.f))
      continue;
    const float3 dir = axis / height;
    const float3 helper = std::fabs(dir.x) < 0.9f ? float3(1.f, 0.f, 0.f) : float3(0.f, 1.f, 0.f);
    const float3 u = linalg::normalize(linalg::cross(dir, helper));
    const float3 w = linalg::cross(dir, u);

    float4 a0[bridge::kMeshAttributeCount], a1[bridge::kMeshAttributeCount];
    for (int slot = 0; slot < bridge::kMeshAttributeCount; ++slot) {
      if (hasAttribute[slot]) {
        a0[slot] = attributeAt(slot, v0, prim);
        a1[slot] = attributeAt(slot, v1, prim);
      }
    }
    auto pushVertex = [&](const float3 &p, const float3 &n, bool second) {
      mesh.positions.push_back(p);
      mesh.normals.push_back(n);
      for (int slot = 0; slot < bridge::kMeshAttributeCount; ++slot)
        if (hasAttribute[slot])
          mesh.attributes[slot].push_back(second ? a1[slot] : a0[slot]);
      return uint32_t(mesh.positions.size() - 1);
    };

    // Side.
    const uint32_t sideBase = uint32_t(mesh.positions.size());
    for (uint32_t j = 0; j < segments; ++j) {
      const float angle = 2.f * kPi * float(j) / float(segments);
      const float3 radial = u * std::cos(angle) + w * std::sin(angle);
      const float3 normal = linalg::normalize(radial * height + dir * (r0 - r1));
      pushVertex(p0 + radial * r0, normal, false);
      pushVertex(p1 + radial * r1, normal, true);
    }
    for (uint32_t j = 0; j < segments; ++j) {
      const uint32_t j1 = (j + 1) % segments;
      const uint32_t a = sideBase + 2 * j;
      const uint32_t b = sideBase + 2 * j + 1;
      const uint32_t c = sideBase + 2 * j1;
      const uint32_t d = sideBase + 2 * j1 + 1;
      const float3 outward = mesh.normals[a] + mesh.normals[c];
      if (r0 > 0.f)
        pushTriangle(mesh, a, c, b, outward);
      if (r1 > 0.f)
        pushTriangle(mesh, b, c, d, outward);
    }

    // Caps.
    for (int end = 0; end < 2; ++end) {
      const bool second = end == 1;
      const float r = second ? r1 : r0;
      if (!(second ? capSecond : capFirst) || !(r > 0.f))
        continue;
      const float3 center = second ? p1 : p0;
      const float3 normal = second ? dir : -dir;
      const uint32_t centerIndex = pushVertex(center, normal, second);
      const uint32_t ringBase = uint32_t(mesh.positions.size());
      for (uint32_t j = 0; j < segments; ++j) {
        const float angle = 2.f * kPi * float(j) / float(segments);
        pushVertex(center + (u * std::cos(angle) + w * std::sin(angle)) * r, normal, second);
      }
      for (uint32_t j = 0; j < segments; ++j)
        pushTriangle(mesh, centerIndex, ringBase + j, ringBase + (j + 1) % segments, normal);
    }
  }

  if (!curve)
    return;

  // Round joints and ends of curves: a sphere at every vertex in use.
  const SphereTemplate sphere(segments, std::max<uint32_t>(segments / 4, 2));
  for (size_t v = 0; v < vertexCount; ++v) {
    if (!curveVertexUsed[v])
      continue;
    const float radius = vertexRadius ? *vertexRadius->valueAt<float>(v) : m_radius;
    if (!(radius > 0.f))
      continue;
    const uint32_t base = uint32_t(mesh.positions.size());
    for (const float3 &d : sphere.directions) {
      mesh.positions.push_back(positions[v] + d * radius);
      mesh.normals.push_back(d);
    }
    for (uint32_t idx : sphere.indices)
      mesh.indices.push_back(base + idx);
    for (int slot = 0; slot < bridge::kMeshAttributeCount; ++slot) {
      if (hasAttribute[slot]) {
        // Primitive attributes are not defined at a joint, vertex ones are.
        const float4 value = attributeAt(slot, v, std::min<size_t>(v, count - 1));
        mesh.attributes[slot].insert(mesh.attributes[slot].end(), sphere.directions.size(), value);
      }
    }
  }
}

void Geometry::buildIsosurface(GeometryMesh &mesh) const
{
  // Photon has no implicit surfaces: the isosurfaces are extracted with
  // marching cubes.
  static const int triTable[256][16] = {
#include "photon/pt/geom/mc_table.inc"
  };
  static const int edgeCorners[12][2] = {
      {0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6}, {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
  static const int cornerOffset[8][3] = {
      {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};

  std::vector<float> values;
  uint3 dims(0u);
  float3 origin(0.f), spacing(1.f);
  m_field->grid(values, dims, origin, spacing);
  if (dims.x < 2 || dims.y < 2 || dims.z < 2 || values.size() < size_t(dims.x) * dims.y * dims.z)
    return;

  std::vector<float> isovalues;
  if (m_isovalueArray && m_isovalueArray->elementType() == ANARI_FLOAT32)
    isovalues.assign(m_isovalueArray->beginAs<float>(), m_isovalueArray->endAs<float>());
  else if (m_hasIsovalue)
    isovalues.push_back(m_isovalue);

  const int nx = int(dims.x), ny = int(dims.y), nz = int(dims.z);
  auto valueAt = [&](int x, int y, int z) {
    x = std::clamp(x, 0, nx - 1);
    y = std::clamp(y, 0, ny - 1);
    z = std::clamp(z, 0, nz - 1);
    return values[(size_t(z) * dims.y + size_t(y)) * dims.x + size_t(x)];
  };
  // The surface normal is the normalized gradient of the field.
  auto gradientAt = [&](int x, int y, int z) {
    return float3((valueAt(x + 1, y, z) - valueAt(x - 1, y, z)) / spacing.x,
        (valueAt(x, y + 1, z) - valueAt(x, y - 1, z)) / spacing.y,
        (valueAt(x, y, z + 1) - valueAt(x, y, z - 1)) / spacing.z);
  };

  for (const float isovalue : isovalues) {
    for (int z = 0; z + 1 < nz; ++z) {
      for (int y = 0; y + 1 < ny; ++y) {
        for (int x = 0; x + 1 < nx; ++x) {
          float cornerValue[8];
          int cubeIndex = 0;
          for (int c = 0; c < 8; ++c) {
            cornerValue[c] = valueAt(x + cornerOffset[c][0], y + cornerOffset[c][1], z + cornerOffset[c][2]);
            if (cornerValue[c] < isovalue)
              cubeIndex |= 1 << c;
          }
          if (cubeIndex == 0 || cubeIndex == 255)
            continue;

          for (int t = 0; triTable[cubeIndex][t] != -1; t += 3) {
            if (mesh.positions.size() > size_t(0xFFFFFFF0u))
              return;
            const uint32_t base = uint32_t(mesh.positions.size());
            for (int k = 0; k < 3; ++k) {
              const int *corners = edgeCorners[triTable[cubeIndex][t + k]];
              const int *a = cornerOffset[corners[0]];
              const int *b = cornerOffset[corners[1]];
              const float va = cornerValue[corners[0]];
              const float vb = cornerValue[corners[1]];
              const float w = va != vb ? std::clamp((isovalue - va) / (vb - va), 0.f, 1.f) : 0.5f;
              const float3 pa(float(x + a[0]), float(y + a[1]), float(z + a[2]));
              const float3 pb(float(x + b[0]), float(y + b[1]), float(z + b[2]));
              mesh.positions.push_back(origin + (pa + (pb - pa) * w) * spacing);
              const float3 ga = gradientAt(x + a[0], y + a[1], z + a[2]);
              const float3 gb = gradientAt(x + b[0], y + b[1], z + b[2]);
              const float3 g = ga + (gb - ga) * w;
              const float len = linalg::length(g);
              mesh.normals.push_back(len > 0.f ? g / len : float3(0.f));
            }
            mesh.indices.push_back(base);
            mesh.indices.push_back(base + 1);
            mesh.indices.push_back(base + 2);
          }
        }
      }
    }
  }

  for (int slot = 0; slot < bridge::kMeshAttributeCount; ++slot) {
    if (m_hasUniformAttribute[slot])
      mesh.attributes[slot].assign(mesh.positions.size(), m_uniformAttributes[slot]);
  }
}

} // namespace photon::anari_device
