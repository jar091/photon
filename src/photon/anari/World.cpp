#include "photon/anari/SceneObjects.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace photon::anari_device {

namespace {

// Collects the triangle mesh, materials, lights and volumes of a world.
struct SceneBuilder
{
  bridge::SceneData &scene;
  const Object &reporter;
  SamplerRegistry samplers;
  std::map<const Material *, uint32_t> materials;
  bool hasDefaultMaterial{false};
  uint32_t defaultMaterial{0};
  bool attributeUsed[bridge::kMeshAttributeCount]{};
  bool warnedEnvironment{false};
  float3 lower{std::numeric_limits<float>::max()};
  float3 upper{-std::numeric_limits<float>::max()};

  SceneBuilder(bridge::SceneData &s, const Object &r) : scene(s), reporter(r)
  {
    samplers.samplers = &scene.samplers;
  }

  size_t vertexCount() const
  {
    return scene.positions.size() / 3;
  }

  void extendBounds(const float3 &p)
  {
    lower = linalg::min(lower, p);
    upper = linalg::max(upper, p);
  }

  uint32_t materialIndex(const Material *material)
  {
    if (!material) {
      if (!hasDefaultMaterial) {
        defaultMaterial = uint32_t(scene.materials.size());
        scene.materials.emplace_back();
        scene.materials.back().roughness = 1.f;
        hasDefaultMaterial = true;
      }
      return defaultMaterial;
    }
    auto it = materials.find(material);
    if (it != materials.end())
      return it->second;
    const uint32_t index = uint32_t(scene.materials.size());
    // The samplers have to be registered before the material is stored: the
    // export may add samplers.
    const bridge::MaterialData data = material->exportData(samplers);
    scene.materials.push_back(data);
    materials[material] = index;
    return index;
  }

  // Marks an attribute slot as used and gives all vertices so far a default.
  void useAttribute(int slot)
  {
    if (attributeUsed[slot])
      return;
    attributeUsed[slot] = true;
    const float defaults[4] = {0.f, 0.f, 0.f, 1.f};
    auto &values = scene.attributes[slot];
    values.resize(vertexCount() * 4);
    for (size_t i = 0; i < values.size(); i += 4)
      std::memcpy(&values[i], defaults, sizeof(defaults));
  }

  void useNormals()
  {
    // Vertices without normals get a zero normal, they are shaded flat.
    if (scene.normals.size() < scene.positions.size())
      scene.normals.resize(scene.positions.size(), 0.f);
  }

  void addSurface(Surface *surface, const Instance *instance, const mat4 &xfm)
  {
    if (!surface || !surface->isValid())
      return;
    const GeometryMesh &mesh = surface->geometry()->mesh();
    if (mesh.indices.empty() || mesh.positions.empty())
      return;

    const size_t base = vertexCount();
    const size_t count = mesh.positions.size();
    if (base + count > size_t(0xFFFFFFF0u)) {
      reporter.reportMessage(ANARI_SEVERITY_WARNING, "too many vertices in the world, a surface is not rendered");
      return;
    }

    const Material *material = surface->material();
    const uint32_t materialId = materialIndex(material);
    const uint32_t used = material ? material->usedAttributes() : 0u;

    scene.positions.resize((base + count) * 3);
    for (size_t i = 0; i < count; ++i) {
      const float4 p = linalg::mul(xfm, float4(mesh.positions[i], 1.f));
      scene.positions[(base + i) * 3 + 0] = p.x;
      scene.positions[(base + i) * 3 + 1] = p.y;
      scene.positions[(base + i) * 3 + 2] = p.z;
      extendBounds(float3(p.x, p.y, p.z));
    }

    if (!mesh.normals.empty()) {
      useNormals();
      // Normals transform with the inverse transpose.
      const anari::math::float3x3 linear(
          float3(xfm[0].x, xfm[0].y, xfm[0].z), float3(xfm[1].x, xfm[1].y, xfm[1].z), float3(xfm[2].x, xfm[2].y, xfm[2].z));
      const float det = linalg::determinant(linear);
      const anari::math::float3x3 normalXfm =
          std::fabs(det) > 1e-20f ? linalg::transpose(linalg::inverse(linear)) : linear;
      scene.normals.resize((base + count) * 3);
      for (size_t i = 0; i < count; ++i) {
        float3 n = linalg::mul(normalXfm, mesh.normals[i]);
        const float len = linalg::length(n);
        n = len > 0.f ? n / len : float3(0.f);
        scene.normals[(base + i) * 3 + 0] = n.x;
        scene.normals[(base + i) * 3 + 1] = n.y;
        scene.normals[(base + i) * 3 + 2] = n.z;
      }
    }

    for (int slot = 0; slot < bridge::kMeshAttributeCount; ++slot) {
      if (!(used & (1u << slot)))
        continue;
      useAttribute(slot);
      auto &values = scene.attributes[slot];
      values.resize((base + count) * 4);
      if (!mesh.attributes[slot].empty()) {
        std::memcpy(&values[base * 4], mesh.attributes[slot].data(), count * 4 * sizeof(float));
      } else {
        // Attributes the geometry does not have come from the instance.
        float4 value(0.f, 0.f, 0.f, 1.f);
        if (instance)
          instance->attribute(slot, value);
        for (size_t i = 0; i < count; ++i)
          std::memcpy(&values[(base + i) * 4], &value, sizeof(value));
      }
    }

    const size_t firstIndex = scene.indices.size();
    scene.indices.resize(firstIndex + mesh.indices.size());
    for (size_t i = 0; i < mesh.indices.size(); ++i)
      scene.indices[firstIndex + i] = uint32_t(base) + mesh.indices[i];
    scene.material_ids.insert(scene.material_ids.end(), mesh.indices.size() / 3, materialId);
  }

  // Quad lights are emissive geometry: they are visible to rays and sampled
  // like any other emissive surface.
  void addQuadLight(const Light &light, const mat4 &xfm)
  {
    float3 corner, edge1, edge2, radiance;
    int side = 0;
    light.exportQuad(xfm, corner, edge1, edge2, radiance, side);
    if (!(linalg::length(linalg::cross(edge1, edge2)) > 0.f))
      return;

    bridge::MaterialData material;
    for (int i = 0; i < 3; ++i) {
      material.base_color[i] = 0.f;
      material.emission[i] = radiance[i];
    }
    material.emitter_only = true;
    material.hidden_to_camera = !light.visible();
    material.emission_side = side;
    const uint32_t materialId = uint32_t(scene.materials.size());
    scene.materials.push_back(material);

    const uint32_t base = uint32_t(vertexCount());
    const float3 corners[4] = {corner, corner + edge1, corner + edge1 + edge2, corner + edge2};
    for (const float3 &p : corners) {
      scene.positions.push_back(p.x);
      scene.positions.push_back(p.y);
      scene.positions.push_back(p.z);
    }
    const uint32_t indices[6] = {base, base + 1, base + 2, base, base + 2, base + 3};
    scene.indices.insert(scene.indices.end(), indices, indices + 6);
    scene.material_ids.push_back(materialId);
    scene.material_ids.push_back(materialId);
  }

  void addLight(Light *light, const mat4 &xfm)
  {
    if (!light || !light->isValid())
      return;
    switch (light->lightType()) {
    case Light::Type::Quad:
      addQuadLight(*light, xfm);
      break;
    case Light::Type::HDRI:
      if (scene.environment.present) {
        if (!warnedEnvironment) {
          reporter.reportMessage(
              ANARI_SEVERITY_WARNING, "photon supports one hdri light per world, the others are ignored");
          warnedEnvironment = true;
        }
      } else {
        light->exportEnvironment(xfm, scene.environment);
      }
      break;
    default:
      scene.lights.push_back(light->exportData(xfm));
      break;
    }
  }

  void addVolume(Volume *volume, const mat4 &xfm)
  {
    if (!volume || !volume->isValid())
      return;
    bridge::VolumeData data;
    volume->exportData(data);

    const float det = linalg::determinant(xfm);
    const mat4 inverse = std::fabs(det) > 1e-20f ? linalg::inverse(xfm) : mat4(linalg::identity);
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 4; ++c)
        data.world_to_local[r * 4 + c] = inverse[c][r];

    float3 lo, hi;
    volume->field()->bounds(lo, hi);
    for (int corner = 0; corner < 8; ++corner) {
      const float3 p((corner & 1) ? hi.x : lo.x, (corner & 2) ? hi.y : lo.y, (corner & 4) ? hi.z : lo.z);
      const float4 w = linalg::mul(xfm, float4(p, 1.f));
      extendBounds(float3(w.x, w.y, w.z));
    }
    scene.volumes.push_back(std::move(data));
  }

  template <typename T, typename Fn>
  static void forEachObject(const ObjectArray *array, Fn &&fn)
  {
    if (!array)
      return;
    std::for_each(array->handlesBegin(), array->handlesEnd(), [&](helium::BaseObject *o) {
      if (o)
        fn(static_cast<T *>(o));
    });
  }

  void addGroupContent(const ObjectArray *surfaces,
      const ObjectArray *volumes,
      const ObjectArray *lights,
      const Instance *instance,
      const mat4 &xfm)
  {
    forEachObject<Surface>(surfaces, [&](Surface *s) { addSurface(s, instance, xfm); });
    forEachObject<Volume>(volumes, [&](Volume *v) { addVolume(v, xfm); });
    forEachObject<Light>(lights, [&](Light *l) { addLight(l, xfm); });
  }

  void finish()
  {
    // Every used per-vertex array has to cover all vertices.
    const size_t count = vertexCount();
    if (!scene.normals.empty())
      scene.normals.resize(count * 3, 0.f);
    for (int slot = 0; slot < bridge::kMeshAttributeCount; ++slot) {
      if (!attributeUsed[slot])
        continue;
      auto &values = scene.attributes[slot];
      const size_t old = values.size();
      values.resize(count * 4);
      const float defaults[4] = {0.f, 0.f, 0.f, 1.f};
      for (size_t i = old; i < values.size(); i += 4)
        std::memcpy(&values[i], defaults, sizeof(defaults));
    }
  }
};

} // namespace

World::World(PhotonGlobalState *s)
    : Object(ANARI_WORLD, s), m_surfaces(this), m_volumes(this), m_lights(this), m_instances(this)
{}

bool World::getProperty(const std::string_view &name, ANARIDataType type, void *ptr, uint64_t size, uint32_t flags)
{
  if (name == "bounds" && type == ANARI_FLOAT32_BOX3) {
    if (m_boundsBuilt <= deviceState()->sceneLastChanged) {
      bridge::SceneData scene;
      buildScene(scene);
    }
    if (!m_boundsValid)
      return false;
    const float bounds[6] = {m_boundsLower.x, m_boundsLower.y, m_boundsLower.z, m_boundsUpper.x, m_boundsUpper.y,
        m_boundsUpper.z};
    std::memcpy(ptr, bounds, sizeof(bounds));
    return true;
  }
  return Object::getProperty(name, type, ptr, size, flags);
}

void World::commitParameters()
{
  m_surfaces = getParamObject<ObjectArray>("surface");
  m_volumes = getParamObject<ObjectArray>("volume");
  m_lights = getParamObject<ObjectArray>("light");
  m_instances = getParamObject<ObjectArray>("instance");
}

void World::finalize()
{
  deviceState()->markSceneChanged();
}

void World::buildScene(bridge::SceneData &scene)
{
  scene = bridge::SceneData();
  SceneBuilder builder(scene, *this);
  const mat4 identity = linalg::identity;

  builder.addGroupContent(m_surfaces.get(), m_volumes.get(), m_lights.get(), nullptr, identity);

  SceneBuilder::forEachObject<Instance>(m_instances.get(), [&](Instance *instance) {
    if (!instance->isValid())
      return;
    const Group *group = instance->group();
    builder.addGroupContent(group->surfaces(), group->volumes(), group->lights(), instance, instance->transform());
  });

  builder.finish();

  m_boundsValid = builder.lower.x <= builder.upper.x;
  if (m_boundsValid) {
    m_boundsLower = builder.lower;
    m_boundsUpper = builder.upper;
  }
  m_boundsBuilt = helium::newTimeStamp();
}

} // namespace photon::anari_device
