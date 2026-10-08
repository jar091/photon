#pragma once

#include "photon/anari/Object.h"
#include "photon/anari/RenderBridge.h"

#include <array>
#include <map>
#include <vector>

namespace photon::anari_device {

// Camera /////////////////////////////////////////////////////////////////////

struct Camera : public Object
{
  Camera(PhotonGlobalState *s, std::string_view subtype);

  void commitParameters() override;

  bridge::CameraData data() const;

 private:
  bool m_orthographic{false};
  bridge::CameraData m_data;
};

// Renderer ///////////////////////////////////////////////////////////////////

struct Renderer : public Object
{
  Renderer(PhotonGlobalState *s, std::string_view subtype);

  void commitParameters() override;
  void finalize() override;

  // The background image pointer stays valid until the next commit.
  void fillSettings(bridge::RenderSettings &settings) const;

 private:
  bridge::RenderSettings m_settings;
  helium::ChangeObserverPtr<Array2D> m_backgroundImage;
  std::vector<float> m_backgroundPixels;
};

// Sampler ////////////////////////////////////////////////////////////////////

struct Sampler : public Object
{
  Sampler(PhotonGlobalState *s, std::string_view subtype);

  void commitParameters() override;
  void finalize() override;
  bool isValid() const override;

  int inAttribute() const;
  void exportData(bridge::SamplerData &out) const;

 private:
  int m_type{-1}; // bridge sampler type, -1: unsupported subtype
  bridge::SamplerData m_data;
  helium::ChangeObserverPtr<Array1D> m_image1D;
  helium::ChangeObserverPtr<Array2D> m_image2D;
  helium::ChangeObserverPtr<Array3D> m_image3D;
};

// Material ///////////////////////////////////////////////////////////////////

// Maps the samplers used by the materials of a scene to their index.
struct SamplerRegistry
{
  std::map<const Sampler *, int> indices;
  std::vector<bridge::SamplerData> *samplers{nullptr};
  int add(const Sampler *sampler);
};

struct MaterialParameter
{
  float4 value{0.f, 0.f, 0.f, 1.f};
  int attribute{-1};
  helium::IntrusivePtr<Sampler> sampler;
};

struct Material : public Object
{
  Material(PhotonGlobalState *s, std::string_view subtype);

  void commitParameters() override;
  void finalize() override;

  bridge::MaterialData exportData(SamplerRegistry &registry) const;
  // Bit mask of the mesh attribute slots the material reads.
  uint32_t usedAttributes() const;

 private:
  MaterialParameter readParameter(const char *name, const float4 &defaultValue);
  bridge::ParamSourceData exportSource(const MaterialParameter &p, SamplerRegistry &registry) const;

  bool m_physicallyBased{false};
  MaterialParameter m_color;
  MaterialParameter m_opacity;
  MaterialParameter m_metallic;
  MaterialParameter m_roughness;
  MaterialParameter m_emissive;
  MaterialParameter m_transmission;
  float m_ior{1.5f};
  float m_specular{0.f};
  float m_clearcoat{0.f};
  float m_clearcoatRoughness{0.f};
  float m_thickness{0.f};
  int m_alphaMode{1};
  float m_alphaCutoff{0.5f};
};

// SpatialField ///////////////////////////////////////////////////////////////

struct SpatialField : public Object
{
  SpatialField(PhotonGlobalState *s, std::string_view subtype);

  void commitParameters() override;
  void finalize() override;
  bool isValid() const override;

  // Fills grid, origin, spacing and filter of `out`.
  void exportData(bridge::VolumeData &out) const;
  void bounds(float3 &lower, float3 &upper) const;

  // Values at the grid points, x runs fastest.
  void grid(std::vector<float> &values, uint3 &dims, float3 &origin, float3 &spacing) const;

 private:
  bool m_supported{false};
  helium::ChangeObserverPtr<Array3D> m_data;
  float3 m_origin{0.f, 0.f, 0.f};
  float3 m_spacing{1.f, 1.f, 1.f};
  bool m_linear{true};
};

// Geometry ///////////////////////////////////////////////////////////////////

// Triangle mesh in the object space of a geometry.
struct GeometryMesh
{
  std::vector<float3> positions;
  std::vector<float3> normals; // empty or one per vertex
  std::vector<uint32_t> indices;
  std::vector<float4> attributes[bridge::kMeshAttributeCount]; // empty or one per vertex
};

struct Geometry : public Object
{
  enum class Type { Triangle, Quad, Sphere, Cylinder, Cone, Curve, Isosurface, Unknown };

  Geometry(PhotonGlobalState *s, std::string_view subtype);

  void commitParameters() override;
  void finalize() override;

  // Triangles of the geometry, primitives without a triangle representation
  // in Photon are tessellated.
  const GeometryMesh &mesh();

 private:
  void buildTriangles(GeometryMesh &mesh, int verticesPerFace) const;
  void buildSpheres(GeometryMesh &mesh) const;
  void buildCylinders(GeometryMesh &mesh) const;
  void buildIsosurface(GeometryMesh &mesh) const;

  Type m_type{Type::Unknown};
  std::string m_subtype;

  helium::ChangeObserverPtr<Array1D> m_index;
  helium::ChangeObserverPtr<Array1D> m_vertexPosition;
  helium::ChangeObserverPtr<Array1D> m_vertexNormal;
  helium::ChangeObserverPtr<Array1D> m_vertexRadius;
  helium::ChangeObserverPtr<Array1D> m_primitiveRadius;
  helium::ChangeObserverPtr<Array1D> m_faceVaryingNormal;
  std::array<helium::ChangeObserverPtr<Array1D>, bridge::kMeshAttributeCount> m_vertexAttributes;
  std::array<helium::ChangeObserverPtr<Array1D>, bridge::kMeshAttributeCount> m_primitiveAttributes;
  std::array<helium::ChangeObserverPtr<Array1D>, bridge::kMeshAttributeCount> m_faceVaryingAttributes;
  float m_radius{1.f};
  bool m_capFirst{false};
  bool m_capSecond{false};

  // Isosurface: field, isovalues and uniform attributes.
  helium::IntrusivePtr<SpatialField> m_field;
  helium::ChangeObserverPtr<Array1D> m_isovalueArray;
  float m_isovalue{0.f};
  bool m_hasIsovalue{false};
  float4 m_uniformAttributes[bridge::kMeshAttributeCount];
  bool m_hasUniformAttribute[bridge::kMeshAttributeCount]{};

  GeometryMesh m_mesh;
  bool m_meshDirty{true};
};

// Surface ////////////////////////////////////////////////////////////////////

struct Surface : public Object
{
  Surface(PhotonGlobalState *s);

  void commitParameters() override;
  void finalize() override;
  bool isValid() const override;

  Geometry *geometry() const;
  Material *material() const;

 private:
  helium::IntrusivePtr<Geometry> m_geometry;
  helium::IntrusivePtr<Material> m_material;
  bool m_visible{true};
};

// Volume /////////////////////////////////////////////////////////////////////

struct Volume : public Object
{
  Volume(PhotonGlobalState *s, std::string_view subtype);

  void commitParameters() override;
  void finalize() override;
  bool isValid() const override;

  const SpatialField *field() const;
  // `out` in the space of the volume, the caller sets world_to_local.
  void exportData(bridge::VolumeData &out) const;

 private:
  bool m_supported{false};
  helium::IntrusivePtr<SpatialField> m_field;
  helium::ChangeObserverPtr<Array1D> m_colorData;
  helium::ChangeObserverPtr<Array1D> m_opacityData;
  float4 m_color{1.f, 1.f, 1.f, 1.f};
  float m_opacity{1.f};
  float2 m_valueRange{0.f, 1.f};
  float m_unitDistance{1.f};
};

// Light //////////////////////////////////////////////////////////////////////

struct Light : public Object
{
  enum class Type { Directional, Point, Spot, Quad, HDRI, Unknown };

  Light(PhotonGlobalState *s, std::string_view subtype);

  void commitParameters() override;
  void finalize() override;
  bool isValid() const override;

  Type lightType() const;
  bool visible() const;

  // Point, spot and directional lights.
  bridge::LightData exportData(const mat4 &xfm) const;
  // Quad lights: corner, edges, radiance and the emitting side (0 = front,
  // 1 = back, 2 = both).
  void exportQuad(const mat4 &xfm, float3 &corner, float3 &edge1, float3 &edge2, float3 &radiance, int &side) const;
  // HDRI lights.
  bool exportEnvironment(const mat4 &xfm, bridge::EnvironmentData &out) const;

 private:
  Type m_type{Type::Unknown};
  float3 m_color{1.f, 1.f, 1.f};
  float3 m_position{0.f, 0.f, 0.f};
  float3 m_direction{0.f, 0.f, -1.f};
  float3 m_up{0.f, 0.f, 1.f};
  float3 m_edge1{1.f, 0.f, 0.f};
  float3 m_edge2{0.f, 1.f, 0.f};
  float m_intensity{1.f}; // W/sr, W/m^2 or W/sr/m^2 depending on the type
  float m_radius{0.f};
  float m_openingAngle{3.14159265f};
  float m_falloffAngle{0.1f};
  float m_scale{1.f};
  int m_side{0};
  bool m_visible{true};
  helium::ChangeObserverPtr<Array2D> m_radiance;
};

// Group, Instance and World //////////////////////////////////////////////////

struct Group : public Object
{
  Group(PhotonGlobalState *s);

  void commitParameters() override;
  void finalize() override;

  const ObjectArray *surfaces() const;
  const ObjectArray *volumes() const;
  const ObjectArray *lights() const;

 private:
  helium::ChangeObserverPtr<ObjectArray> m_surfaces;
  helium::ChangeObserverPtr<ObjectArray> m_volumes;
  helium::ChangeObserverPtr<ObjectArray> m_lights;
};

struct Instance : public Object
{
  Instance(PhotonGlobalState *s, std::string_view subtype);

  void commitParameters() override;
  void finalize() override;
  bool isValid() const override;

  const Group *group() const;
  const mat4 &transform() const;
  // Uniform attribute of the instance, used when the geometry has none.
  bool attribute(int slot, float4 &value) const;

 private:
  helium::IntrusivePtr<Group> m_group;
  mat4 m_transform;
  float4 m_attributes[bridge::kMeshAttributeCount];
  bool m_hasAttribute[bridge::kMeshAttributeCount]{};
};

struct World : public Object
{
  World(PhotonGlobalState *s);

  bool getProperty(const std::string_view &name,
      ANARIDataType type,
      void *ptr,
      uint64_t size,
      uint32_t flags) override;

  void commitParameters() override;
  void finalize() override;

  // Converts everything the world references to the scene of the path tracer.
  void buildScene(bridge::SceneData &scene);

 private:
  helium::ChangeObserverPtr<ObjectArray> m_surfaces;
  helium::ChangeObserverPtr<ObjectArray> m_volumes;
  helium::ChangeObserverPtr<ObjectArray> m_lights;
  helium::ChangeObserverPtr<ObjectArray> m_instances;

  float3 m_boundsLower{0.f, 0.f, 0.f};
  float3 m_boundsUpper{0.f, 0.f, 0.f};
  bool m_boundsValid{false};
  helium::TimeStamp m_boundsBuilt{0};
};

} // namespace photon::anari_device

PHOTON_ANARI_TYPEFOR_SPECIALIZATION(photon::anari_device::Camera *, ANARI_CAMERA);
PHOTON_ANARI_TYPEFOR_SPECIALIZATION(photon::anari_device::Renderer *, ANARI_RENDERER);
PHOTON_ANARI_TYPEFOR_SPECIALIZATION(photon::anari_device::Sampler *, ANARI_SAMPLER);
PHOTON_ANARI_TYPEFOR_SPECIALIZATION(photon::anari_device::Material *, ANARI_MATERIAL);
PHOTON_ANARI_TYPEFOR_SPECIALIZATION(photon::anari_device::Geometry *, ANARI_GEOMETRY);
PHOTON_ANARI_TYPEFOR_SPECIALIZATION(photon::anari_device::Surface *, ANARI_SURFACE);
PHOTON_ANARI_TYPEFOR_SPECIALIZATION(photon::anari_device::SpatialField *, ANARI_SPATIAL_FIELD);
PHOTON_ANARI_TYPEFOR_SPECIALIZATION(photon::anari_device::Volume *, ANARI_VOLUME);
PHOTON_ANARI_TYPEFOR_SPECIALIZATION(photon::anari_device::Light *, ANARI_LIGHT);
PHOTON_ANARI_TYPEFOR_SPECIALIZATION(photon::anari_device::Group *, ANARI_GROUP);
PHOTON_ANARI_TYPEFOR_SPECIALIZATION(photon::anari_device::Instance *, ANARI_INSTANCE);
PHOTON_ANARI_TYPEFOR_SPECIALIZATION(photon::anari_device::World *, ANARI_WORLD);
