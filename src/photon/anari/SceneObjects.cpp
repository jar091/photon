#include "photon/anari/SceneObjects.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <initializer_list>

namespace photon::anari_device {

namespace {

constexpr float kPi = 3.14159265358979323846f;

// Mesh attribute slot of an ANARI attribute name, -1 for unknown names.
int attributeSlot(const std::string &name)
{
  if (name == "attribute0")
    return 0;
  if (name == "attribute1")
    return 1;
  if (name == "attribute2")
    return 2;
  if (name == "attribute3")
    return 3;
  if (name == "color")
    return bridge::kAttributeColor;
  // The mesh is stored in world space, object space attributes use the world
  // space values.
  if (name == "worldPosition" || name == "objectPosition")
    return bridge::kAttributeWorldPosition;
  if (name == "worldNormal" || name == "objectNormal")
    return bridge::kAttributeWorldNormal;
  return -1;
}

// Column-major helium matrix to a row-major array.
void toRowMajor(const mat4 &m, float *out)
{
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c)
      out[r * 4 + c] = m[c][r];
}

int wrapModeFromString(const std::string &mode)
{
  if (mode == "repeat")
    return 1;
  if (mode == "mirrorRepeat")
    return 2;
  return 0;
}

float3 transformPoint(const mat4 &m, const float3 &p)
{
  const float4 r = linalg::mul(m, float4(p, 1.f));
  return float3(r.x, r.y, r.z);
}

float3 transformVector(const mat4 &m, const float3 &v)
{
  const float4 r = linalg::mul(m, float4(v, 0.f));
  return float3(r.x, r.y, r.z);
}

float3 safeNormalize(const float3 &v, const float3 &fallback)
{
  const float len = linalg::length(v);
  return len > 0.f ? v / len : fallback;
}

} // namespace

// Camera /////////////////////////////////////////////////////////////////////

Camera::Camera(PhotonGlobalState *s, std::string_view subtype) : Object(ANARI_CAMERA, s)
{
  m_orthographic = subtype == "orthographic";
  if (subtype != "perspective" && subtype != "orthographic") {
    reportMessage(ANARI_SEVERITY_WARNING,
        "photon does not support camera subtype '%s', using a perspective camera",
        std::string(subtype).c_str());
  }
  m_data.orthographic = m_orthographic;
}

void Camera::commitParameters()
{
  const float3 position = getParam<float3>("position", float3(0.f, 0.f, 0.f));
  const float3 direction = getParam<float3>("direction", float3(0.f, 0.f, -1.f));
  const float3 up = getParam<float3>("up", float3(0.f, 1.f, 0.f));
  for (int i = 0; i < 3; ++i) {
    m_data.position[i] = position[i];
    m_data.direction[i] = direction[i];
    m_data.up[i] = up[i];
  }
  m_data.orthographic = m_orthographic;
  m_data.fovy = getParam<float>("fovy", kPi / 3.f);
  m_data.aspect = getParam<float>("aspect", 1.f);
  m_data.height = getParam<float>("height", 1.f);
  m_data.aperture_radius = getParam<float>("apertureRadius", 0.f);
  m_data.focus_distance = getParam<float>("focusDistance", 1.f);

  float4 region(0.f, 0.f, 1.f, 1.f);
  getParam("imageRegion", ANARI_FLOAT32_BOX2, &region);
  for (int i = 0; i < 4; ++i)
    m_data.image_region[i] = region[i];
}

bridge::CameraData Camera::data() const
{
  return m_data;
}

// Renderer ///////////////////////////////////////////////////////////////////

Renderer::Renderer(PhotonGlobalState *s, std::string_view subtype)
    : Object(ANARI_RENDERER, s), m_backgroundImage(this)
{
  if (subtype != "default") {
    reportMessage(ANARI_SEVERITY_INFO,
        "photon has no renderer subtype '%s', using 'default'",
        std::string(subtype).c_str());
  }
}

void Renderer::commitParameters()
{
  float4 background(0.f, 0.f, 0.f, 1.f);
  float3 background3(0.f, 0.f, 0.f);
  if (getParam("background", ANARI_FLOAT32_VEC3, &background3))
    background = float4(background3, 1.f);
  getParam("background", ANARI_FLOAT32_VEC4, &background);
  for (int i = 0; i < 4; ++i)
    m_settings.background[i] = background[i];
  m_backgroundImage = getParamObject<Array2D>("background");

  const float3 ambientColor = getParam<float3>("ambientColor", float3(1.f, 1.f, 1.f));
  for (int i = 0; i < 3; ++i)
    m_settings.ambient_color[i] = ambientColor[i];
  m_settings.ambient_radiance = std::max(getParam<float>("ambientRadiance", 0.f), 0.f);

  m_settings.pixel_samples = uint32_t(std::clamp(getParam<int>("pixelSamples", 1), 1, 65536));
  // 'maxRayDepth' counts the indirect bounces, the path tracer the scattering
  // events that receive direct light.
  m_settings.max_depth = uint32_t(std::clamp(getParam<int>("maxRayDepth", 8), 0, 256)) + 1;
  m_settings.throughput_clamp = getParam<float>("throughputClamp", 0.f);
  m_settings.adaptive_sampling = getParam<bool>("adaptiveSampling", false);
}

void Renderer::finalize()
{
  m_backgroundPixels.clear();
  if (m_backgroundImage) {
    const size_t count = m_backgroundImage->totalSize();
    m_backgroundPixels.resize(count * 4);
    for (size_t i = 0; i < count; ++i) {
      const float4 c = arrayElementAsFloat4(*m_backgroundImage, i);
      std::memcpy(&m_backgroundPixels[i * 4], &c, sizeof(c));
    }
    m_settings.background_width = uint32_t(m_backgroundImage->size(0));
    m_settings.background_height = uint32_t(m_backgroundImage->size(1));
  } else {
    m_settings.background_width = 0;
    m_settings.background_height = 0;
  }
}

void Renderer::fillSettings(bridge::RenderSettings &settings) const
{
  const uint32_t width = settings.width;
  const uint32_t height = settings.height;
  settings = m_settings;
  settings.width = width;
  settings.height = height;
  settings.background_image = m_backgroundPixels.empty() ? nullptr : m_backgroundPixels.data();
}

// Sampler ////////////////////////////////////////////////////////////////////

Sampler::Sampler(PhotonGlobalState *s, std::string_view subtype)
    : Object(ANARI_SAMPLER, s), m_image1D(this), m_image2D(this), m_image3D(this)
{
  if (subtype == "image1D")
    m_type = 0;
  else if (subtype == "image2D")
    m_type = 1;
  else if (subtype == "image3D")
    m_type = 2;
  else if (subtype == "transform")
    m_type = 3;
  else {
    reportMessage(ANARI_SEVERITY_WARNING,
        "photon does not support sampler subtype '%s', the sampler is ignored",
        std::string(subtype).c_str());
  }
}

void Sampler::commitParameters()
{
  m_data = bridge::SamplerData();
  m_data.type = m_type < 0 ? 3 : m_type;

  const std::string inAttribute = getParamString("inAttribute", "attribute0");
  int slot = attributeSlot(inAttribute);
  if (slot < 0) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "unknown sampler inAttribute '%s', using 'attribute0'",
        inAttribute.c_str());
    slot = 0;
  }
  m_data.in_attribute = slot;

  m_data.linear = getParamString("filter", "linear") != "nearest";
  m_data.wrap[0] = wrapModeFromString(getParamString("wrapMode1", getParamString("wrapMode", "clampToEdge")));
  m_data.wrap[1] = wrapModeFromString(getParamString("wrapMode2", "clampToEdge"));
  m_data.wrap[2] = wrapModeFromString(getParamString("wrapMode3", "clampToEdge"));

  const mat4 identity = linalg::identity;
  toRowMajor(getParam<mat4>("inTransform", identity), m_data.in_transform);
  const float4 inOffset = getParam<float4>("inOffset", float4(0.f, 0.f, 0.f, 0.f));
  mat4 outTransform = getParam<mat4>("outTransform", identity);
  if (m_type == 3)
    outTransform = getParam<mat4>("transform", outTransform);
  toRowMajor(outTransform, m_data.out_transform);
  const float4 outOffset = getParam<float4>("outOffset", float4(0.f, 0.f, 0.f, 0.f));
  for (int i = 0; i < 4; ++i) {
    m_data.in_offset[i] = inOffset[i];
    m_data.out_offset[i] = outOffset[i];
  }

  m_image1D = m_type == 0 ? getParamObject<Array1D>("image") : nullptr;
  m_image2D = m_type == 1 ? getParamObject<Array2D>("image") : nullptr;
  m_image3D = m_type == 2 ? getParamObject<Array3D>("image") : nullptr;
}

void Sampler::finalize()
{
  m_data.texels.clear();
  const helium::Array *image = nullptr;
  if (m_type == 0 && m_image1D) {
    image = m_image1D.get();
    m_data.size[0] = uint32_t(m_image1D->size());
  } else if (m_type == 1 && m_image2D) {
    image = m_image2D.get();
    m_data.size[0] = uint32_t(m_image2D->size(0));
    m_data.size[1] = uint32_t(m_image2D->size(1));
  } else if (m_type == 2 && m_image3D) {
    image = m_image3D.get();
    m_data.size[0] = uint32_t(m_image3D->size(0));
    m_data.size[1] = uint32_t(m_image3D->size(1));
    m_data.size[2] = uint32_t(m_image3D->size(2));
  }

  if (image) {
    const size_t count = image->totalSize();
    m_data.texels.resize(count * 4);
    for (size_t i = 0; i < count; ++i) {
      const float4 c = arrayElementAsFloat4(*image, i);
      std::memcpy(&m_data.texels[i * 4], &c, sizeof(c));
    }
  } else if (m_type >= 0 && m_type < 3) {
    reportMessage(ANARI_SEVERITY_WARNING, "missing required parameter 'image' on image sampler");
  }

  deviceState()->markSceneChanged();
}

bool Sampler::isValid() const
{
  // Only the parameters decide: materials look at their samplers while they
  // are committed, before the image has been converted in finalize().
  if (m_type < 0)
    return false;
  const helium::Array *image = m_type == 0 ? static_cast<const helium::Array *>(m_image1D.get())
      : (m_type == 1 ? static_cast<const helium::Array *>(m_image2D.get())
                     : static_cast<const helium::Array *>(m_image3D.get()));
  return m_type == 3 || (image && image->totalSize() > 0);
}

int Sampler::inAttribute() const
{
  return m_data.in_attribute;
}

void Sampler::exportData(bridge::SamplerData &out) const
{
  out = m_data;
}

// Material ///////////////////////////////////////////////////////////////////

int SamplerRegistry::add(const Sampler *sampler)
{
  auto it = indices.find(sampler);
  if (it != indices.end())
    return it->second;
  const int index = int(samplers->size());
  samplers->emplace_back();
  sampler->exportData(samplers->back());
  indices[sampler] = index;
  return index;
}

Material::Material(PhotonGlobalState *s, std::string_view subtype) : Object(ANARI_MATERIAL, s)
{
  m_physicallyBased = subtype == "physicallyBased";
  if (subtype != "matte" && subtype != "physicallyBased") {
    reportMessage(ANARI_SEVERITY_WARNING,
        "photon does not support material subtype '%s', using a default matte material",
        std::string(subtype).c_str());
  }
  // Defaults of a material that is never committed.
  m_color.value = m_physicallyBased ? float4(1.f, 1.f, 1.f, 1.f) : float4(0.8f, 0.8f, 0.8f, 1.f);
  m_opacity.value = float4(1.f);
  m_metallic.value = float4(m_physicallyBased ? 1.f : 0.f);
  m_roughness.value = float4(1.f);
  m_emissive.value = float4(0.f, 0.f, 0.f, 1.f);
  m_transmission.value = float4(0.f);
}

MaterialParameter Material::readParameter(const char *name, const float4 &defaultValue)
{
  MaterialParameter p;
  p.value = defaultValue;

  float f = 0.f;
  float3 v3(0.f);
  float4 v4(0.f);
  if (getParam(name, ANARI_FLOAT32, &f))
    p.value = float4(f, f, f, 1.f);
  else if (getParam(name, ANARI_FLOAT32_VEC3, &v3))
    p.value = float4(v3, 1.f);
  else if (getParam(name, ANARI_FLOAT32_VEC4, &v4))
    p.value = v4;

  const std::string attribute = getParamString(name, "");
  if (!attribute.empty()) {
    p.attribute = attributeSlot(attribute);
    if (p.attribute < 0) {
      reportMessage(ANARI_SEVERITY_WARNING,
          "unknown attribute '%s' for material parameter '%s'",
          attribute.c_str(),
          name);
    }
  }

  // An invalid sampler is skipped when the material is used (see finalize()).
  p.sampler = getParamObject<Sampler>(name);
  return p;
}

void Material::commitParameters()
{
  if (m_physicallyBased) {
    m_color = readParameter("baseColor", float4(1.f, 1.f, 1.f, 1.f));
    m_metallic = readParameter("metallic", float4(1.f));
    m_roughness = readParameter("roughness", float4(1.f));
    m_emissive = readParameter("emissive", float4(0.f, 0.f, 0.f, 1.f));
    m_transmission = readParameter("transmission", float4(0.f));
    m_ior = getParam<float>("ior", 1.5f);
    m_specular = getParam<float>("specular", 0.f);
    m_clearcoat = getParam<float>("clearcoat", 0.f);
    m_clearcoatRoughness = getParam<float>("clearcoatRoughness", 0.f);
    m_thickness = getParam<float>("thickness", 0.f);
    if (hasParam("normal") || hasParam("occlusion") || hasParam("clearcoatNormal")) {
      reportMessage(ANARI_SEVERITY_WARNING,
          "photon ignores the 'normal', 'occlusion' and 'clearcoatNormal' maps of physicallyBased materials");
    }
  } else {
    m_color = readParameter("color", float4(0.8f, 0.8f, 0.8f, 1.f));
    m_metallic = MaterialParameter();
    m_metallic.value = float4(0.f);
    m_roughness = MaterialParameter();
    m_roughness.value = float4(1.f);
    m_emissive = MaterialParameter();
    m_transmission = MaterialParameter();
    m_transmission.value = float4(0.f);
    m_specular = 0.f;
    m_clearcoat = 0.f;
  }

  m_opacity = readParameter("opacity", float4(1.f));

  const std::string alphaMode = getParamString("alphaMode", "opaque");
  m_alphaMode = alphaMode == "blend" ? 0 : (alphaMode == "mask" ? 2 : 1);
  m_alphaCutoff = getParam<float>("alphaCutoff", 0.5f);
}

void Material::finalize()
{
  for (const MaterialParameter *p : {&m_color, &m_opacity, &m_metallic, &m_roughness, &m_emissive, &m_transmission}) {
    if (p->sampler && !p->sampler->isValid()) {
      reportMessage(
          ANARI_SEVERITY_WARNING, "material uses an invalid sampler, the default value of the parameter is used");
    }
  }
  deviceState()->markSceneChanged();
}

bridge::ParamSourceData Material::exportSource(const MaterialParameter &p, SamplerRegistry &registry) const
{
  bridge::ParamSourceData src;
  if (p.sampler && p.sampler->isValid()) {
    src.kind = 2;
    src.index = registry.add(p.sampler.ptr);
  } else if (p.attribute >= 0) {
    src.kind = 1;
    src.index = p.attribute;
  }
  return src;
}

bridge::MaterialData Material::exportData(SamplerRegistry &registry) const
{
  bridge::MaterialData m;
  for (int i = 0; i < 3; ++i) {
    m.base_color[i] = m_color.value[i];
    m.emission[i] = m_emissive.value[i];
  }
  m.opacity = m_opacity.value.x;
  m.metallic = m_metallic.value.x;
  m.roughness = m_roughness.value.x;
  m.transmission = m_transmission.value.x;
  m.ior = m_ior;
  m.specular = std::clamp(m_specular, 0.f, 1.f);
  m.clearcoat = m_clearcoat;
  m.clearcoat_roughness = std::max(m_clearcoatRoughness * m_clearcoatRoughness, 0.001f);

  m.base_color_src = exportSource(m_color, registry);
  m.opacity_src = exportSource(m_opacity, registry);
  m.metallic_src = exportSource(m_metallic, registry);
  m.roughness_src = exportSource(m_roughness, registry);
  m.emission_src = exportSource(m_emissive, registry);
  m.transmission_src = exportSource(m_transmission, registry);

  m.alpha_mode = m_alphaMode;
  m.alpha_cutoff = m_alphaCutoff;
  // The alpha of the color scales the opacity.
  if (m.base_color_src.kind != 0)
    m.base_color_alpha = m_alphaMode != 1;
  else
    m.opacity *= m_color.value.w;

  // A wall thickness of 0 is thin-walled glass without refraction.
  m.thin_walled = !(m_thickness > 0.f);
  m.emission_side = 2;
  return m;
}

uint32_t Material::usedAttributes() const
{
  uint32_t mask = 0;
  auto add = [&mask](const MaterialParameter &p) {
    int slot = -1;
    if (p.sampler && p.sampler->isValid())
      slot = p.sampler->inAttribute();
    else if (p.attribute >= 0)
      slot = p.attribute;
    if (slot >= 0 && slot < bridge::kMeshAttributeCount)
      mask |= 1u << slot;
  };
  add(m_color);
  add(m_opacity);
  add(m_metallic);
  add(m_roughness);
  add(m_emissive);
  add(m_transmission);
  return mask;
}

// Surface ////////////////////////////////////////////////////////////////////

Surface::Surface(PhotonGlobalState *s) : Object(ANARI_SURFACE, s) {}

void Surface::commitParameters()
{
  m_geometry = getParamObject<Geometry>("geometry");
  m_material = getParamObject<Material>("material");
  m_visible = getParam<bool>("visible", true);
}

void Surface::finalize()
{
  if (!m_geometry)
    reportMessage(ANARI_SEVERITY_WARNING, "missing required parameter 'geometry' on surface");
  if (!m_material)
    reportMessage(ANARI_SEVERITY_WARNING, "missing parameter 'material' on surface, using a default material");
  deviceState()->markSceneChanged();
}

bool Surface::isValid() const
{
  return m_geometry && m_geometry->isValid() && m_visible;
}

Geometry *Surface::geometry() const
{
  return m_geometry.ptr;
}

Material *Surface::material() const
{
  return m_material.ptr;
}

// SpatialField ///////////////////////////////////////////////////////////////

SpatialField::SpatialField(PhotonGlobalState *s, std::string_view subtype)
    : Object(ANARI_SPATIAL_FIELD, s), m_data(this)
{
  m_supported = subtype == "structuredRegular";
  if (!m_supported) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "photon does not support spatial field subtype '%s', volumes using it are not rendered",
        std::string(subtype).c_str());
  }
}

void SpatialField::commitParameters()
{
  m_data = m_supported ? getParamObject<Array3D>("data") : nullptr;
  m_origin = getParam<float3>("origin", float3(0.f, 0.f, 0.f));
  m_spacing = getParam<float3>("spacing", float3(1.f, 1.f, 1.f));
  m_linear = getParamString("filter", "linear") != "nearest";
}

void SpatialField::finalize()
{
  if (m_supported && !m_data)
    reportMessage(ANARI_SEVERITY_WARNING, "missing required parameter 'data' on structuredRegular field");
  deviceState()->markSceneChanged();
}

bool SpatialField::isValid() const
{
  return m_supported && m_data && m_data->totalSize() > 0;
}

void SpatialField::exportData(bridge::VolumeData &out) const
{
  for (int i = 0; i < 3; ++i) {
    out.origin[i] = m_origin[i];
    out.spacing[i] = m_spacing[i] != 0.f ? m_spacing[i] : 1.f;
    out.dims[i] = uint32_t(m_data->size(i));
  }
  out.linear = m_linear;
  const size_t count = m_data->totalSize();
  out.values.resize(count);
  if (m_data->elementType() == ANARI_FLOAT32) {
    std::memcpy(out.values.data(), m_data->data(), count * sizeof(float));
  } else {
    for (size_t i = 0; i < count; ++i)
      out.values[i] = arrayElementAsFloat4(*m_data, i).x;
  }
}

void SpatialField::grid(std::vector<float> &values, uint3 &dims, float3 &origin, float3 &spacing) const
{
  bridge::VolumeData data;
  exportData(data);
  values = std::move(data.values);
  dims = uint3(data.dims[0], data.dims[1], data.dims[2]);
  origin = float3(data.origin[0], data.origin[1], data.origin[2]);
  spacing = float3(data.spacing[0], data.spacing[1], data.spacing[2]);
}

void SpatialField::bounds(float3 &lower, float3 &upper) const
{
  lower = m_origin;
  upper = m_origin;
  if (!m_data)
    return;
  for (int i = 0; i < 3; ++i) {
    const float extent = m_spacing[i] * float(std::max<size_t>(m_data->size(i), 2) - 1);
    lower[i] = std::min(m_origin[i], m_origin[i] + extent);
    upper[i] = std::max(m_origin[i], m_origin[i] + extent);
  }
}

// Volume /////////////////////////////////////////////////////////////////////

Volume::Volume(PhotonGlobalState *s, std::string_view subtype)
    : Object(ANARI_VOLUME, s), m_colorData(this), m_opacityData(this)
{
  m_supported = subtype == "transferFunction1D";
  if (!m_supported) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "photon does not support volume subtype '%s', the volume is not rendered",
        std::string(subtype).c_str());
  }
}

void Volume::commitParameters()
{
  m_field = getParamObject<SpatialField>("value");
  if (!m_field)
    m_field = getParamObject<SpatialField>("field");

  m_valueRange = float2(0.f, 1.f);
  double range64[2] = {0., 1.};
  if (getParam("valueRange", ANARI_FLOAT64_BOX1, range64))
    m_valueRange = float2(float(range64[0]), float(range64[1]));
  getParam("valueRange", ANARI_FLOAT32_BOX1, &m_valueRange);

  m_colorData = getParamObject<Array1D>("color");
  m_opacityData = getParamObject<Array1D>("opacity");

  m_color = float4(1.f, 1.f, 1.f, 1.f);
  float3 color3(1.f);
  if (getParam("color", ANARI_FLOAT32_VEC3, &color3))
    m_color = float4(color3, 1.f);
  getParam("color", ANARI_FLOAT32_VEC4, &m_color);
  m_opacity = getParam<float>("opacity", 1.f);

  m_unitDistance = getParam<float>("unitDistance", 1.f);
  // Scale used by older versions of the extension.
  if (hasParam("densityScale") && !hasParam("unitDistance")) {
    const float densityScale = getParam<float>("densityScale", 1.f);
    m_unitDistance = densityScale > 0.f ? 1.f / densityScale : 1.f;
  }
}

void Volume::finalize()
{
  if (m_supported && !m_field)
    reportMessage(ANARI_SEVERITY_WARNING, "missing required parameter 'value' on transferFunction1D volume");
  deviceState()->markSceneChanged();
}

bool Volume::isValid() const
{
  return m_supported && m_field && m_field->isValid();
}

const SpatialField *Volume::field() const
{
  return m_field.ptr;
}

void Volume::exportData(bridge::VolumeData &out) const
{
  m_field->exportData(out);
  out.value_range[0] = m_valueRange.x;
  out.value_range[1] = m_valueRange.y;
  out.density_scale = m_unitDistance > 0.f ? 1.f / m_unitDistance : 1.f;

  // Colors and opacities are resampled to one table whose first and last entry
  // sit on the ends of the value range. The table stores the extinction over a
  // unit distance, -ln(1 - opacity), which is not linear in the opacity: the
  // table is finer than the input arrays.
  const size_t colorCount = m_colorData ? m_colorData->size() : 0;
  const size_t opacityCount = m_opacityData ? m_opacityData->size() : 0;
  const size_t inputCount = std::max(colorCount, opacityCount);
  const size_t count = inputCount <= 1 ? 2 : std::clamp<size_t>((inputCount - 1) * 32 + 1, 2, 8192);

  auto lookup = [](const Array1D &array, size_t n, float x) {
    if (n == 1)
      return arrayElementAsFloat4(array, 0);
    const float f = x * float(n - 1);
    const size_t i = std::min(size_t(f), n - 2);
    const float t = f - float(i);
    return arrayElementAsFloat4(array, i) * (1.f - t) + arrayElementAsFloat4(array, i + 1) * t;
  };

  out.transfer.resize(count * 4);
  for (size_t i = 0; i < count; ++i) {
    const float x = float(i) / float(count - 1);
    float4 color = m_color;
    if (colorCount > 0)
      color = lookup(*m_colorData, colorCount, x);
    float opacity = m_opacity;
    if (opacityCount > 0)
      opacity = lookup(*m_opacityData, opacityCount, x).x;
    else if (colorCount > 0 && anari::componentsOf(m_colorData->elementType()) == 4)
      opacity = color.w * m_opacity;
    out.transfer[i * 4 + 0] = std::clamp(color.x, 0.f, 1.f);
    out.transfer[i * 4 + 1] = std::clamp(color.y, 0.f, 1.f);
    out.transfer[i * 4 + 2] = std::clamp(color.z, 0.f, 1.f);
    // An opacity of 1 is fully opaque after the unit distance; it is limited
    // to keep the extinction finite.
    opacity = std::clamp(opacity, 0.f, 0.9999f);
    out.transfer[i * 4 + 3] = -std::log(1.f - opacity);
  }
}

// Light //////////////////////////////////////////////////////////////////////

Light::Light(PhotonGlobalState *s, std::string_view subtype) : Object(ANARI_LIGHT, s), m_radiance(this)
{
  if (subtype == "directional")
    m_type = Type::Directional;
  else if (subtype == "point")
    m_type = Type::Point;
  else if (subtype == "spot")
    m_type = Type::Spot;
  else if (subtype == "quad")
    m_type = Type::Quad;
  else if (subtype == "hdri")
    m_type = Type::HDRI;
  else {
    reportMessage(ANARI_SEVERITY_WARNING,
        "photon does not support light subtype '%s', the light is ignored",
        std::string(subtype).c_str());
  }
}

void Light::commitParameters()
{
  m_color = getParam<float3>("color", float3(1.f, 1.f, 1.f));
  m_visible = getParam<bool>("visible", true);
  m_position = getParam<float3>("position", float3(0.f, 0.f, 0.f));
  m_radiance = nullptr;

  switch (m_type) {
  case Type::Directional: {
    m_direction = getParam<float3>("direction", float3(0.f, 0.f, -1.f));
    const float angularDiameter = std::max(getParam<float>("angularDiameter", 0.f), 0.f);
    m_radius = 0.5f * angularDiameter;
    // 'irradiance' in W/m^2, alternatively the 'radiance' of the disk.
    if (hasParam("irradiance") || !hasParam("radiance") || !(m_radius > 0.f)) {
      m_intensity = getParam<float>("irradiance", 1.f);
    } else {
      const float solidAngle = 2.f * kPi * (1.f - std::cos(m_radius));
      m_intensity = getParam<float>("radiance", 1.f) * solidAngle;
    }
    break;
  }
  case Type::Point: {
    m_radius = std::max(getParam<float>("radius", 0.f), 0.f);
    // 'intensity' in W/sr, 'power' in W, 'radiance' in W/sr/m^2.
    if (hasParam("intensity") || (!hasParam("power") && !hasParam("radiance")))
      m_intensity = getParam<float>("intensity", 1.f);
    else if (hasParam("power"))
      m_intensity = getParam<float>("power", 1.f) / (4.f * kPi);
    else
      m_intensity = getParam<float>("radiance", 1.f) * kPi * m_radius * m_radius;
    break;
  }
  case Type::Spot: {
    m_direction = getParam<float3>("direction", float3(0.f, 0.f, -1.f));
    m_radius = std::max(getParam<float>("radius", 0.f), 0.f);
    // 'openingAngle' is the full apex angle of the cone.
    m_openingAngle = std::clamp(getParam<float>("openingAngle", kPi), 0.f, 2.f * kPi);
    m_falloffAngle = std::clamp(getParam<float>("falloffAngle", 0.1f), 0.f, 0.5f * m_openingAngle);
    if (hasParam("intensity") || !hasParam("power")) {
      m_intensity = getParam<float>("intensity", 1.f);
    } else {
      const float solidAngle = 2.f * kPi * (1.f - std::cos(0.5f * m_openingAngle));
      m_intensity = getParam<float>("power", 1.f) / std::max(solidAngle, 1e-6f);
    }
    break;
  }
  case Type::Quad: {
    m_edge1 = getParam<float3>("edge1", float3(1.f, 0.f, 0.f));
    m_edge2 = getParam<float3>("edge2", float3(0.f, 1.f, 0.f));
    const std::string side = getParamString("side", "front");
    m_side = side == "back" ? 1 : (side == "both" ? 2 : 0);
    const float area = linalg::length(linalg::cross(m_edge1, m_edge2));
    // Radiance of the quad in W/sr/m^2.
    if (hasParam("radiance")) {
      m_intensity = getParam<float>("radiance", 1.f);
    } else if (hasParam("intensity") || !hasParam("power")) {
      m_intensity = area > 0.f ? getParam<float>("intensity", 1.f) / area : 0.f;
    } else {
      const float sides = m_side == 2 ? 2.f : 1.f;
      m_intensity = area > 0.f ? getParam<float>("power", 1.f) / (kPi * area * sides) : 0.f;
    }
    break;
  }
  case Type::HDRI: {
    m_direction = getParam<float3>("direction", float3(1.f, 0.f, 0.f));
    m_up = getParam<float3>("up", float3(0.f, 0.f, 1.f));
    m_scale = getParam<float>("scale", 1.f);
    m_radiance = getParamObject<Array2D>("radiance");
    const std::string layout = getParamString("layout", "equirectangular");
    if (layout != "equirectangular") {
      reportMessage(ANARI_SEVERITY_WARNING,
          "photon only supports the 'equirectangular' layout of hdri lights, not '%s'",
          layout.c_str());
    }
    break;
  }
  default:
    break;
  }

  m_intensity = std::isfinite(m_intensity) ? std::max(m_intensity, 0.f) : 0.f;
}

void Light::finalize()
{
  if (m_type == Type::HDRI && !m_radiance)
    reportMessage(ANARI_SEVERITY_WARNING, "missing required parameter 'radiance' on hdri light");
  deviceState()->markSceneChanged();
}

bool Light::isValid() const
{
  if (m_type == Type::Unknown)
    return false;
  if (m_type == Type::HDRI)
    return m_radiance && m_radiance->totalSize() > 0;
  return true;
}

Light::Type Light::lightType() const
{
  return m_type;
}

bool Light::visible() const
{
  return m_visible;
}

bridge::LightData Light::exportData(const mat4 &xfm) const
{
  bridge::LightData l;
  l.type = m_type == Type::Directional ? 1 : (m_type == Type::Spot ? 2 : 0);
  const float3 position = transformPoint(xfm, m_position);
  const float3 direction = safeNormalize(transformVector(xfm, m_direction), float3(0.f, 0.f, -1.f));
  for (int i = 0; i < 3; ++i) {
    l.position[i] = position[i];
    l.direction[i] = direction[i];
    l.color[i] = m_color[i];
  }
  l.intensity = m_intensity;
  l.radius = m_radius;
  l.spot_angle = 0.5f * m_openingAngle;
  l.spot_falloff = m_falloffAngle;
  return l;
}

void Light::exportQuad(const mat4 &xfm, float3 &corner, float3 &edge1, float3 &edge2, float3 &radiance, int &side) const
{
  corner = transformPoint(xfm, m_position);
  edge1 = transformVector(xfm, m_edge1);
  edge2 = transformVector(xfm, m_edge2);
  radiance = m_color * m_intensity;
  side = m_side;
}

bool Light::exportEnvironment(const mat4 &xfm, bridge::EnvironmentData &out) const
{
  if (!m_radiance)
    return false;
  const size_t width = m_radiance->size(0);
  const size_t height = m_radiance->size(1);
  if (width == 0 || height == 0)
    return false;

  out.present = true;
  out.visible = m_visible;
  out.width = uint32_t(width);
  out.height = uint32_t(height);
  out.pixels.resize(width * height * 3);
  for (size_t i = 0; i < width * height; ++i) {
    const float4 c = arrayElementAsFloat4(*m_radiance, i);
    out.pixels[i * 3 + 0] = c.x * m_color.x * m_scale;
    out.pixels[i * 3 + 1] = c.y * m_color.y * m_scale;
    out.pixels[i * 3 + 2] = c.z * m_color.z * m_scale;
  }

  // The image center maps to 'direction', the first row to -'up', and the
  // image runs from left to right when seen from the inside.
  const float3 direction = safeNormalize(transformVector(xfm, m_direction), float3(1.f, 0.f, 0.f));
  float3 up = safeNormalize(transformVector(xfm, m_up), float3(0.f, 0.f, 1.f));
  if (linalg::length(linalg::cross(up, direction)) < 1e-6f)
    up = std::fabs(direction.z) < 0.9f ? float3(0.f, 0.f, 1.f) : float3(0.f, 1.f, 0.f);
  const float3 down = -up;
  const float3 forward = -linalg::normalize(linalg::cross(down, direction));
  const float3 right = linalg::normalize(linalg::cross(forward, down));
  const float3 rows[3] = {-right, down, -forward};
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      out.world_to_map[r * 3 + c] = rows[r][c];
  return true;
}

// Group //////////////////////////////////////////////////////////////////////

Group::Group(PhotonGlobalState *s) : Object(ANARI_GROUP, s), m_surfaces(this), m_volumes(this), m_lights(this) {}

void Group::commitParameters()
{
  m_surfaces = getParamObject<ObjectArray>("surface");
  m_volumes = getParamObject<ObjectArray>("volume");
  m_lights = getParamObject<ObjectArray>("light");
}

void Group::finalize()
{
  deviceState()->markSceneChanged();
}

const ObjectArray *Group::surfaces() const
{
  return m_surfaces.get();
}

const ObjectArray *Group::volumes() const
{
  return m_volumes.get();
}

const ObjectArray *Group::lights() const
{
  return m_lights.get();
}

// Instance ///////////////////////////////////////////////////////////////////

Instance::Instance(PhotonGlobalState *s, std::string_view subtype) : Object(ANARI_INSTANCE, s)
{
  m_transform = linalg::identity;
  if (subtype != "transform") {
    reportMessage(ANARI_SEVERITY_WARNING,
        "photon does not support instance subtype '%s', using 'transform'",
        std::string(subtype).c_str());
  }
}

void Instance::commitParameters()
{
  const mat4 identity = linalg::identity;
  m_transform = getParam<mat4>("transform", identity);
  m_group = getParamObject<Group>("group");

  static const char *names[bridge::kMeshAttributeCount] = {
      "attribute0", "attribute1", "attribute2", "attribute3", "color"};
  for (int slot = 0; slot < bridge::kMeshAttributeCount; ++slot) {
    float f = 0.f;
    float3 v3(0.f);
    float4 v4(0.f, 0.f, 0.f, 1.f);
    m_hasAttribute[slot] = true;
    if (getParam(names[slot], ANARI_FLOAT32_VEC4, &v4))
      m_attributes[slot] = v4;
    else if (getParam(names[slot], ANARI_FLOAT32_VEC3, &v3))
      m_attributes[slot] = float4(v3, 1.f);
    else if (getParam(names[slot], ANARI_FLOAT32, &f))
      m_attributes[slot] = float4(f, 0.f, 0.f, 1.f);
    else
      m_hasAttribute[slot] = false;
  }
}

void Instance::finalize()
{
  if (!m_group)
    reportMessage(ANARI_SEVERITY_WARNING, "missing required parameter 'group' on instance");
  deviceState()->markSceneChanged();
}

bool Instance::isValid() const
{
  return m_group;
}

const Group *Instance::group() const
{
  return m_group.ptr;
}

const mat4 &Instance::transform() const
{
  return m_transform;
}

bool Instance::attribute(int slot, float4 &value) const
{
  if (slot < 0 || slot >= bridge::kMeshAttributeCount || !m_hasAttribute[slot])
    return false;
  value = m_attributes[slot];
  return true;
}

} // namespace photon::anari_device

PHOTON_ANARI_TYPEFOR_DEFINITION(photon::anari_device::Camera *);
PHOTON_ANARI_TYPEFOR_DEFINITION(photon::anari_device::Renderer *);
PHOTON_ANARI_TYPEFOR_DEFINITION(photon::anari_device::Sampler *);
PHOTON_ANARI_TYPEFOR_DEFINITION(photon::anari_device::Material *);
PHOTON_ANARI_TYPEFOR_DEFINITION(photon::anari_device::Geometry *);
PHOTON_ANARI_TYPEFOR_DEFINITION(photon::anari_device::Surface *);
PHOTON_ANARI_TYPEFOR_DEFINITION(photon::anari_device::SpatialField *);
PHOTON_ANARI_TYPEFOR_DEFINITION(photon::anari_device::Volume *);
PHOTON_ANARI_TYPEFOR_DEFINITION(photon::anari_device::Light *);
PHOTON_ANARI_TYPEFOR_DEFINITION(photon::anari_device::Group *);
PHOTON_ANARI_TYPEFOR_DEFINITION(photon::anari_device::Instance *);
PHOTON_ANARI_TYPEFOR_DEFINITION(photon::anari_device::World *);
