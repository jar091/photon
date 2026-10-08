#pragma once

// Interface between the ANARI device (compiled as plain C++, uses helium) and
// the Kokkos path tracer (compiled with the Kokkos device compiler, e.g. nvcc).
// It only uses standard C++ types so neither side sees the headers of the other.

#include <cstdint>
#include <string>
#include <vector>

namespace photon::bridge {

// Vertex attribute slots, they match photon::pt::ATTRIBUTE_*.
constexpr int kAttribute0 = 0;
constexpr int kAttributeColor = 4;
constexpr int kMeshAttributeCount = 5;
constexpr int kAttributeWorldPosition = 5;
constexpr int kAttributeWorldNormal = 6;

struct ParamSourceData
{
  int kind{0};  // 0 = constant, 1 = vertex attribute, 2 = sampler
  int index{0}; // attribute slot or index into SceneData::samplers
};

struct MaterialData
{
  float base_color[3]{0.8f, 0.8f, 0.8f};
  float opacity{1.f};
  float metallic{0.f};
  float roughness{1.f}; // perceptual roughness, the GGX alpha is its square
  float ior{1.5f};
  float transmission{0.f};
  float specular{0.f}; // weight of the dielectric specular lobe, 0..1
  float emission[3]{0.f, 0.f, 0.f};
  float clearcoat{0.f};
  float clearcoat_roughness{0.03f};
  float sheen{0.f};

  ParamSourceData base_color_src;
  ParamSourceData opacity_src;
  ParamSourceData metallic_src;
  ParamSourceData roughness_src;
  ParamSourceData emission_src;
  ParamSourceData transmission_src;

  int alpha_mode{1}; // 0 = blend, 1 = opaque, 2 = mask
  float alpha_cutoff{0.5f};
  bool base_color_alpha{false}; // alpha of the base color source scales the opacity
  bool thin_walled{false};      // transmission passes straight through (no refraction)
  bool emitter_only{false};     // light source geometry, paths end on it
  bool hidden_to_camera{false};
  int emission_side{2};         // 0 = front, 1 = back, 2 = both
};

struct SamplerData
{
  int type{1}; // 0 = image1D, 1 = image2D, 2 = image3D, 3 = transform
  int in_attribute{0};
  bool linear{true};
  int wrap[3]{0, 0, 0}; // 0 = clampToEdge, 1 = repeat, 2 = mirrorRepeat
  uint32_t size[3]{1, 1, 1};
  // Row-major 4x4 matrices.
  float in_transform[16]{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  float in_offset[4]{0, 0, 0, 0};
  float out_transform[16]{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  float out_offset[4]{0, 0, 0, 0};
  std::vector<float> texels; // RGBA, size[0] * size[1] * size[2] texels
};

struct LightData
{
  int type{0}; // 0 = point, 1 = directional, 2 = spot
  float position[3]{0.f, 0.f, 0.f};
  float direction[3]{0.f, 0.f, -1.f};
  float color[3]{1.f, 1.f, 1.f};
  // Point/spot: radiant intensity (W/sr). Directional: irradiance (W/m^2).
  float intensity{1.f};
  // Point/spot: radius of the sphere. Directional: angular radius (radians).
  float radius{0.f};
  float spot_angle{3.14159265f}; // half opening angle (radians)
  float spot_falloff{0.f};       // width of the smooth rim (radians)
};

struct EnvironmentData
{
  bool present{false};
  bool visible{true};
  uint32_t width{0};
  uint32_t height{0};
  std::vector<float> pixels; // RGB radiance, row 0 is the bottom (-up) row
  // Rotation from world space to the space of the map: +Y is the top row, the
  // image center is +X (row-major 3x3).
  float world_to_map[9]{1, 0, 0, 0, 1, 0, 0, 0, 1};
};

struct VolumeData
{
  float world_to_local[12]{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}; // row-major 3x4
  float origin[3]{0.f, 0.f, 0.f};
  float spacing[3]{1.f, 1.f, 1.f};
  uint32_t dims[3]{1, 1, 1};
  std::vector<float> values;    // dims[0] * dims[1] * dims[2], x runs fastest
  std::vector<float> transfer;  // RGBA table spanning value_range, A is the opacity
  float value_range[2]{0.f, 1.f};
  float density_scale{1.f}; // extinction = opacity * density_scale
  bool linear{true};
};

struct SceneData
{
  // Indexed triangle mesh in world space.
  std::vector<float> positions;                    // xyz per vertex
  std::vector<float> normals;                      // xyz per vertex, or empty
  std::vector<float> attributes[kMeshAttributeCount]; // RGBA per vertex, or empty
  std::vector<uint32_t> indices;                   // three per triangle
  std::vector<uint32_t> material_ids;              // one per triangle

  std::vector<MaterialData> materials;
  std::vector<SamplerData> samplers;
  std::vector<LightData> lights;
  EnvironmentData environment;
  std::vector<VolumeData> volumes;
};

struct CameraData
{
  float position[3]{0.f, 0.f, 0.f};
  float direction[3]{0.f, 0.f, -1.f};
  float up[3]{0.f, 1.f, 0.f};
  bool orthographic{false};
  float fovy{1.0471976f}; // radians
  float height{1.f};      // orthographic image plane height
  float aspect{1.f};
  float aperture_radius{0.f};
  float focus_distance{1.f};
  float image_region[4]{0.f, 0.f, 1.f, 1.f};
};

struct RenderSettings
{
  uint32_t width{0};
  uint32_t height{0};
  uint32_t pixel_samples{1};
  uint32_t max_depth{6}; // scattering events lit by next event estimation
  float background[4]{0.f, 0.f, 0.f, 1.f};
  // Optional RGBA background image, rows start at the bottom.
  const float *background_image{nullptr};
  uint32_t background_width{0};
  uint32_t background_height{0};
  float ambient_color[3]{1.f, 1.f, 1.f};
  float ambient_radiance{0.f};
  bool adaptive_sampling{false};
  float throughput_clamp{0.f}; // <= 0: off
};

// The path tracer of one ANARI device: scene, acceleration structure and the
// accumulated frame. All Kokkos work runs on one internal thread, the methods
// block until it is done and can be called from any thread (one at a time).
class Renderer
{
 public:
  Renderer();
  ~Renderer();
  Renderer(const Renderer &) = delete;
  Renderer &operator=(const Renderer &) = delete;

  // Starts Kokkos (once per process) and creates the ray tracing back-end.
  // Returns false and a message when that is not possible.
  bool initialize(std::string &error);

  // Name of the ray tracing back-end ("optix", "embree", "kokkos").
  std::string backend_name() const;
  // Name of the Kokkos execution space ("Cuda", "Threads", "Serial", ...).
  std::string execution_space() const;

  bool set_scene(const SceneData &scene, std::string &error);

  // Renders `pixel_samples` more samples per pixel. `reset` drops the samples
  // accumulated so far.
  bool render(const CameraData &camera, const RenderSettings &settings, bool reset, std::string &error);

  uint32_t accumulated_samples() const;

  // Results of the accumulated frame, rows start at the bottom.
  bool read_color(float *rgba, std::string &error);  // linear radiance and alpha
  bool read_depth(float *depth, std::string &error); // distance along the ray, infinity for no hit
  bool read_normal(float *xyz, std::string &error);
  bool read_albedo(float *rgb, std::string &error);

  // Releases the path tracers of all renderers and finalizes Kokkos. Called by
  // the exit handler of the process, renderers cannot be used afterwards.
  static void shutdown();

 private:
  struct Impl;
  Impl *m_impl{nullptr};
  uint32_t m_samples{0};
};

} // namespace photon::bridge
