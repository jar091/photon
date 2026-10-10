#include "photon/anari/RenderBridge.h"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <thread>

#include "photon/pt/backend/ray_backend.h"
#include "photon/pt/camera.h"
#include "photon/pt/environment_map.h"
#include "photon/pt/pathtracer.h"
#include "photon/pt/scene.h"

#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <corecrt_startup.h>
#endif

namespace photon::bridge {

namespace {

using namespace photon::pt;

// Keeps this library loaded until the process ends: Kokkos can be initialized
// only once per process and its threads and device state do not survive
// unloading the library.
void pin_library()
{
#ifdef _WIN32
  HMODULE module = nullptr;
  GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_PIN | GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
      reinterpret_cast<LPCWSTR>(&pin_library), &module);
#endif
}

// State of Kokkos in this process (it can be initialized only once).
std::atomic<bool> g_kokkos_initialized{false};
std::atomic<bool> g_kokkos_finalized{false};
// True once the exit handler has made it safe to run the static destructors of
// Kokkos (see Renderer::shutdown()).
std::atomic<bool> g_exit_safe{false};

// The single thread that runs all Kokkos code of the process. Kokkos is
// initialized on it on first use and stays initialized while devices are
// created and destroyed; the exit handler of the process cleans up
// (Renderer::shutdown()). The thread is detached and the object is never
// destroyed, so nothing has to be joined when the process exits.
class KokkosThread
{
 public:
  static KokkosThread &instance()
  {
    static KokkosThread *thread = new KokkosThread();
    return *thread;
  }

  // Runs `task` on the Kokkos thread and waits for it. Returns false and the
  // message when the task threw an exception or cannot run any more because
  // the process is shutting down.
  bool run(const std::function<void()> &task, std::string &error)
  {
    if (!thread_alive() || g_kokkos_finalized) {
      error = "the process is shutting down";
      return false;
    }
    std::lock_guard<std::timed_mutex> caller_lock(m_caller_mutex);
    std::unique_lock<std::mutex> lock(m_mutex);
    m_task = &task;
    m_done = false;
    m_error.clear();
    m_failed = false;
    m_wake.notify_all();
    while (!m_done) {
      m_finished.wait_for(lock, std::chrono::milliseconds(250));
      // Objects can be released while the process exits, when the Kokkos
      // thread is gone already: nothing will ever finish the task then.
      if (!m_done && !thread_alive()) {
        m_task = nullptr;
        m_done = true;
        error = "the process is shutting down";
        return false;
      }
    }
    m_task = nullptr;
    if (m_failed)
      error = m_error;
    return !m_failed;
  }

  // Starts Kokkos. Must be called from a task.
  void initialize_kokkos()
  {
    if (m_kokkos_ready)
      return;

#if defined(KOKKOS_ENABLE_CUDA)
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0)
      throw std::runtime_error("no CUDA device found");
#endif

    if (!Kokkos::is_initialized()) {
      Kokkos::InitializationSettings settings;
      settings.set_disable_warnings(true);
      int threads = int(std::thread::hardware_concurrency());
      if (const char *env = std::getenv("PHOTON_NUM_THREADS")) {
        const int requested = std::atoi(env);
        if (requested > 0)
          threads = requested;
      }
      if (threads > 0)
        settings.set_num_threads(threads);
      Kokkos::initialize(settings);
    }
    m_kokkos_ready = true;
    g_kokkos_initialized = true;

    // Kokkos has to be prepared for the end of the process: its static
    // destructors wait for worker threads and release device memory, which
    // fails once the other threads and the CUDA runtime are gone.
#ifdef _WIN32
    // Process wide exit handler of the C runtime: it runs before the other
    // threads are terminated (handlers registered with atexit() in a DLL run
    // after that).
    _crt_atexit(&KokkosThread::at_process_exit);
#else
    std::atexit(&KokkosThread::at_process_exit);
#endif
  }

  // Like run(), but stops waiting after `timeout`. The task may still be
  // running then (or start later), so it must not refer to the stack of the
  // caller; no further tasks are accepted in that case.
  bool run_with_timeout(std::function<void()> task, std::chrono::milliseconds timeout)
  {
    if (!thread_alive() || g_kokkos_finalized)
      return false;
    if (!m_caller_mutex.try_lock_for(timeout))
      return false;
    std::unique_lock<std::mutex> lock(m_mutex);
    m_owned_task = std::move(task);
    m_task = &m_owned_task;
    m_done = false;
    m_error.clear();
    m_failed = false;
    m_wake.notify_all();
    if (!m_finished.wait_for(lock, timeout, [this] { return m_done; }))
      return false; // m_caller_mutex stays locked
    m_task = nullptr;
    const bool ok = !m_failed;
    lock.unlock();
    m_caller_mutex.unlock();
    return ok;
  }

  // Keeps further tasks from running. Returns false when a task is still
  // running after a few seconds.
  bool lock_for_shutdown()
  {
    return m_caller_mutex.try_lock_for(std::chrono::seconds(5));
  }

  // Finalizes Kokkos, after all Kokkos objects have been released. The Threads
  // back-end has to be finalized by a task (on the thread that initialized it).
  void finalize_kokkos()
  {
    m_parking_allowed = false;
    if (Kokkos::is_initialized() && !Kokkos::is_finalized())
      Kokkos::finalize();
    m_kokkos_ready = false;
    g_kokkos_finalized = true;
    g_exit_safe = true;
  }

 private:
  KokkosThread()
  {
    pin_library();
    std::thread thread([this] { loop(); });
#ifdef _WIN32
    DuplicateHandle(GetCurrentProcess(), static_cast<HANDLE>(thread.native_handle()), GetCurrentProcess(),
        &m_thread_handle, SYNCHRONIZE, FALSE, 0);
    if (HMODULE ntdll = GetModuleHandleW(L"ntdll.dll"))
      m_shutdown_in_progress = reinterpret_cast<ShutdownQuery>(GetProcAddress(ntdll, "RtlDllShutdownInProgress"));
#endif
    thread.detach();
  }

  // False once the process is exiting or the Kokkos thread has been terminated
  // (Windows ends all other threads before the libraries are unloaded).
  bool thread_alive() const
  {
#ifdef _WIN32
    if (m_shutdown_in_progress && m_shutdown_in_progress())
      return false;
    if (m_thread_handle && WaitForSingleObject(m_thread_handle, 0) != WAIT_TIMEOUT)
      return false;
#endif
    return true;
  }

  static void at_process_exit()
  {
    Renderer::shutdown();
  }

  bool task_pending()
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_task != nullptr && !m_done;
  }

  // Blocks until a task arrives. The worker threads of the Kokkos Threads
  // back-end spin while they are idle, so they are parked inside a parallel
  // region that sleeps until there is work.
  void wait_for_task()
  {
#if defined(KOKKOS_ENABLE_THREADS)
    if (m_kokkos_ready && m_parking_allowed && !task_pending()) {
      const int concurrency = Kokkos::Threads().concurrency();
      if (concurrency > 1) {
        KokkosThread *self = this;
        Kokkos::parallel_for(
            "photon_park", Kokkos::RangePolicy<Kokkos::Threads>(0, concurrency), [self](const int) {
              std::unique_lock<std::mutex> lock(self->m_mutex);
              self->m_wake.wait(lock, [self] { return self->m_task != nullptr && !self->m_done; });
            });
        return;
      }
    }
#endif
    std::unique_lock<std::mutex> lock(m_mutex);
    m_wake.wait(lock, [this] { return m_task != nullptr && !m_done; });
  }

  void loop()
  {
    for (;;) {
      wait_for_task();

      const std::function<void()> *task = nullptr;
      {
        std::lock_guard<std::mutex> lock(m_mutex);
        task = m_task;
      }

      bool failed = false;
      std::string message;
      try {
        (*task)();
      } catch (const std::exception &e) {
        failed = true;
        message = e.what();
      } catch (...) {
        failed = true;
        message = "unknown exception";
      }

      {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_failed = failed;
        m_error = message;
        m_done = true;
      }
      m_finished.notify_all();
    }
  }

  std::timed_mutex m_caller_mutex;
  std::mutex m_mutex;
  std::condition_variable m_wake;
  std::condition_variable m_finished;
  const std::function<void()> *m_task{nullptr};
  std::function<void()> m_owned_task;
  bool m_done{true};
  bool m_failed{false};
  std::string m_error;
  bool m_kokkos_ready{false};
  std::atomic<bool> m_parking_allowed{true};
#ifdef _WIN32
  using ShutdownQuery = BOOLEAN(NTAPI *)();
  HANDLE m_thread_handle{nullptr};
  ShutdownQuery m_shutdown_in_progress{nullptr};
#endif
};

Vec3 vec3_of(const float *v)
{
  return {v[0], v[1], v[2]};
}

ParamSource convert_source(const ParamSourceData &src)
{
  ParamSource out;
  out.kind = src.kind == 1 ? ParamSourceKind::Attribute
                           : (src.kind == 2 ? ParamSourceKind::Sampler : ParamSourceKind::Constant);
  out.index = src.index;
  return out;
}

Material convert_material(const MaterialData &in)
{
  Material m;
  m.base_color = vec3_of(in.base_color);
  m.metallic = in.metallic;
  // The BSDF uses the GGX alpha directly.
  m.roughness = in.roughness * in.roughness;
  m.ior = in.ior;
  m.transmission = in.transmission;
  // Disney convention: 0.5 is the full dielectric reflectance.
  m.specular = 0.5f * in.specular;
  m.clearcoat = in.clearcoat;
  m.clearcoat_roughness = in.clearcoat_roughness;
  m.sheen = in.sheen;
  m.emission = vec3_of(in.emission);
  m.emission_strength = 1.f;
  m.alpha = in.opacity;

  m.base_color_src = convert_source(in.base_color_src);
  m.alpha_src = convert_source(in.opacity_src);
  m.metallic_src = convert_source(in.metallic_src);
  m.roughness_src = convert_source(in.roughness_src);
  m.emission_src = convert_source(in.emission_src);
  m.transmission_src = convert_source(in.transmission_src);
  m.normal_src = convert_source(in.normal_src);
  m.roughness_src_squared = 1;
  m.base_color_alpha = in.base_color_alpha ? 1 : 0;

  // A sourced opacity scales the constant, which then has to be neutral.
  if (m.alpha_src.kind != ParamSourceKind::Constant)
    m.alpha = 1.f;

  m.alpha_mode = in.alpha_mode == 0 ? AlphaMode::Blend : (in.alpha_mode == 2 ? AlphaMode::Mask : AlphaMode::Opaque);
  m.alpha_cutoff = in.alpha_cutoff;

  m.thin = in.thin_walled ? 1 : 0;
  m.flags = MATERIAL_FLAG_REFRACTIVE;
  if (in.emitter_only)
    m.flags |= MATERIAL_FLAG_EMITTER_ONLY;
  if (in.hidden_to_camera)
    m.flags |= MATERIAL_FLAG_HIDDEN_TO_CAMERA;
  if (in.emission_side == 2)
    m.flags |= MATERIAL_FLAG_EMIT_BOTH_SIDES;
  else if (in.emission_side == 1)
    m.flags |= MATERIAL_FLAG_EMIT_BACK_ONLY;
  return m;
}

template <typename ViewType, typename Fill>
void upload(ViewType &view, const Fill &fill)
{
  auto host = Kokkos::create_mirror_view(view);
  fill(host);
  Kokkos::deep_copy(view, host);
}

} // namespace

namespace detail {

// Adds a rendered frame to the accumulated one. A free function with external
// linkage: CUDA does not allow device lambdas inside the private Renderer::Impl.
void accumulate_frame(const Kokkos::View<Vec3 **, Kokkos::LayoutRight> &color,
    const Kokkos::View<f32 **, Kokkos::LayoutRight> &alpha,
    const Kokkos::View<f32 **, Kokkos::LayoutRight> &depth,
    const Kokkos::View<Vec3 **, Kokkos::LayoutRight> &normal,
    const Kokkos::View<Vec3 **, Kokkos::LayoutRight> &albedo,
    const RenderResult &result,
    f32 weight,
    bool first,
    u32 w,
    u32 h)
{
  Kokkos::parallel_for(
      "frame_accumulate", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {int(h), int(w)}),
      KOKKOS_LAMBDA(const int y, const int x) {
        if (first) {
          color(y, x) = result.color(y, x) * weight;
          alpha(y, x) = result.alpha(y, x) * weight;
          depth(y, x) = result.depth(y, x);
          normal(y, x) = result.normal(y, x);
          albedo(y, x) = result.albedo(y, x);
        } else {
          color(y, x) = color(y, x) + result.color(y, x) * weight;
          alpha(y, x) = alpha(y, x) + result.alpha(y, x) * weight;
        }
      });
  Kokkos::fence();
}

} // namespace detail

struct Renderer::Impl
{
  std::unique_ptr<RayBackend> backend;
  std::optional<Scene> scene;
  PathTracer pathtracer;
  std::string backend_name;
  std::string execution_space;

  // Accumulated frame.
  Kokkos::View<Vec3 **, Kokkos::LayoutRight> accum_color;
  Kokkos::View<f32 **, Kokkos::LayoutRight> accum_alpha;
  Kokkos::View<f32 **, Kokkos::LayoutRight> first_depth;
  Kokkos::View<Vec3 **, Kokkos::LayoutRight> first_normal;
  Kokkos::View<Vec3 **, Kokkos::LayoutRight> first_albedo;
  Kokkos::View<Vec4 **, Kokkos::LayoutRight> background_image;
  std::vector<float> background_copy;
  u32 width{0};
  u32 height{0};
  u32 samples{0};
  u32 sample_counter{0}; // samples rendered so far, decorrelates the frames

  void build_scene(const SceneData &in);
  void render(const CameraData &camera, const RenderSettings &settings, bool reset);
};

void Renderer::Impl::build_scene(const SceneData &in)
{
  Scene s;

  size_t vertex_count = in.positions.size() / 3;
  size_t triangle_count = in.indices.size() / 3;
  // The ray tracing back-ends need at least one triangle: a degenerate one is
  // never hit.
  const bool empty = triangle_count == 0 || vertex_count == 0;
  if (empty) {
    vertex_count = 3;
    triangle_count = 1;
  }

  TriangleMesh mesh;
  mesh.positions = Kokkos::View<Vec3 *>("scene_positions", vertex_count);
  mesh.indices = Kokkos::View<u32 *>("scene_indices", triangle_count * 3);
  mesh.material_ids = Kokkos::View<u32 *>("scene_material_ids", triangle_count);

  Aabb bounds;
  upload(mesh.positions, [&](auto &h) {
    for (size_t i = 0; i < vertex_count; ++i) {
      h(i) = empty ? Vec3{0.f, 0.f, 0.f} : vec3_of(&in.positions[3 * i]);
      bounds.extend(h(i));
    }
  });
  upload(mesh.indices, [&](auto &h) {
    for (size_t i = 0; i < triangle_count * 3; ++i)
      h(i) = empty ? u32(i) : in.indices[i];
  });
  const u32 material_count = u32(std::max<size_t>(in.materials.size(), 1));
  upload(mesh.material_ids, [&](auto &h) {
    for (size_t i = 0; i < triangle_count; ++i) {
      const u32 id = (!empty && i < in.material_ids.size()) ? in.material_ids[i] : 0u;
      h(i) = id < material_count ? id : 0u;
    }
  });

  if (!empty && in.normals.size() >= vertex_count * 3) {
    mesh.normals = Kokkos::View<Vec3 *>("scene_normals", vertex_count);
    upload(mesh.normals, [&](auto &h) {
      for (size_t i = 0; i < vertex_count; ++i)
        h(i) = vec3_of(&in.normals[3 * i]);
    });
  }

  s.mesh = mesh;

  // Vertex attributes.
  {
    size_t total = 0;
    for (int slot = 0; slot < kMeshAttributeCount; ++slot) {
      if (!empty && in.attributes[slot].size() >= vertex_count * 4) {
        s.attributes.offset[slot] = int(total);
        total += vertex_count;
      }
    }
    if (total > 0) {
      s.attributes.values = Kokkos::View<Vec4 *>("scene_attributes", total);
      upload(s.attributes.values, [&](auto &h) {
        for (int slot = 0; slot < kMeshAttributeCount; ++slot) {
          if (s.attributes.offset[slot] < 0)
            continue;
          const size_t base = size_t(s.attributes.offset[slot]);
          const float *src = in.attributes[slot].data();
          for (size_t i = 0; i < vertex_count; ++i)
            h(base + i) = Vec4{src[4 * i], src[4 * i + 1], src[4 * i + 2], src[4 * i + 3]};
        }
      });
    }
  }

  // Materials.
  std::vector<Material> materials;
  materials.reserve(material_count);
  for (const auto &m : in.materials)
    materials.push_back(convert_material(m));
  if (materials.empty())
    materials.push_back(Material{});
  s.material_count = material_count;
  s.materials = Kokkos::View<Material *>("scene_materials", material_count);
  upload(s.materials, [&](auto &h) {
    for (u32 i = 0; i < material_count; ++i)
      h(i) = materials[i];
  });

  // Samplers.
  if (!in.samplers.empty()) {
    size_t texel_count = 0;
    for (const auto &sd : in.samplers)
      texel_count += sd.texels.size() / 4;
    s.samplers.count = u32(in.samplers.size());
    s.samplers.infos = Kokkos::View<SamplerInfo *>("scene_sampler_infos", in.samplers.size());
    s.samplers.texels = Kokkos::View<Vec4 *>("scene_sampler_texels", std::max<size_t>(texel_count, 1));
    auto infos_h = Kokkos::create_mirror_view(s.samplers.infos);
    auto texels_h = Kokkos::create_mirror_view(s.samplers.texels);
    size_t offset = 0;
    for (size_t i = 0; i < in.samplers.size(); ++i) {
      const SamplerData &sd = in.samplers[i];
      SamplerInfo si;
      si.type = sd.type == 0 ? SamplerType::Image1D
                             : (sd.type == 2 ? SamplerType::Image3D
                                             : (sd.type == 3 ? SamplerType::Transform : SamplerType::Image2D));
      si.in_attribute = sd.in_attribute;
      si.linear = sd.linear ? 1u : 0u;
      for (int d = 0; d < 3; ++d) {
        si.wrap[d] = sd.wrap[d] == 1 ? WrapMode::Repeat
                                     : (sd.wrap[d] == 2 ? WrapMode::MirrorRepeat : WrapMode::ClampToEdge);
        si.size[d] = std::max<uint32_t>(sd.size[d], 1);
      }
      si.offset = u32(offset);
      std::memcpy(si.in_transform, sd.in_transform, sizeof(si.in_transform));
      std::memcpy(si.in_offset, sd.in_offset, sizeof(si.in_offset));
      std::memcpy(si.out_transform, sd.out_transform, sizeof(si.out_transform));
      std::memcpy(si.out_offset, sd.out_offset, sizeof(si.out_offset));

      const size_t n = sd.texels.size() / 4;
      const size_t expected = size_t(si.size[0]) * si.size[1] * si.size[2];
      if (si.type != SamplerType::Transform && n < expected) {
        // Incomplete image: a constant sampler keeps the lookups in bounds.
        si.type = SamplerType::Transform;
      }
      for (size_t t = 0; t < n; ++t)
        texels_h(offset + t) =
            Vec4{sd.texels[4 * t], sd.texels[4 * t + 1], sd.texels[4 * t + 2], sd.texels[4 * t + 3]};
      offset += n;
      infos_h(i) = si;
    }
    Kokkos::deep_copy(s.samplers.infos, infos_h);
    Kokkos::deep_copy(s.samplers.texels, texels_h);
  }

  // Lights.
  if (!in.lights.empty()) {
    s.light_count = u32(in.lights.size());
    s.lights = Kokkos::View<Light *>("scene_lights", in.lights.size());
    upload(s.lights, [&](auto &h) {
      for (size_t i = 0; i < in.lights.size(); ++i) {
        const LightData &ld = in.lights[i];
        Light l;
        l.type = ld.type == 1 ? LightType::Directional : (ld.type == 2 ? LightType::Spot : LightType::Point);
        l.position = vec3_of(ld.position);
        l.direction = vec3_of(ld.direction);
        l.color = vec3_of(ld.color);
        l.intensity = ld.intensity;
        l.radius = ld.radius;
        l.spot_angle = ld.spot_angle;
        l.spot_falloff = ld.spot_falloff;
        h(i) = l;
      }
    });
  }

  // Environment map.
  const EnvironmentData &env = in.environment;
  if (env.present && env.width > 0 && env.height > 0
      && env.pixels.size() >= size_t(env.width) * env.height * 3) {
    EnvironmentMap map;
    map.width = env.width;
    map.height = env.height;
    map.pixels = Kokkos::View<Vec3 **, Kokkos::LayoutRight>("env_pixels", env.height, env.width);
    map.marginal_cdf = Kokkos::View<f32 *>("env_marginal", env.height + 1);
    map.conditional_cdf = Kokkos::View<f32 **>("env_conditional", env.height, env.width + 1);
    upload(map.pixels, [&](auto &h) {
      for (u32 y = 0; y < env.height; ++y)
        for (u32 x = 0; x < env.width; ++x) {
          const float *p = &env.pixels[(size_t(y) * env.width + x) * 3];
          auto finite = [](float v) { return std::isfinite(v) && v > 0.f ? v : 0.f; };
          h(y, x) = Vec3{finite(p[0]), finite(p[1]), finite(p[2])};
        }
    });
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c) {
        map.world_to_tex[r * 3 + c] = env.world_to_map[r * 3 + c];
        map.tex_to_world[c * 3 + r] = env.world_to_map[r * 3 + c];
      }
    map.build_cdf();
    s.env_map = map;
    s.env_visible = env.visible;
  }

  // Volumes.
  if (!in.volumes.empty()) {
    size_t value_count = 0;
    size_t tf_count = 0;
    for (const auto &vd : in.volumes) {
      value_count += vd.values.size();
      tf_count += vd.transfer.size() / 4;
    }
    s.volumes.count = u32(in.volumes.size());
    s.volumes.volumes = Kokkos::View<TfVolume *>("scene_volumes", in.volumes.size());
    s.volumes.values = Kokkos::View<f32 *>("scene_volume_values", std::max<size_t>(value_count, 1));
    s.volumes.tf = Kokkos::View<Vec4 *>("scene_volume_tf", std::max<size_t>(tf_count, 1));
    auto volumes_h = Kokkos::create_mirror_view(s.volumes.volumes);
    auto values_h = Kokkos::create_mirror_view(s.volumes.values);
    auto tf_h = Kokkos::create_mirror_view(s.volumes.tf);
    std::vector<f32> majorants;
    size_t value_offset = 0;
    size_t tf_offset = 0;
    for (size_t i = 0; i < in.volumes.size(); ++i) {
      const VolumeData &vd = in.volumes[i];
      TfVolume v;
      std::memcpy(v.world_to_local, vd.world_to_local, sizeof(v.world_to_local));
      v.origin = vec3_of(vd.origin);
      v.spacing = vec3_of(vd.spacing);
      for (int d = 0; d < 3; ++d)
        v.dims[d] = std::max<uint32_t>(vd.dims[d], 1);
      v.data_offset = u32(value_offset);
      v.tf_offset = u32(tf_offset);
      v.tf_size = u32(vd.transfer.size() / 4);
      v.value_lo = vd.value_range[0];
      v.value_hi = vd.value_range[1];
      v.density_scale = vd.density_scale;
      v.linear = vd.linear ? 1u : 0u;

      f32 max_opacity = 0.f;
      for (size_t t = 0; t < vd.transfer.size() / 4; ++t) {
        const Vec4 c{vd.transfer[4 * t], vd.transfer[4 * t + 1], vd.transfer[4 * t + 2], vd.transfer[4 * t + 3]};
        tf_h(tf_offset + t) = c;
        max_opacity = std::max(max_opacity, c.w);
      }
      v.majorant = max_opacity * vd.density_scale;
      const bool complete = vd.values.size() >= size_t(v.dims[0]) * v.dims[1] * v.dims[2];
      if (!complete)
        v.majorant = 0.f; // incomplete data, the volume is skipped

      // Macro cells with the largest extinction the transfer function has for
      // the values inside each of them. Large grids get cells of about 4 grid
      // cells per axis; small grids are refined instead, every grid cell is
      // split into several macro cells (a coarse grid with a steep transfer
      // function is mostly empty space, too).
      v.macro_offset = u32(majorants.size());
      u32 subdivision[3] = {0, 0, 0}; // macro cells per grid cell, 0: coarse cells
      for (int d = 0; d < 3; ++d) {
        const u32 cells = v.dims[d] > 1 ? v.dims[d] - 1 : 1;
        if (cells <= 32) {
          subdivision[d] = std::min<u32>(std::max<u32>((32 + cells - 1) / cells, 1), 16);
          v.macro_dims[d] = cells * subdivision[d];
        } else {
          v.macro_dims[d] = std::min<u32>(std::max<u32>((v.dims[d] + 2) / 4, 1), 128);
        }
      }
      const size_t table_size = vd.transfer.size() / 4;
      const f32 range = vd.value_range[1] - vd.value_range[0];
      const bool refined = subdivision[0] > 0 && subdivision[1] > 0 && subdivision[2] > 0;
      if (!refined) {
        for (int d = 0; d < 3; ++d)
          if (subdivision[d] > 0)
            v.macro_dims[d] = std::min<u32>(std::max<u32>((v.dims[d] + 2) / 4, 1), 128);
      }
      // Trilinear interpolation of the grid, like the renderer does it.
      auto node = [&](u32 x, u32 y, u32 z) {
        x = std::min(x, v.dims[0] - 1);
        y = std::min(y, v.dims[1] - 1);
        z = std::min(z, v.dims[2] - 1);
        return vd.values[(size_t(z) * v.dims[1] + y) * v.dims[0] + x];
      };
      auto interpolate = [&](f32 gx, f32 gy, f32 gz) {
        const u32 x = u32(gx), y = u32(gy), z = u32(gz);
        const f32 fx = gx - f32(x), fy = gy - f32(y), fz = gz - f32(z);
        const f32 c00 = node(x, y, z) * (1.f - fx) + node(x + 1, y, z) * fx;
        const f32 c10 = node(x, y + 1, z) * (1.f - fx) + node(x + 1, y + 1, z) * fx;
        const f32 c01 = node(x, y, z + 1) * (1.f - fx) + node(x + 1, y, z + 1) * fx;
        const f32 c11 = node(x, y + 1, z + 1) * (1.f - fx) + node(x + 1, y + 1, z + 1) * fx;
        return (c00 * (1.f - fy) + c10 * fy) * (1.f - fz) + (c01 * (1.f - fy) + c11 * fy) * fz;
      };
      for (u32 mz = 0; mz < v.macro_dims[2]; ++mz) {
        for (u32 my = 0; my < v.macro_dims[1]; ++my) {
          for (u32 mx = 0; mx < v.macro_dims[0]; ++mx) {
            f32 bound = 0.f;
            if (complete && table_size > 0) {
              const u32 m[3] = {mx, my, mz};
              f32 value_min = std::numeric_limits<f32>::max();
              f32 value_max = -std::numeric_limits<f32>::max();
              if (refined) {
                // The macro cell lies inside one grid cell, where the field is
                // trilinear: its extremes are at the corners of the macro cell.
                for (int corner = 0; corner < 8; ++corner) {
                  f32 g[3];
                  for (int d = 0; d < 3; ++d)
                    g[d] = f32(m[d] + ((corner >> d) & 1)) / f32(subdivision[d]);
                  const f32 value = interpolate(g[0], g[1], g[2]);
                  value_min = std::min(value_min, value);
                  value_max = std::max(value_max, value);
                }
              } else {
                // Grid points that influence the cell (trilinear interpolation).
                u32 first[3], last[3];
                for (int d = 0; d < 3; ++d) {
                  const f32 cells = f32(v.dims[d] > 1 ? v.dims[d] - 1 : 1);
                  const f32 lo = f32(m[d]) / f32(v.macro_dims[d]) * cells;
                  const f32 hi = f32(m[d] + 1) / f32(v.macro_dims[d]) * cells;
                  first[d] = u32(std::max(std::floor(lo) - 1.f, 0.f));
                  last[d] = std::min<u32>(u32(std::ceil(hi)) + 1, v.dims[d] - 1);
                }
                for (u32 z = first[2]; z <= last[2]; ++z)
                  for (u32 y = first[1]; y <= last[1]; ++y)
                    for (u32 x = first[0]; x <= last[0]; ++x) {
                      const f32 value = vd.values[(size_t(z) * v.dims[1] + y) * v.dims[0] + x];
                      value_min = std::min(value_min, value);
                      value_max = std::max(value_max, value);
                    }
              }
              // Table entries the value interval can reach.
              auto table_position = [&](f32 value) {
                f32 x = range != 0.f ? (value - vd.value_range[0]) / range : 0.f;
                x = std::min(std::max(x, 0.f), 1.f);
                return x * f32(table_size - 1);
              };
              const size_t i0 = size_t(std::floor(table_position(value_min)));
              const size_t i1 = std::min(size_t(std::ceil(table_position(value_max))), table_size - 1);
              for (size_t t = i0; t <= i1; ++t)
                bound = std::max(bound, vd.transfer[4 * t + 3]);
              if (!(value_min <= value_max))
                bound = max_opacity; // non-finite data
            }
            // A small margin covers rounding in the interpolation.
            majorants.push_back(bound * vd.density_scale * 1.01f);
          }
        }
      }

      for (size_t k = 0; k < vd.values.size(); ++k)
        values_h(value_offset + k) = vd.values[k];

      value_offset += vd.values.size();
      tf_offset += vd.transfer.size() / 4;
      volumes_h(i) = v;
    }
    Kokkos::deep_copy(s.volumes.volumes, volumes_h);
    Kokkos::deep_copy(s.volumes.values, values_h);
    Kokkos::deep_copy(s.volumes.tf, tf_h);
    s.volumes.majorants = Kokkos::View<f32 *>("scene_volume_majorants", std::max<size_t>(majorants.size(), 1));
    upload(s.volumes.majorants, [&](auto &h) {
      for (size_t k = 0; k < majorants.size(); ++k)
        h(k) = majorants[k];
    });
  }

  // Emissive triangles for next event estimation.
  if (!empty) {
    std::vector<u32> ids;
    std::vector<f32> areas;
    f32 total_area = 0.f;
    for (size_t t = 0; t < triangle_count; ++t) {
      const u32 mat_id = t < in.material_ids.size() && in.material_ids[t] < material_count ? in.material_ids[t] : 0u;
      if (!material_may_emit(materials[mat_id]))
        continue;
      const Vec3 p0 = vec3_of(&in.positions[3 * size_t(in.indices[3 * t + 0])]);
      const Vec3 p1 = vec3_of(&in.positions[3 * size_t(in.indices[3 * t + 1])]);
      const Vec3 p2 = vec3_of(&in.positions[3 * size_t(in.indices[3 * t + 2])]);
      const f32 area = 0.5f * length(cross(p1 - p0, p2 - p0));
      if (!(area > 0.f))
        continue;
      ids.push_back(u32(t));
      areas.push_back(area);
      total_area += area;
    }
    if (!ids.empty()) {
      s.emissive_count = u32(ids.size());
      s.total_emissive_area = total_area;
      s.emissive_prim_ids = Kokkos::View<u32 *>("emissive_ids", ids.size());
      s.emissive_prim_areas = Kokkos::View<f32 *>("emissive_areas", ids.size());
      upload(s.emissive_prim_ids, [&](auto &h) {
        for (size_t i = 0; i < ids.size(); ++i)
          h(i) = ids[i];
      });
      upload(s.emissive_prim_areas, [&](auto &h) {
        for (size_t i = 0; i < areas.size(); ++i)
          h(i) = areas[i];
      });
    }
  }

  // Ray offset relative to the size of the scene.
  s.ray_epsilon = 1e-3f;
  if (!empty && bounds.hi.x >= bounds.lo.x) {
    const f32 eps = 1e-4f * length(bounds.hi - bounds.lo);
    s.ray_epsilon = std::min(std::max(eps, 1e-6f), 1e-2f);
  }

  backend->build_accel(s);
  pathtracer.set_scene(s);
  pathtracer.set_backend_ref(*backend);
  scene = std::move(s);
}

void Renderer::Impl::render(const CameraData &camera, const RenderSettings &settings, bool reset)
{
  const u32 w = settings.width;
  const u32 h = settings.height;
  if (w == 0 || h == 0)
    throw std::runtime_error("invalid frame size");
  if (!scene)
    throw std::runtime_error("no scene");

  if (w != width || h != height) {
    accum_color = Kokkos::View<Vec3 **, Kokkos::LayoutRight>("frame_accum_color", h, w);
    accum_alpha = Kokkos::View<f32 **, Kokkos::LayoutRight>("frame_accum_alpha", h, w);
    first_depth = Kokkos::View<f32 **, Kokkos::LayoutRight>("frame_depth", h, w);
    first_normal = Kokkos::View<Vec3 **, Kokkos::LayoutRight>("frame_normal", h, w);
    first_albedo = Kokkos::View<Vec3 **, Kokkos::LayoutRight>("frame_albedo", h, w);
    width = w;
    height = h;
    reset = true;
  }

  // Background image.
  const size_t bg_size = size_t(settings.background_width) * settings.background_height * 4;
  if (settings.background_image && bg_size > 0) {
    const bool changed = background_copy.size() != bg_size
        || background_image.extent(0) != settings.background_height
        || background_image.extent(1) != settings.background_width
        || std::memcmp(background_copy.data(), settings.background_image, bg_size * sizeof(float)) != 0;
    if (changed) {
      background_copy.assign(settings.background_image, settings.background_image + bg_size);
      background_image = Kokkos::View<Vec4 **, Kokkos::LayoutRight>(
          "frame_background", settings.background_height, settings.background_width);
      upload(background_image, [&](auto &host) {
        for (u32 y = 0; y < settings.background_height; ++y)
          for (u32 x = 0; x < settings.background_width; ++x) {
            const float *p = &background_copy[(size_t(y) * settings.background_width + x) * 4];
            host(y, x) = Vec4{p[0], p[1], p[2], p[3]};
          }
      });
    }
  } else if (!background_copy.empty()) {
    background_copy.clear();
    background_image = Kokkos::View<Vec4 **, Kokkos::LayoutRight>();
  }

  const Vec3 position = vec3_of(camera.position);
  Vec3 direction = normalize(vec3_of(camera.direction));
  if (length(direction) == 0.f)
    direction = Vec3{0.f, 0.f, -1.f};
  Vec3 up = vec3_of(camera.up);
  if (length(cross(direction, up)) < 1e-6f)
    up = Kokkos::fabs(direction.y) < 0.9f ? Vec3{0.f, 1.f, 0.f} : Vec3{1.f, 0.f, 0.f};

  const f32 aspect = camera.aspect > 0.f ? camera.aspect : f32(w) / f32(h);
  Camera cam;
  if (camera.orthographic) {
    cam = Camera::make_orthographic(position, position + direction, up, camera.height, aspect);
  } else {
    const f32 fovy_degrees = camera.fovy * 180.f / 3.14159265358979f;
    const bool lens = camera.aperture_radius > 0.f && camera.focus_distance > 0.f;
    cam = Camera::make_perspective(position, position + direction, up, fovy_degrees, aspect,
        lens ? 2.f * camera.aperture_radius : 0.f, lens ? camera.focus_distance : 1.f);
  }
  cam.clip_near = camera.clip_near > 0.f ? camera.clip_near : 0.f;
  cam.clip_far = camera.clip_far < 1e30f ? camera.clip_far : 1e30f;

  RenderParams &params = pathtracer.params;
  params.width = w;
  params.height = h;
  params.samples_per_pixel = std::max<u32>(settings.pixel_samples, 1);
  params.max_depth = std::max<u32>(settings.max_depth, 1);
  params.sample_offset = sample_counter;
  params.background_color = vec3_of(settings.background);
  params.background_alpha = settings.background[3];
  params.background_image = background_image;
  params.ambient_color = vec3_of(settings.ambient_color);
  params.ambient_radiance = settings.ambient_radiance;
  params.adaptive_sampling = settings.adaptive_sampling;
  params.throughput_clamp = settings.throughput_clamp;
  params.max_diffuse_bounces = settings.max_diffuse_bounces;
  params.max_glossy_bounces = settings.max_glossy_bounces;
  params.max_transmission_bounces = settings.max_transmission_bounces;
  params.max_volume_bounces = settings.max_volume_bounces;
  // as in Cycles the value limits the mean of the channels
  params.clamp_direct_sum = 3.f * std::max(settings.clamp_direct, 0.f);
  params.clamp_indirect_sum = 3.f * std::max(settings.clamp_indirect, 0.f);
  params.bottom_up = true;
  for (int i = 0; i < 4; ++i)
    params.image_region[i] = camera.image_region[i];
  pathtracer.camera = cam;

  const RenderResult result = pathtracer.render();

  const u32 spp = params.samples_per_pixel;
  sample_counter += spp;

  const bool first = reset || samples == 0;
  detail::accumulate_frame(
      accum_color, accum_alpha, first_depth, first_normal, first_albedo, result, f32(spp), first, w, h);

  samples = first ? spp : samples + spp;
}

namespace {

// Renderers with a path tracer, for Renderer::shutdown(). Never destroyed.
struct RendererRegistry
{
  std::mutex mutex;
  std::vector<Renderer *> renderers;

  static RendererRegistry &instance()
  {
    static RendererRegistry *registry = new RendererRegistry();
    return *registry;
  }
};

} // namespace

Renderer::Renderer() = default;

Renderer::~Renderer()
{
  Impl *impl = nullptr;
  {
    auto &registry = RendererRegistry::instance();
    std::lock_guard<std::mutex> lock(registry.mutex);
    auto &list = registry.renderers;
    list.erase(std::remove(list.begin(), list.end(), this), list.end());
    impl = m_impl;
    m_impl = nullptr;
  }
  if (!impl)
    return;
  std::string error;
  // Kokkos views have to be released on the Kokkos thread.
  KokkosThread::instance().run([impl] { delete impl; }, error);
}

void Renderer::shutdown()
{
  if (!g_kokkos_initialized || g_kokkos_finalized || g_exit_safe)
    return;

  // Take the path tracers away from all renderers, the devices may be
  // released later (or never). The list is owned by whoever releases them.
  auto *impls = new std::vector<Impl *>();
  {
    auto &registry = RendererRegistry::instance();
    std::lock_guard<std::mutex> lock(registry.mutex);
    for (Renderer *renderer : registry.renderers) {
      if (renderer->m_impl)
        impls->push_back(renderer->m_impl);
      renderer->m_impl = nullptr;
    }
    registry.renderers.clear();
  }

#if defined(KOKKOS_ENABLE_THREADS)
  // The Threads back-end can only be finalized on the Kokkos thread. This
  // thread holds the exit lock of the C runtime while exit handlers run: should
  // the task ever need that lock (it registers an exit handler) it cannot
  // finish, so the wait is limited. The process is then ended by DllMain below.
  KokkosThread::instance().run_with_timeout(
      [impls] {
        for (Impl *impl : *impls)
          delete impl;
        delete impls;
        KokkosThread::instance().finalize_kokkos();
      },
      std::chrono::seconds(10));
#else
  // Release everything and finalize Kokkos on this thread: the Kokkos thread
  // is idle, and work done here cannot deadlock on the exit lock of the C
  // runtime (finalizing the CUDA back-end registers exit handlers).
  if (!KokkosThread::instance().lock_for_shutdown())
    return;
  for (Impl *impl : *impls)
    delete impl;
  delete impls;
  KokkosThread::instance().finalize_kokkos();
#endif
}

bool Renderer::initialize(std::string &error)
{
  if (m_impl)
    return true;
  Impl *impl = nullptr;
  const bool ok = KokkosThread::instance().run(
      [&impl] {
        KokkosThread::instance().initialize_kokkos();
        auto created = std::make_unique<Impl>();
        created->backend = create_best_backend();
        created->backend_name = created->backend->name();
        created->execution_space = Kokkos::DefaultExecutionSpace::name();
        impl = created.release();
      },
      error);
  if (ok) {
    auto &registry = RendererRegistry::instance();
    std::lock_guard<std::mutex> lock(registry.mutex);
    m_impl = impl;
    registry.renderers.push_back(this);
  }
  return ok;
}

std::string Renderer::backend_name() const
{
  return m_impl ? m_impl->backend_name : std::string();
}

std::string Renderer::execution_space() const
{
  return m_impl ? m_impl->execution_space : std::string();
}

bool Renderer::set_scene(const SceneData &scene, std::string &error)
{
  if (!initialize(error))
    return false;
  Impl *impl = m_impl;
  m_samples = 0;
  return KokkosThread::instance().run(
      [impl, &scene] {
        impl->samples = 0;
        impl->build_scene(scene);
      },
      error);
}

bool Renderer::render(const CameraData &camera, const RenderSettings &settings, bool reset, std::string &error)
{
  if (!initialize(error))
    return false;
  Impl *impl = m_impl;
  uint32_t samples = 0;
  const bool ok = KokkosThread::instance().run(
      [impl, &camera, &settings, reset, &samples] {
        impl->render(camera, settings, reset);
        samples = impl->samples;
      },
      error);
  if (ok)
    m_samples = samples;
  return ok;
}

uint32_t Renderer::accumulated_samples() const
{
  return m_samples;
}

bool Renderer::read_color(float *rgba, std::string &error)
{
  if (!m_impl || !rgba)
    return false;
  Impl *impl = m_impl;
  return KokkosThread::instance().run(
      [impl, rgba] {
        if (impl->samples == 0)
          throw std::runtime_error("no frame has been rendered");
        auto color = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, impl->accum_color);
        auto alpha = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, impl->accum_alpha);
        const float inv = 1.f / float(impl->samples);
        const size_t w = impl->width;
        for (size_t y = 0; y < impl->height; ++y)
          for (size_t x = 0; x < w; ++x) {
            float *out = rgba + (y * w + x) * 4;
            const Vec3 c = color(y, x);
            out[0] = c.x * inv;
            out[1] = c.y * inv;
            out[2] = c.z * inv;
            const float a = alpha(y, x) * inv;
            out[3] = a < 0.f ? 0.f : (a > 1.f ? 1.f : a);
          }
      },
      error);
}

bool Renderer::read_depth(float *out, std::string &error)
{
  if (!m_impl || !out)
    return false;
  Impl *impl = m_impl;
  return KokkosThread::instance().run(
      [impl, out] {
        if (impl->samples == 0)
          throw std::runtime_error("no frame has been rendered");
        auto depth = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, impl->first_depth);
        const size_t w = impl->width;
        for (size_t y = 0; y < impl->height; ++y)
          for (size_t x = 0; x < w; ++x) {
            const float d = depth(y, x);
            out[y * w + x] = d < 1e29f ? d : std::numeric_limits<float>::infinity();
          }
      },
      error);
}

namespace {

template <typename ViewType>
void read_vec3(const ViewType &view, size_t w, size_t h, float *out)
{
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
  for (size_t y = 0; y < h; ++y)
    for (size_t x = 0; x < w; ++x) {
      const Vec3 v = host(y, x);
      out[(y * w + x) * 3 + 0] = v.x;
      out[(y * w + x) * 3 + 1] = v.y;
      out[(y * w + x) * 3 + 2] = v.z;
    }
}

} // namespace

bool Renderer::read_normal(float *out, std::string &error)
{
  if (!m_impl || !out)
    return false;
  Impl *impl = m_impl;
  return KokkosThread::instance().run(
      [impl, out] {
        if (impl->samples == 0)
          throw std::runtime_error("no frame has been rendered");
        read_vec3(impl->first_normal, impl->width, impl->height, out);
      },
      error);
}

bool Renderer::read_albedo(float *out, std::string &error)
{
  if (!m_impl || !out)
    return false;
  Impl *impl = m_impl;
  return KokkosThread::instance().run(
      [impl, out] {
        if (impl->samples == 0)
          throw std::runtime_error("no frame has been rendered");
        read_vec3(impl->first_albedo, impl->width, impl->height, out);
      },
      error);
}

} // namespace photon::bridge

#ifdef _WIN32

// Safety net for processes that end without running the exit handlers of the C
// runtime (ExitProcess() called directly, _exit(), ...): Renderer::shutdown()
// has not run then, all other threads are gone, and the static destructors of
// Kokkos that run after this function would wait for its worker threads forever
// (Threads back-end) or fail to release device memory (CUDA back-end). The
// process is ended here instead, with the exit code it was given.
BOOL WINAPI DllMain(HINSTANCE, DWORD reason, LPVOID reserved)
{
  if (reason == DLL_PROCESS_DETACH && reserved != nullptr && photon::bridge::g_kokkos_initialized
      && !photon::bridge::g_exit_safe) {
    _flushall();
    DWORD code = 0;
    if (!GetExitCodeProcess(GetCurrentProcess(), &code) || code == STILL_ACTIVE)
      code = 0;
    TerminateProcess(GetCurrentProcess(), code);
  }
  return TRUE;
}

#endif
