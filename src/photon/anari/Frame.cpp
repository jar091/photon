#include "photon/anari/Frame.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>

namespace photon::anari_device {

namespace {

uint8_t toByte(float v)
{
  if (!(v > 0.f))
    return 0;
  return uint8_t(std::min(v, 1.f) * 255.f + 0.5f);
}

float linearToSrgb(float v)
{
  if (!(v > 0.f))
    return 0.f;
  v = std::min(v, 1.f);
  return v <= 0.0031308f ? 12.92f * v : 1.055f * std::pow(v, 1.f / 2.4f) - 0.055f;
}

} // namespace

Frame::Frame(PhotonGlobalState *s) : helium::BaseFrame(s) {}

Frame::~Frame()
{
  auto *state = deviceState();
  if (state->lastFrame == this)
    state->lastFrame = nullptr;
}

bool Frame::isValid() const
{
  return m_renderer && m_renderer->isValid() && m_camera && m_camera->isValid() && m_world && m_world->isValid()
      && m_size.x > 0 && m_size.y > 0;
}

PhotonGlobalState *Frame::deviceState() const
{
  return (PhotonGlobalState *)helium::BaseObject::m_state;
}

bool Frame::getProperty(const std::string_view &name, ANARIDataType type, void *ptr, uint64_t, uint32_t)
{
  if (type == ANARI_FLOAT32 && name == "duration") {
    helium::writeToVoidP(ptr, m_duration);
    return true;
  }
  if (type == ANARI_INT32 && name == "numSamples") {
    helium::writeToVoidP(ptr, int(deviceState()->renderer.accumulated_samples()));
    return true;
  }
  if (type == ANARI_BOOL && name == "valid") {
    helium::writeToVoidP(ptr, isValid());
    return true;
  }
  return false;
}

void Frame::commitParameters()
{
  m_renderer = getParamObject<Renderer>("renderer");
  m_camera = getParamObject<Camera>("camera");
  m_world = getParamObject<World>("world");
  m_colorType = getParam<anari::DataType>("channel.color", ANARI_UNKNOWN);
  m_depthType = getParam<anari::DataType>("channel.depth", ANARI_UNKNOWN);
  m_normalType = getParam<anari::DataType>("channel.normal", ANARI_UNKNOWN);
  m_albedoType = getParam<anari::DataType>("channel.albedo", ANARI_UNKNOWN);
  m_size = getParam<uint2>("size", uint2(0u, 0u));
  m_accumulation = getParam<bool>("accumulation", false);
}

void Frame::finalize()
{
  if (!m_renderer)
    reportMessage(ANARI_SEVERITY_WARNING, "missing required parameter 'renderer' on frame");
  if (!m_camera)
    reportMessage(ANARI_SEVERITY_WARNING, "missing required parameter 'camera' on frame");
  if (!m_world)
    reportMessage(ANARI_SEVERITY_WARNING, "missing required parameter 'world' on frame");
  if (m_size.x == 0 || m_size.y == 0)
    reportMessage(ANARI_SEVERITY_WARNING, "invalid frame dimensions");

  if (m_colorType != ANARI_UNKNOWN && m_colorType != ANARI_FLOAT32_VEC4 && m_colorType != ANARI_UFIXED8_VEC4
      && m_colorType != ANARI_UFIXED8_RGBA_SRGB) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "unsupported type %s for 'channel.color', using UFIXED8_RGBA_SRGB",
        anari::toString(m_colorType));
    m_colorType = ANARI_UFIXED8_RGBA_SRGB;
  }
  if (m_depthType != ANARI_UNKNOWN && m_depthType != ANARI_FLOAT32) {
    reportMessage(ANARI_SEVERITY_WARNING, "'channel.depth' has to be FLOAT32, the channel is disabled");
    m_depthType = ANARI_UNKNOWN;
  }
  if (m_normalType != ANARI_UNKNOWN && m_normalType != ANARI_FLOAT32_VEC3) {
    reportMessage(ANARI_SEVERITY_WARNING, "'channel.normal' has to be FLOAT32_VEC3, the channel is disabled");
    m_normalType = ANARI_UNKNOWN;
  }
  if (m_albedoType != ANARI_UNKNOWN && m_albedoType != ANARI_FLOAT32_VEC3) {
    reportMessage(ANARI_SEVERITY_WARNING, "'channel.albedo' has to be FLOAT32_VEC3, the channel is disabled");
    m_albedoType = ANARI_UNKNOWN;
  }

  allocateBuffers();
}

void Frame::allocateBuffers()
{
  const size_t pixels = size_t(m_size.x) * size_t(m_size.y);
  m_linearColor.assign(pixels * 4, 0.f);
  m_colorBytes.assign(m_colorType == ANARI_FLOAT32_VEC4 ? 0 : pixels * 4, 0);
  m_depth.assign(m_depthType == ANARI_FLOAT32 ? pixels : 0, std::numeric_limits<float>::infinity());
  m_normal.assign(m_normalType == ANARI_FLOAT32_VEC3 ? pixels * 3 : 0, 0.f);
  m_albedo.assign(m_albedoType == ANARI_FLOAT32_VEC3 ? pixels * 3 : 0, 0.f);
}

void Frame::convertColor()
{
  if (m_colorType == ANARI_FLOAT32_VEC4 || m_colorBytes.empty())
    return;
  const size_t pixels = m_linearColor.size() / 4;
  const bool srgb = m_colorType == ANARI_UFIXED8_RGBA_SRGB;
  for (size_t i = 0; i < pixels; ++i) {
    const float *in = &m_linearColor[i * 4];
    uint8_t *out = &m_colorBytes[i * 4];
    for (int c = 0; c < 3; ++c)
      out[c] = toByte(srgb ? linearToSrgb(in[c]) : in[c]);
    out[3] = toByte(in[3]);
  }
}

void Frame::renderFrame()
{
  auto *state = deviceState();
  state->commitBuffer.flush();

  const auto start = std::chrono::steady_clock::now();

  if (!isValid()) {
    reportMessage(ANARI_SEVERITY_ERROR, "skipping render of incomplete frame object");
    allocateBuffers();
    return;
  }

  std::string error;
  bool reset = !m_accumulation;

  // Scene: rebuilt when anything in the world changed.
  if (state->renderedWorld != m_world.ptr || state->renderedWorldBuilt <= state->sceneLastChanged) {
    bridge::SceneData scene;
    m_world->buildScene(scene);
    state->renderedWorld = nullptr;
    if (!state->renderer.set_scene(scene, error)) {
      reportMessage(ANARI_SEVERITY_ERROR, "photon cannot build the scene: %s", error.c_str());
      allocateBuffers();
      return;
    }
    state->renderedWorld = m_world.ptr;
    state->renderedWorldBuilt = helium::newTimeStamp();
    reset = true;
    reportMessage(ANARI_SEVERITY_INFO,
        "photon scene: %zu triangles, %zu materials, %zu lights, %zu volumes, ray back-end '%s' on Kokkos %s",
        scene.indices.size() / 3,
        scene.materials.size(),
        scene.lights.size(),
        scene.volumes.size(),
        state->renderer.backend_name().c_str(),
        state->renderer.execution_space().c_str());
  }

  // The accumulation restarts when any object was committed since the last
  // frame (camera, renderer, frame, ...) or another frame used the device.
  if (state->lastFrame != this || state->commitBuffer.lastObjectFinalization() > m_lastRendered)
    reset = true;
  state->lastFrame = this;

  bridge::RenderSettings settings;
  settings.width = m_size.x;
  settings.height = m_size.y;
  m_renderer->fillSettings(settings);

  const bridge::CameraData camera = m_camera->data();

  if (m_linearColor.size() != size_t(m_size.x) * m_size.y * 4)
    allocateBuffers();

  bool ok = state->renderer.render(camera, settings, reset, error);
  ok = ok && state->renderer.read_color(m_linearColor.data(), error);
  if (ok && !m_depth.empty())
    ok = state->renderer.read_depth(m_depth.data(), error);
  if (ok && !m_normal.empty())
    ok = state->renderer.read_normal(m_normal.data(), error);
  if (ok && !m_albedo.empty())
    ok = state->renderer.read_albedo(m_albedo.data(), error);

  if (!ok) {
    reportMessage(ANARI_SEVERITY_ERROR, "photon failed to render the frame: %s", error.c_str());
    allocateBuffers();
    return;
  }

  convertColor();

  m_lastRendered = helium::newTimeStamp();
  m_duration = std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count();
}

void *Frame::map(std::string_view channel, uint32_t *width, uint32_t *height, ANARIDataType *pixelType)
{
  *width = m_size.x;
  *height = m_size.y;

  if (channel == "channel.color" && m_colorType != ANARI_UNKNOWN) {
    *pixelType = m_colorType;
    if (m_colorType == ANARI_FLOAT32_VEC4 && !m_linearColor.empty())
      return m_linearColor.data();
    if (!m_colorBytes.empty())
      return m_colorBytes.data();
  } else if (channel == "channel.depth" && !m_depth.empty()) {
    *pixelType = ANARI_FLOAT32;
    return m_depth.data();
  } else if (channel == "channel.normal" && !m_normal.empty()) {
    *pixelType = ANARI_FLOAT32_VEC3;
    return m_normal.data();
  } else if (channel == "channel.albedo" && !m_albedo.empty()) {
    *pixelType = ANARI_FLOAT32_VEC3;
    return m_albedo.data();
  }

  *width = 0;
  *height = 0;
  *pixelType = ANARI_UNKNOWN;
  return nullptr;
}

void Frame::unmap(std::string_view)
{
  // no-op
}

int Frame::frameReady(ANARIWaitMask)
{
  return 1;
}

void Frame::discard()
{
  // no-op
}

} // namespace photon::anari_device

PHOTON_ANARI_TYPEFOR_DEFINITION(photon::anari_device::Frame *);
