#pragma once

#include "photon/anari/SceneObjects.h"
// helium
#include "helium/BaseFrame.h"

#include <vector>

namespace photon::anari_device {

struct Frame : public helium::BaseFrame
{
  Frame(PhotonGlobalState *s);
  ~Frame() override;

  bool isValid() const override;

  bool getProperty(const std::string_view &name,
      ANARIDataType type,
      void *ptr,
      uint64_t size,
      uint32_t flags) override;

  void commitParameters() override;
  void finalize() override;

  // Rendering is synchronous: the frame is complete when this returns.
  void renderFrame() override;

  void *map(std::string_view channel, uint32_t *width, uint32_t *height, ANARIDataType *pixelType) override;
  void unmap(std::string_view channel) override;
  int frameReady(ANARIWaitMask m) override;
  void discard() override;

 private:
  PhotonGlobalState *deviceState() const;
  void allocateBuffers();
  void convertColor();

  helium::IntrusivePtr<Renderer> m_renderer;
  helium::IntrusivePtr<Camera> m_camera;
  helium::IntrusivePtr<World> m_world;

  uint2 m_size{0u, 0u};
  anari::DataType m_colorType{ANARI_UNKNOWN};
  anari::DataType m_depthType{ANARI_UNKNOWN};
  anari::DataType m_normalType{ANARI_UNKNOWN};
  anari::DataType m_albedoType{ANARI_UNKNOWN};
  bool m_accumulation{false};

  // Linear RGBA of the last render and the buffers handed out by map().
  std::vector<float> m_linearColor;
  std::vector<uint8_t> m_colorBytes;
  std::vector<float> m_depth;
  std::vector<float> m_normal;
  std::vector<float> m_albedo;

  float m_duration{0.f};
  helium::TimeStamp m_lastRendered{0};
};

} // namespace photon::anari_device

PHOTON_ANARI_TYPEFOR_SPECIALIZATION(photon::anari_device::Frame *, ANARI_FRAME);
