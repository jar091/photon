// Renders a triangle through the public ANARI API of the photon device and
// checks the frame and the world bounds.

#include <anari/anari.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>

static void statusCallback(const void *, ANARIDevice, ANARIObject, ANARIDataType, ANARIStatusSeverity sev,
    ANARIStatusCode, const char *msg)
{
  if (sev <= ANARI_SEVERITY_WARNING)
    std::fprintf(stderr, "ANARI(%d): %s\n", int(sev), msg ? msg : "<null>");
}

int main()
{
  ANARILibrary lib = anariLoadLibrary("photon", statusCallback, nullptr);
  assert(lib);
  ANARIDevice d = anariNewDevice(lib, "default");
  assert(d);

  const float positions[9] = {-1.f, -1.f, 0.f, 1.f, -1.f, 0.f, 0.f, 1.f, 0.f};
  const float colors[12] = {1.f, 0.f, 0.f, 1.f, 0.f, 1.f, 0.f, 1.f, 0.f, 0.f, 1.f, 1.f};
  const uint32_t indices[3] = {0u, 1u, 2u};

  ANARIArray1D positionArray = anariNewArray1D(d, positions, nullptr, nullptr, ANARI_FLOAT32_VEC3, 3);
  ANARIArray1D colorArray = anariNewArray1D(d, colors, nullptr, nullptr, ANARI_FLOAT32_VEC4, 3);
  ANARIArray1D indexArray = anariNewArray1D(d, indices, nullptr, nullptr, ANARI_UINT32_VEC3, 1);

  ANARIGeometry geometry = anariNewGeometry(d, "triangle");
  anariSetParameter(d, geometry, "vertex.position", ANARI_ARRAY1D, &positionArray);
  anariSetParameter(d, geometry, "vertex.color", ANARI_ARRAY1D, &colorArray);
  anariSetParameter(d, geometry, "primitive.index", ANARI_ARRAY1D, &indexArray);
  anariCommitParameters(d, geometry);

  ANARIMaterial material = anariNewMaterial(d, "matte");
  anariSetParameter(d, material, "color", ANARI_STRING, "color");
  anariCommitParameters(d, material);

  ANARISurface surface = anariNewSurface(d);
  anariSetParameter(d, surface, "geometry", ANARI_GEOMETRY, &geometry);
  anariSetParameter(d, surface, "material", ANARI_MATERIAL, &material);
  anariCommitParameters(d, surface);

  ANARIArray1D surfaceArray = anariNewArray1D(d, &surface, nullptr, nullptr, ANARI_SURFACE, 1);
  ANARIWorld world = anariNewWorld(d);
  anariSetParameter(d, world, "surface", ANARI_ARRAY1D, &surfaceArray);
  anariCommitParameters(d, world);

  float bounds[6] = {};
  assert(anariGetProperty(d, world, "bounds", ANARI_FLOAT32_BOX3, bounds, sizeof(bounds), ANARI_WAIT));
  assert(std::fabs(bounds[0] + 1.f) < 1e-5f && std::fabs(bounds[3] - 1.f) < 1e-5f);
  assert(std::fabs(bounds[1] + 1.f) < 1e-5f && std::fabs(bounds[4] - 1.f) < 1e-5f);

  ANARICamera camera = anariNewCamera(d, "perspective");
  const float position[3] = {0.f, 0.f, 3.f};
  const float direction[3] = {0.f, 0.f, -1.f};
  const float up[3] = {0.f, 1.f, 0.f};
  anariSetParameter(d, camera, "position", ANARI_FLOAT32_VEC3, position);
  anariSetParameter(d, camera, "direction", ANARI_FLOAT32_VEC3, direction);
  anariSetParameter(d, camera, "up", ANARI_FLOAT32_VEC3, up);
  anariCommitParameters(d, camera);

  ANARIRenderer renderer = anariNewRenderer(d, "default");
  const float background[4] = {0.f, 0.f, 0.f, 0.f};
  const float ambient = 1.f;
  const int samples = 8;
  anariSetParameter(d, renderer, "background", ANARI_FLOAT32_VEC4, background);
  anariSetParameter(d, renderer, "ambientRadiance", ANARI_FLOAT32, &ambient);
  anariSetParameter(d, renderer, "pixelSamples", ANARI_INT32, &samples);
  anariCommitParameters(d, renderer);

  ANARIFrame frame = anariNewFrame(d);
  const uint32_t size[2] = {32u, 32u};
  const ANARIDataType colorType = ANARI_FLOAT32_VEC4;
  const ANARIDataType depthType = ANARI_FLOAT32;
  anariSetParameter(d, frame, "size", ANARI_UINT32_VEC2, size);
  anariSetParameter(d, frame, "channel.color", ANARI_DATA_TYPE, &colorType);
  anariSetParameter(d, frame, "channel.depth", ANARI_DATA_TYPE, &depthType);
  anariSetParameter(d, frame, "world", ANARI_WORLD, &world);
  anariSetParameter(d, frame, "camera", ANARI_CAMERA, &camera);
  anariSetParameter(d, frame, "renderer", ANARI_RENDERER, &renderer);
  anariCommitParameters(d, frame);

  anariRenderFrame(d, frame);
  anariFrameReady(d, frame, ANARI_WAIT);

  uint32_t width = 0, height = 0;
  ANARIDataType type = ANARI_UNKNOWN;
  const float *pixels = static_cast<const float *>(anariMapFrame(d, frame, "channel.color", &width, &height, &type));
  assert(pixels && width == 32u && height == 32u && type == ANARI_FLOAT32_VEC4);

  // The triangle covers the image center, the corners show the background.
  const float *center = pixels + (16u * width + 16u) * 4u;
  assert(center[0] + center[1] + center[2] > 0.1f);
  assert(center[3] > 0.99f);
  const float *corner = pixels + (31u * width) * 4u;
  assert(corner[3] < 0.01f);
  anariUnmapFrame(d, frame, "channel.color");

  // Depth is the distance along the ray, infinite where nothing is hit.
  const float *depth = static_cast<const float *>(anariMapFrame(d, frame, "channel.depth", &width, &height, &type));
  assert(depth && width == 32u && height == 32u && type == ANARI_FLOAT32);
  assert(std::fabs(depth[16u * width + 16u] - 3.f) < 0.05f);
  assert(std::isinf(depth[31u * width]));
  anariUnmapFrame(d, frame, "channel.depth");

  anariRelease(d, frame);
  anariRelease(d, renderer);
  anariRelease(d, camera);
  anariRelease(d, world);
  anariRelease(d, surfaceArray);
  anariRelease(d, surface);
  anariRelease(d, material);
  anariRelease(d, geometry);
  anariRelease(d, indexArray);
  anariRelease(d, colorArray);
  anariRelease(d, positionArray);
  anariRelease(d, d);

  std::printf("ok\n");
  return 0;
}
