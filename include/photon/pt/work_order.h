#pragma once

#include <Kokkos_Core.hpp>

#include "photon/pt/math.h"

namespace photon::pt {

// Order in which a kernel visits the items of a wavefront (pixels, rays).
//
// The cost per item varies a lot (background next to volumes or glass) and the
// expensive items are neighbours in the image. Host back-ends give every thread
// one contiguous part of the index range, so visiting the items in image order
// leaves most threads idle. Here the items are grouped in blocks, and the
// blocks are visited in a permuted order that spreads every part of the range
// over the whole image; inside a block the memory access stays sequential.
//
// Usage: parallel_for over [0, range()), skip indices whose item() is negative.
struct WorkOrder
{
  u32 count{0};
  u32 block{256};
  u32 block_count{0};
  u32 stride{1};

  WorkOrder() = default;

  explicit WorkOrder(u32 item_count) : count(item_count)
  {
    block_count = (count + block - 1) / block;
    // A stride without a common factor with the number of blocks visits every
    // block exactly once; one near the golden ratio of the count spreads
    // consecutive blocks well.
    stride = 1;
    if (block_count > 2) {
      u32 candidate = u32(0.6180339887 * double(block_count));
      if (candidate < 1)
        candidate = 1;
      for (; candidate < block_count; ++candidate) {
        u32 a = candidate;
        u32 b = block_count;
        while (b != 0) {
          const u32 r = a % b;
          a = b;
          b = r;
        }
        if (a == 1) {
          stride = candidate;
          break;
        }
      }
    }
  }

  u32 range() const
  {
    return block_count * block;
  }

  KOKKOS_FUNCTION int item(int i) const
  {
    const u32 b = u32(i) / block;
    const u32 o = u32(i) % block;
    const u32 permuted = u32((static_cast<unsigned long long>(b) * stride) % block_count);
    const u32 index = permuted * block + o;
    return index < count ? int(index) : -1;
  }
};

} // namespace photon::pt
