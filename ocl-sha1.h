/* SHA1 OpenCL: Copyright (c) Volcano Authors 2018.
 * Licensed under the GPLv3.
 *
 * This is the CPU-side companion code for sha1.cl (a sha1 implementation in
 * OpenCL)
 */

#include "ocl-device.h"
#include "ocl-program.h"
#include "hashapi.h"

#pragma once

namespace gitmine {

struct SHA1buffer {
  uint32_t buffer[64/sizeof(uint32_t)];
};

#define B2H_DIGEST_LEN (8)

#ifndef SHA_DIGEST_LEN
#define SHA_DIGEST_LEN (5)
#elif SHA_DIGEST_LEN != 5
#error SHA_DIGEST_LEN must be 5
#endif
struct SHA1context {
  uint32_t len;  // The overall length of the message to digest.
  uint32_t bytesRemaining;
  uint32_t hash[SHA_DIGEST_LEN];
  uint64_t b2hash[B2H_DIGEST_LEN];
  uint64_t b2iv[B2H_DIGEST_LEN];
};

// testGPUsha1 will perform a full end-to-end SHA1. The result should
// exactly match what the CPU gets.
int testGPUsha1(OpenCLdev& dev, OpenCLprog& psha, const std::vector<char>& buf,
                Sha1Hash& out);

int findOnGPU(OpenCLdev& dev, OpenCLprog& p, const std::vector<char>& buf,
              Sha1Hash& out);

}  // namespace git-mine
