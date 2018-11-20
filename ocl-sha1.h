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

#define B2H_DIGEST_LEN (8)

#ifndef SHA_DIGEST_LEN
#define SHA_DIGEST_LEN (5)
#elif SHA_DIGEST_LEN != 5
#error SHA_DIGEST_LEN must be 5
#endif

int findOnGPU(OpenCLdev& dev, OpenCLprog& prog, const CommitMessage& commit,
              long long atime_hint, long long ctime_hint);

}  // namespace git-mine
