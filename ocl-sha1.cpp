/* SHA1 OpenCL: Copyright (c) Volcano Authors 2018.
 * Licensed under the GPLv3.
 *
 * This is the CPU-side companion code for sha1.cl (a sha1 implementation in
 * OpenCL)
 */

#include "ocl-sha1.h"
#include "ocl-device.h"
#include "ocl-program.h"
#include "hashapi.h"

namespace gitmine {

static const uint32_t sha1_IV[] = {
  0x67452301,
  0xefcdab89,
  0x98badcfe,
  0x10325476,
  0xc3d2e1f0,
};

static const uint64_t blake2b_IV[8] = {
  0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
  0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
  0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
  0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
};

int testGPUsha1(OpenCLdev& dev, OpenCLprog& p, const std::vector<char>& buf,
               Sha1Hash& shaout) {
  std::vector<SHA1context> out(1);
  std::vector<SHA1context> ctx(1);
  ctx.at(0).len = buf.size();
  ctx.at(0).bytesRemaining = buf.size();
  // Initialize context hash constants (initialization vector).
  // Usually the sha1() call does this, but this code sets up the GPU call
  // so it is *only* the SHA1_Update() part of the algorithm.
  for (size_t i = 0; i < SHA_DIGEST_LEN; i++) {
    ctx.at(0).hash[i] = sha1_IV[i];
  }

  for (size_t i = 0; i < B2H_DIGEST_LEN; i++) {
    ctx.at(0).b2hash[i] = blake2b_IV[i];
    ctx.at(0).b2iv[i] = blake2b_IV[i];
  }

  std::vector<SHA1buffer> cpubuf;
  for (size_t i = 0; i < buf.size(); ) {
    cpubuf.emplace_back();
    size_t len = sizeof(SHA1buffer);
    if (buf.size() - i < len) {
      len = buf.size() - i;
      memset(&cpubuf.back(), 0, sizeof(SHA1buffer));
    }
    memcpy(&cpubuf.back(), &buf.at(i), len);
    i += len;
  }
  if (cpubuf.size() != (buf.size() + sizeof(SHA1buffer) - 1) / sizeof(SHA1buffer)) {
    fprintf(stderr, "BUG: cpubuf %zu, want %zu\n", cpubuf.size(),
            buf.size() / sizeof(SHA1buffer));
    return 1;
  }

  OpenCLmem gpuctx(dev);
  if (gpuctx.create(CL_MEM_READ_ONLY, sizeof(ctx[0]) * ctx.size()) ||
      p.setArg(0, gpuctx)) {
    return 1;
  }
  OpenCLmem gpubuf(dev);
  if (gpubuf.create(CL_MEM_READ_ONLY, sizeof(cpubuf[0]) * cpubuf.size()) ||
      p.setArg(1, gpubuf)) {
    return 1;
  }

  OpenCLmem gpuout(dev);
  if (gpuout.create(CL_MEM_WRITE_ONLY, sizeof(out[0]) * out.size()) ||
      p.setArg(2, gpuout)) {
    return 1;
  }

  OpenCLqueue q(dev);
  if (q.open() || q.writeBuffer(gpuctx, ctx) ||
      q.writeBuffer(gpubuf, cpubuf)) {
    return 1;
  }
  OpenCLevent completeEvent;
  std::vector<size_t> global_work_size{ ctx.size() };
  if (q.NDRangeKernel(p, global_work_size.size(), NULL, 
                      global_work_size.data(), NULL, &completeEvent.handle)) {
    return 1;
  }
  completeEvent.waitForSignal();

  if (q.readBuffer(gpuout, out) || q.finish()) {
    return 1;
  }
  memcpy(shaout.result, out.at(0).hash, sizeof(shaout.result));
  return 0;
}

int findOnGPU(OpenCLdev& dev, OpenCLprog& p, const std::vector<char>& buf,
              Sha1Hash& shaout) {
  std::vector<SHA1context> out(1);
  std::vector<SHA1context> b2hout(1);
  std::vector<SHA1context> ctx(1);
  ctx.at(0).len = buf.size();
  ctx.at(0).bytesRemaining = buf.size();
  // Initialize context hash constants (initialization vector).
  // Usually the sha1() call does this, but this code sets up the GPU call
  // so it is *only* the SHA1_Update() part of the algorithm.
  for (size_t i = 0; i < SHA_DIGEST_LEN; i++) {
    ctx.at(0).hash[i] = sha1_IV[i];
  }

  for (size_t i = 0; i < B2H_DIGEST_LEN; i++) {
    ctx.at(0).b2hash[i] = blake2b_IV[i];
    ctx.at(0).b2iv[i] = blake2b_IV[i];
  }

  std::vector<SHA1buffer> cpubuf;
  for (size_t i = 0; i < buf.size(); ) {
    cpubuf.emplace_back();
    size_t len = sizeof(SHA1buffer);
    if (buf.size() - i < len) {
      len = buf.size() - i;
      memset(&cpubuf.back(), 0, sizeof(SHA1buffer));
    }
    memcpy(&cpubuf.back(), &buf.at(i), len);
    i += len;
  }
  if (cpubuf.size() != (buf.size() + sizeof(SHA1buffer) - 1) / sizeof(SHA1buffer)) {
    fprintf(stderr, "BUG: cpubuf %zu, want %zu\n", cpubuf.size(),
            buf.size() / sizeof(SHA1buffer));
    return 1;
  }

  OpenCLmem gpuctx(dev);
  if (gpuctx.create(CL_MEM_READ_ONLY, sizeof(ctx[0]) * ctx.size()) ||
      p.setArg(0, gpuctx)) {
    return 1;
  }
  OpenCLmem gpubuf(dev);
  if (gpubuf.create(CL_MEM_READ_ONLY, sizeof(cpubuf[0]) * cpubuf.size()) ||
      p.setArg(1, gpubuf)) {
    return 1;
  }

  OpenCLmem gpuoutsha(dev);
  if (gpuoutsha.create(CL_MEM_WRITE_ONLY, sizeof(out[0]) * out.size()) ||
      p.setArg(2, gpuoutsha)) {
    return 1;
  }

  OpenCLqueue q(dev);
  if (q.open() || q.writeBuffer(gpuctx, ctx) ||
      q.writeBuffer(gpubuf, cpubuf)) {
    return 1;
  }
  OpenCLevent completeEventSha;
  std::vector<size_t> global_work_size{ ctx.size() };
  if (q.NDRangeKernel(p, global_work_size.size(), NULL,
                      global_work_size.data(), NULL,
                      &completeEventSha.handle)) {
    return 1;
  }
  completeEventSha.waitForSignal();

  if (q.readBuffer(gpuoutsha, out) || q.finish()) {
    return 1;
  }
  memcpy(shaout.result, out.at(0).hash, sizeof(shaout.result));
  char shabuf[1024];
  if (shaout.dump(shabuf, sizeof(shabuf))) {
    return 1;
  }
  fprintf(stderr, "found sha1: %s\n", shabuf);
  Blake2Hash b2h;
  memcpy(b2h.result, out.at(0).b2hash, sizeof(b2h.result));
  if (b2h.dump(shabuf, sizeof(shabuf))) {
    return 1;
  }
  fprintf(stderr, "blake2: %s\n", shabuf);
  return 0;
}

}  // namespace git-mine
