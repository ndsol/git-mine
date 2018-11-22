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

#define MIN_MATCH_LEN (4)

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

struct B2SHAbuffer {
  uint32_t buffer[64/sizeof(uint32_t)];
};

struct B2SHAconst {
  B2SHAconst() {
    // Initialize context hash constants (initialization vector).
    // Usually the sha1() call does this, but this code sets up the GPU call
    // so it is *only* the SHA1_Update() part of the algorithm.
    for (size_t i = 0; i < SHA_DIGEST_LEN; i++) {
      shaiv[i] = sha1_IV[i];
    }
    for (size_t i = 0; i < B2H_DIGEST_LEN; i++) {
      b2iv[i] = blake2b_IV[i];
    }
  }

  uint64_t b2iv[B2H_DIGEST_LEN];
  uint32_t shaiv[SHA_DIGEST_LEN];

  uint32_t len;  // The overall length of the message to digest.
  uint32_t bytesRemaining;  // Bytes to be digested on the GPU.
  uint32_t buffers;  // Buffers to be digested.
};

struct B2SHAstate {
  B2SHAstate() {
    for (size_t i = 0; i < SHA_DIGEST_LEN; i++) {
      hash[i] = 0;
    }
    for (size_t i = 0; i < B2H_DIGEST_LEN; i++) {
      b2hash[i] = 0;
    }
    counterPos = 0;
    counts = 1;
    matchCount = 0;
    matchLen = MIN_MATCH_LEN;
  }

  uint64_t b2hash[B2H_DIGEST_LEN];
  uint32_t hash[SHA_DIGEST_LEN];

  uint32_t counterPos;
  uint64_t counts;
  uint64_t matchCount;
  uint32_t matchLen;
};

// CPUprep prepares the work for sha1.cl, and holds its output.
// This abstracts the setup/teardown so that actual functions below can drive
// the algorithm.
struct CPUprep {
  CPUprep(OpenCLdev& dev, OpenCLprog& prog, const CommitMessage& commit)
      : dev(dev), prog(prog), q(dev), commit(commit), gpufixed(dev)
      , gpustate(dev), gpubuf(dev), fixed(1), atime_hint(0), ctime_hint(0) {}

  OpenCLdev& dev;
  OpenCLprog& prog;
  OpenCLqueue q;
  const CommitMessage& commit;

  OpenCLmem gpufixed;
  OpenCLmem gpustate;
  OpenCLmem gpubuf;
  OpenCLevent completeEvent;
  std::vector<B2SHAstate> state;
  std::vector<B2SHAconst> fixed;
  long long atime_hint;
  long long ctime_hint;

  int init() {
    if (q.open()) {
      fprintf(stderr, "q.open failed\n");
      return 1;
    }
    return 0;
  }

  // buildGPUbuf creates gpubuf and populates it from commit.
  int buildGPUbuf() {
    if (fixed.size() != 1) {
      fprintf(stderr, "BUG: fixed.size=%zu\n", fixed.size());
      return 1;
    }

    if (atime_hint < commit.author_btime) {
      if (atime_hint) {
        fprintf(stderr, "invalid atime_hint %lld (must be at least %lld)\n",
                atime_hint, commit.author_btime);
      }
      atime_hint = commit.author_btime;
    }
    if (ctime_hint < commit.committer_btime) {
      if (ctime_hint) {
        fprintf(stderr, "invalid ctime_hint %lld (must be at least %lld)\n",
                ctime_hint, commit.committer_btime);
      }
      ctime_hint = commit.committer_btime;
    }
    // Adjust atime across the range atime_work.
    // Then, if no match is found, bump ctime and try again.
    long long atime_work = ctime_hint - atime_hint;

    // buf contains the raw commit bytes.
    CommitMessage noodle(commit);
    noodle.author_btime = atime_hint;
    noodle.author_time = std::to_string(atime_hint);
    noodle.committer_btime = ctime_hint;
    noodle.committer_time = std::to_string(ctime_hint);
    std::vector<char> buf(noodle.header.data(),
                          noodle.header.data() + noodle.header.size());
    std::string s = noodle.toRawString();
    buf.insert(buf.end(), s.c_str(), s.c_str() + s.size());

    // First copy buf into B2SHAbuffer-sized chunks on the CPU.
    std::vector<B2SHAbuffer> cpubuf;
    for (size_t i = 0; i < buf.size(); ) {
      cpubuf.emplace_back();
      size_t len = sizeof(B2SHAbuffer);
      if (buf.size() - i < len) {
        len = buf.size() - i;
        memset(&cpubuf.back(), 0, sizeof(B2SHAbuffer));
      }
      memcpy(&cpubuf.back(), &buf.at(i), len);
      i += len;
    }
    if (cpubuf.size() != (buf.size() + sizeof(B2SHAbuffer) - 1) /
                        sizeof(B2SHAbuffer)) {
      fprintf(stderr, "BUG: cpubuf %zu, want %zu\n", cpubuf.size(),
              buf.size() / sizeof(B2SHAbuffer));
      return 1;
    }
    fixed.at(0).len = buf.size();
    fixed.at(0).bytesRemaining = buf.size();
    fixed.at(0).buffers = cpubuf.size();

    // Now with cpubuf.size() ready, create gpubuf.
    if (gpubuf.getHandle()) {
      // gpubuf already created.
      if (state.size() == 1 && q.writeBuffer(gpubuf.getHandle(), cpubuf)) {
        fprintf(stderr, "writeBuffer(gpubuf) failed while resetting gpubuf\n");
        return 1;
      }
    } else if (gpubuf.createIO(q, cpubuf, state.size())) {
      fprintf(stderr, "gpubuf.createInput failed\n");
      return 1;
    }
    if (state.size() > 1) {
      state.at(0).counterPos = noodle.header.size() + noodle.parent.size() +
                               noodle.author.size() + noodle.author_time.size()
                               - 1;
      state.at(0).counts = float(atime_work) / float(state.size());
      static const uint64_t maxCounts = 2048;
      if (state.at(0).counts > maxCounts) {
        fprintf(stderr, "not enough workers: %lu > %lu\n", state.at(0).counts,
                maxCounts);
        return 1;
      }
      // Copy cpubuf to gpubuf.
      size_t i = 0;
      for (;;) {
        if (q.writeBuffer(gpubuf.getHandle(), cpubuf, i)) {
          fprintf(stderr, "buildGPUbuf: writeBuffer[%zu] failed\n", i);
          return 1;
        }
        i++;
        if (i >= state.size()) break;
        // Build a cpubuf for this worker. This code only runs for i > 0.
        long long atime = commit.author_btime;
        float workMax = state.size();
        long long my_work_start = (float(i) * atime_work) / workMax;
        long long my_work_end = (float(i + 1) * atime_work) / workMax;
        noodle.author_btime = atime + my_work_start;
        noodle.author_time = std::to_string(noodle.author_btime);
        noodle.committer_btime = ctime_hint;
        noodle.committer_time = std::to_string(ctime_hint);
        // counterPos points to the last digit in author.
        state.at(i).counterPos = noodle.header.size() + noodle.parent.size() +
                               noodle.author.size() + noodle.author_time.size()
                               - 1;
        state.at(i).counts = (uint32_t) (my_work_end - my_work_start);

        buf.clear();
        buf.insert(buf.end(), noodle.header.data(),
                  noodle.header.data() + noodle.header.size());
        s = noodle.toRawString();
        buf.insert(buf.end(), s.c_str(), s.c_str() + s.size());
        if (buf.size() != fixed.at(0).len) {
          fprintf(stderr, "BUG: buf.size %zu, want %u\n", buf.size(),
                  fixed.at(0).len);
          return 1;
        }
        cpubuf.clear();
        for (size_t i = 0; i < buf.size(); ) {
          cpubuf.emplace_back();
          size_t len = sizeof(B2SHAbuffer);
          if (buf.size() - i < len) {
            len = buf.size() - i;
            memset(&cpubuf.back(), 0, sizeof(B2SHAbuffer));
          }
          memcpy(&cpubuf.back(), &buf.at(i), len);
          i += len;
        }
      }
    }

    // Copy fixed and state to GPU.
    if (gpufixed.getHandle()) {
      if (!gpustate.getHandle()) {
        fprintf(stderr, "BUG: gpufixed created, and not gpustate?\n");
        return 1;
      }
      // gpufixed and gpustate already created.
      if (q.writeBuffer(gpufixed.getHandle(), fixed) ||
          q.writeBuffer(gpustate.getHandle(), state)) {
        fprintf(stderr, "writeBuffer(gpufixed or gpustate) failed\n");
        return 1;
      }
    } else if (gpufixed.createInput(q, fixed) || gpustate.createIO(q, state)) {
      fprintf(stderr, "gpufixed or gpustate failed\n");
      return 1;
    }
    // Set program arguments.
    if (prog.setArg(0, gpufixed) || prog.setArg(1, gpustate) ||
        prog.setArg(2, gpubuf)) {
      fprintf(stderr, "prog.setArg failed\n");
      return 1;
    }
    return 0;
  }

  int start(const std::vector<size_t>& global_work_size) {
    if (q.NDRangeKernel(prog, global_work_size.size(), NULL, 
                        global_work_size.data(), NULL,
                        &completeEvent.handle)) {
      fprintf(stderr, "NDRangeKernel failed\n");
      return 1;
    }
    return 0;
  }

  int wait() {
    completeEvent.waitForSignal();

    if (gpustate.copyTo(q, state) || q.finish()) {
      fprintf(stderr, "gpustate.copyTo or finish failed\n");
      return 1;
    }
    return 0;
  }
};

// Test that the GPU kernel produces the same hash as the CPU.
static int testGPUsha1(OpenCLdev& dev, OpenCLprog& prog,
                       const CommitMessage& commit) {
  CPUprep prep(dev, prog, commit);
  prep.state.resize(1);
  // Defaults for state.at(0) do not need to be modified.

  Sha1Hash cpusha;
  cpusha.update(commit.header.data(), commit.header.size());
  std::string s = commit.toRawString();
  cpusha.update(s.c_str(), s.size());
  cpusha.flush();

  if (prep.init() || prep.buildGPUbuf()) {
    fprintf(stderr, "prep.init or prep.buildGPUbuf failed\n");
    return 1;
  }
  if (prep.start({ prep.state.size() }) || prep.wait()) {
    fprintf(stderr, "prep.start or prep.wait failed\n");
    return 1;
  }
  Sha1Hash shaout;
  memcpy(shaout.result, prep.state.at(0).hash, sizeof(shaout.result));

  if (memcmp(shaout.result, cpusha.result, sizeof(cpusha.result))) {
    char shabuf[1024];
    if (cpusha.dump(shabuf, sizeof(shabuf))) {
      return 1;
    }
    fprintf(stderr, "CPU sha1: %s\n", shabuf);
    if (shaout.dump(shabuf, sizeof(shabuf))) {
      return 1;
    }
    fprintf(stderr, "GPU sha1: %s - mismatch!\n", shabuf);
    return 1;
  }
  return 0;
}

int findOnGPU(OpenCLdev& dev, OpenCLprog& prog, const CommitMessage& commit,
              long long atime_hint, long long ctime_hint) {
  if (testGPUsha1(dev, prog, commit)) {
    fprintf(stderr, "testGPUsha1 failed\n");
    return 1;
  }

  CPUprep prep(dev, prog, commit);
  prep.atime_hint = atime_hint;
  prep.ctime_hint = ctime_hint;

  // Set context for each worker.
  static const size_t numWorkers = 512*3;
  for (size_t i = 0; i < numWorkers; i++) {
    prep.state.emplace_back();
  }

  if (prep.init()) {
    fprintf(stderr, "prep.init failed\n");
    return 1;
  }
  size_t good = 0;
  while (!good) {
    fprintf(stderr, "try ctime %lld\n", prep.ctime_hint);
    if (prep.buildGPUbuf()) {
      fprintf(stderr, "buildGPUbuf failed\n");
      return 1;
    }
    if (prep.start({ prep.state.size() })) {
      fprintf(stderr, "prep.start failed\n");
      return 1;
    }
    if (prep.wait()) {
      fprintf(stderr, "prep.wait failed\n");
      return 1;
    }

    atime_hint = prep.atime_hint;
    ctime_hint = prep.ctime_hint;
    long long atime_work = ctime_hint - atime_hint;
    float workMax = prep.state.size();
    for (size_t i = 0; i < prep.state.size(); i++) {
      if (prep.state.at(i).matchLen == MIN_MATCH_LEN) {
        continue;
      }
      uint64_t matchCount = prep.state.at(i).matchCount;
      CommitMessage noodle(commit);
      long long atime = commit.author_btime;
      long long my_work_end = (float(i + 1) * atime_work) / workMax;
      noodle.author_btime = atime + my_work_end - matchCount;
      noodle.author_time = std::to_string(noodle.author_btime);
      noodle.committer_btime = ctime_hint;
      noodle.committer_time = std::to_string(ctime_hint);
      fprintf(stderr, "%zu match=%u bytes  atime=%lld  ctime=%lld\n",
              i, prep.state.at(i).matchLen, noodle.author_btime,
              noodle.committer_btime);

      Sha1Hash shaout;
      Blake2Hash b2h;
      noodle.hash(shaout, b2h);
      char shabuf[1024];
      if (shaout.dump(shabuf, sizeof(shabuf))) {
        fprintf(stderr, "shaout.dump failed in findOnGPU\n");
        return 1;
      }
      fprintf(stderr, "%zu sha1: \e[1;31m%.*s\e[0m%s\n", i,
              prep.state.at(i).matchLen * 2, shabuf,
              &shabuf[prep.state.at(i).matchLen * 2]);
      char b2hbuf[1024];
      if (b2h.dump(b2hbuf, sizeof(b2hbuf))) {
        fprintf(stderr, "b2h.dump failed in findOnGPU\n");
        return 1;
      }
      shabuf[prep.state.at(i).matchLen * 2] = 0;
      char* b2hpos = strstr(b2hbuf, shabuf);
      if (b2hpos) {
        good++;
      }
      int b2hlen = b2hpos ? (b2hpos - b2hbuf) : strlen(b2hbuf);
      int b2hMatchLen = prep.state.at(i).matchLen * 2;
      if (b2hlen + b2hMatchLen > (int)strlen(b2hbuf)) {
        b2hMatchLen = strlen(b2hbuf) - b2hlen;
      }
      fprintf(stderr, "%zu blake2: %.*s\e[1;31m%.*s\e[0m%s\n", i, b2hlen,
              b2hbuf, b2hMatchLen, &b2hbuf[b2hlen],
              &b2hbuf[b2hlen + b2hMatchLen]);
    }
    prep.ctime_hint++;
  }
  return 0;
}

}  // namespace git-mine
