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
#include <chrono>

namespace gitmine {

typedef std::chrono::steady_clock Clock;

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
    for (size_t i = 0; i < 4; i++) {
      lastfullpadding[i] = 0;
    }
    for (size_t i = 0; i < 4; i++) {
      lastfulllen[i] = 0;
    }
    for (size_t i = 0; i < 4; i++) {
      zeropaddingandlen[i] = 0;
    }
  }

  uint64_t b2iv[B2H_DIGEST_LEN];
  uint32_t lastfullpadding[4];
  uint32_t lastfulllen[4];
  uint32_t zeropaddingandlen[4];
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
    counterPos = 0;
    counts = 1;
    matchCount = 0;
    matchLen = MIN_MATCH_LEN;
    ctimePos = 0;
    ctimeCount = 1;
  }

  uint32_t hash[SHA_DIGEST_LEN];

  uint32_t counterPos;
  uint64_t counts;
  uint64_t matchCount;
  uint32_t matchLen;
  uint32_t matchCtimeCount;
  uint32_t ctimePos;
  uint32_t ctimeCount;
};

// CPUprep prepares the work for sha1.cl, and holds its output.
// This wraps the memory buffers and basic setup making the algorithm shorter
// to write.
//
// The most important part here is that the code iterates the atime and ctime.
// The atime must be <= ctime, so work is divided up to compute all possible
// atimes without changing ctime, then increment ctime.
//
// There are tricky cases for computing the last few atimes up to the ctime,
// ctime itself changing each time, so that part may not be as efficient.
//
// If there are more workers than atimes, give each worker a single atime and
// divide up a large chunk of ctimes. This produces lots more atimes by running
// up the ctime.
struct CPUprep {
  CPUprep(OpenCLdev& dev, OpenCLprog& prog, OpenCLqueue& q,
          const CommitMessage& commit, long long start_atime,
          long long start_ctime)
      : dev(dev), prog(prog), q(q), commit(commit), gpufixed(dev)
      , gpustate(dev), gpubuf(dev), fixed(1), testOnly(0), wantValidTime(1)
      , prev_work_done(0), total_work_done(0), ctimeCount(1), timesValid(false)
      , global_start_atime(start_atime), global_start_ctime(start_ctime) {}

  OpenCLdev& dev;
  OpenCLprog& prog;
  OpenCLqueue& q;
  const CommitMessage& commit;

  OpenCLmem gpufixed;
  OpenCLmem gpustate;
  OpenCLmem gpubuf;
  OpenCLevent completeEvent;
  std::vector<B2SHAstate> state;
  std::vector<B2SHAstate> result;
  std::vector<B2SHAconst> fixed;
  std::vector<B2SHAbuffer> cpubuf;
  int testOnly;
  int wantValidTime;

  void writePadding(uint32_t* a, size_t len) {
    a[(len/4) & 3] = 0x80 << (24 - (len & 3)*8);
  }

  void writeLen(uint32_t* a, size_t len) {
    a[2] = len >> (32-3);
    a[3] = len << 3;
  }

  void copyCountersFrom(CPUprep& other) {
    global_start_atime = other.global_start_atime;
    global_start_ctime = other.global_start_ctime;
  }

  void markAllCtimeDone() {
    global_start_ctime += ctimeCount;
  }

  void setCtimeCount(unsigned c) {
    ctimeCount = c;
  }

  // getAFirst returns the first atime that should be processed by worker_i.
  long long getAFirst(size_t worker_i) const {
    long long atime_work = global_start_ctime - global_start_atime;
    float fNumWorkers = state.size();

    return global_start_atime + (long long)(
           (float(worker_i) * atime_work) / fNumWorkers);
  }

  // getAEnd returns the atime after the last atime that should be processed.
  long long getAEnd(size_t worker_i) const {
    long long atime_work = global_start_ctime - global_start_atime;
    float fNumWorkers = state.size();

    return global_start_atime + (long long)(
           (float(worker_i + 1) * atime_work) / fNumWorkers);
  }

  // getCFirst returns the first ctime that should be processed by worker_i.
  long long getCFirst(size_t worker_i) const {
    (void)worker_i;
    return global_start_ctime;
  }

  long long getCEnd(size_t worker_i) const {
    (void)worker_i;
    return global_start_ctime + ctimeCount;
  }

  long long getWorkCount() const {
    return total_work_done;
  }

  void saveWorkCountToPrev() {
    prev_work_done = total_work_done;
  }

  long long getWorkSincePrev() const {
    return total_work_done - prev_work_done;
  }

  int allocState(size_t maxWorkers) {
    size_t bufsPerWorker = commit.header.size() + commit.toRawString().length();
    bufsPerWorker = (bufsPerWorker + sizeof(B2SHAbuffer) - 1)
                    / sizeof(B2SHAbuffer);
    std::vector<B2SHAstate> onestate;
    onestate.resize(1);
    if (gpustate.createIO(q, onestate, maxWorkers)) {
      fprintf(stderr, "gpuState.createIO failed: maxWorkers=%zu\n", maxWorkers);
      return 1;
    }
    std::vector<B2SHAbuffer> onebuf;
    onebuf.resize(1);
    if (gpubuf.createIO(q, onebuf, maxWorkers * bufsPerWorker)) {
      fprintf(stderr, "gpubuf.createInput failed (%zu)\n",
              maxWorkers * bufsPerWorker);
      return 1;
    }
    return 0;
  }

  // buildGPUbuf creates gpubuf and populates it from commit.
  // state.size() should be set to the number of kernel executions to divide
  // the work into.
  int buildGPUbuf() {
    saveWorkCountToPrev();
    total_work_done += (global_start_ctime - global_start_atime) * ctimeCount;

    if (global_start_atime < commit.atime() ||
        global_start_ctime < commit.ctime()) {
      fprintf(stderr, "BUG: start_atime %lld < commit %lld\n",
              global_start_atime, commit.atime());
      fprintf(stderr, "     start_ctime %lld < commit %lld\n",
              global_start_ctime, commit.ctime());
      return 1;
    }
    if (fixed.size() != 1) {
      fprintf(stderr, "BUG: fixed.size=%zu\n", fixed.size());
      return 1;
    }
    result.resize(state.size());

    CommitMessage noodle(commit);
    cpubuf.clear();
    for (size_t i = 0; i < state.size(); i++) {
      if (0 && !gpubuf.getHandle() && (i & 8191) == 8191) {
        fprintf(stderr, "cpu: create %6zu/%zu threads\n", i, state.size());
      }
      noodle.set_atime(getAFirst(i));
      noodle.set_ctime(getCFirst(i));

      // counterPos points to the last digit in author.
      state.at(i).counterPos = noodle.header.size() + noodle.parent.size() +
                              noodle.author.size() + noodle.author_time.size()
                              - 1;
      state.at(i).ctimePos = state.at(i).counterPos + noodle.author_tz.size() +
                             noodle.committer.size() +
                             noodle.committer_time.size();
      state.at(i).counts = (uint32_t) (getAEnd(i) - getAFirst(i));
      state.at(i).ctimeCount = ctimeCount;
      if (testOnly) {
        state.at(i).counts = 1;
      }

      // buf contains the raw commit bytes.
      std::vector<char> buf(noodle.header.data(),
                            noodle.header.data() + noodle.header.size());
      {
        std::string s = noodle.toRawString();
        buf.insert(buf.end(), s.c_str(), s.c_str() + s.size());
      }
      if (i != 0 && buf.size() != fixed.at(0).len) {
        fprintf(stderr, "BUG: buf.size %zu, want %u\n", buf.size(),
                fixed.at(0).len);
        return 1;
      }

      // First copy buf into B2SHAbuffer-sized chunks on the CPU.
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

      if (i == 0) {
        // Use buf to find fixed parameters.
        if (cpubuf.size() != (buf.size() + sizeof(B2SHAbuffer) - 1) /
                            sizeof(B2SHAbuffer)) {
          fprintf(stderr, "BUG: cpubuf %zu, want %zu\n", cpubuf.size(),
                  buf.size() / sizeof(B2SHAbuffer));
          return 1;
        }
        fixed.at(0).len = buf.size();
        fixed.at(0).bytesRemaining = buf.size();
        fixed.at(0).buffers = cpubuf.size();
        if ((buf.size() & 63) != 0) {
          writePadding(fixed.at(0).lastfullpadding, buf.size());
          if ((buf.size() & 63) < 56) {
            writeLen(fixed.at(0).lastfulllen, buf.size());
          }
        } else {
          writePadding(fixed.at(0).zeropaddingandlen, buf.size());
          writeLen(fixed.at(0).zeropaddingandlen, buf.size());
        }
      }
    }

    // Now create gpubuf and copy cpubuf to it.
    if (!gpubuf.getHandle()) {
      fprintf(stderr, "BUG: must call allocState() before buildGPUbuf()\n");
      return 1;
    }
    // gpubuf already created.
    if (q.writeBuffer(gpubuf.getHandle(), cpubuf)) {
      fprintf(stderr, "writeBuffer(gpubuf) failed while resetting gpubuf\n");
      return 1;
    }

    // Copy fixed and state to GPU.
    if (!gpustate.getHandle()) {
      fprintf(stderr, "BUG: must call allocState() before buildGPUbuf()\n");
      return 1;
    }
    if (q.writeBuffer(gpustate.getHandle(), state)) {
      fprintf(stderr, "writeBuffer(gpustate) failed\n");
      return 1;
    }
    if (gpufixed.getHandle()) {
      // gpufixed and gpustate already created.
      if (q.writeBuffer(gpufixed.getHandle(), fixed)) {
        fprintf(stderr, "writeBuffer(gpufixed) failed\n");
        return 1;
      }
    } else {
      if (gpufixed.createInput(q, fixed)) {
        fprintf(stderr, "gpufixed.createInput failed\n");
        return 1;
      }
      // Set program arguments.
      if (prog.setArg(0, gpufixed) || prog.setArg(1, gpustate) ||
          prog.setArg(2, gpubuf)) {
        fprintf(stderr, "prog.setArg failed\n");
        return 1;
      }
    }
    return 0;
  }

  int start(std::vector<size_t> global_work_size) {
    size_t* local_size = NULL;  // OpenCL can auto-tune local_size.
    if (q.NDRangeKernel(prog, global_work_size.size(), NULL, 
                        global_work_size.data(), local_size)) {
      fprintf(stderr, "NDRangeKernel failed\n");
      return 1;
    }
    if (gpustate.copyTo(q, result, completeEvent)) {
      fprintf(stderr, "gpustate.copyTo or finish failed\n");
      return 1;
    }
    return 0;
  }

  int wait() {
    completeEvent.waitForSignal();
    if (wantValidTime) {
      if (q.finish()) {
        return 1;
      }
      timesValid = true;
    } else {
      timesValid = false;
    }
    return 0;
  }

  float submitTime() {
    cl_ulong submitT, endT;
    if (completeEvent.getSubmitTime(submitT)) {
      fprintf(stderr, "submitTime: getSubmitTime failed\n");
      return 0;
    }
    if (completeEvent.getEndTime(endT)) {
      fprintf(stderr, "submitTime: getEndTime failed\n");
      return 0;
    }
    return float(endT - submitT) * 1e-9;
  }

  float execTime() {
    cl_ulong startT, endT;
    if (completeEvent.getStartTime(startT)) {
      fprintf(stderr, "submitTime: getStartTime failed\n");
      return 0;
    }
    if (completeEvent.getEndTime(endT)) {
      fprintf(stderr, "submitTime: getEndTime failed\n");
      return 0;
    }
    return float(endT - startT) * 1e-9;
  }

  bool validTiming() const { return timesValid; }

  float getWorkRate() {
    return getWorkSincePrev() / submitTime();
  }

 protected:
  long long prev_work_done;
  long long total_work_done;
  unsigned ctimeCount;
  bool timesValid;
  long long global_start_atime;
  long long global_start_ctime;
};

// Test that the GPU kernel produces the same hash as the CPU.
static int testGPUsha1(OpenCLdev& dev, OpenCLprog& prog,
                       const CommitMessage& commit) {
  OpenCLqueue q(dev);
  if (q.open()) {
    fprintf(stderr, "q.open failed\n");
    return 1;
  }
  CPUprep prep(dev, prog, q, commit, commit.atime(), commit.ctime());
  prep.state.resize(1);
  if (prep.allocState(1)) {
    return 1;
  }
  // Defaults for state.at(0) do not need to be modified.
  prep.testOnly = 1;

  Sha1Hash cpusha;
  cpusha.update(commit.header.data(), commit.header.size());
  std::string s = commit.toRawString();
  cpusha.update(s.c_str(), s.size());
  cpusha.flush();

  if (prep.buildGPUbuf()) {
    fprintf(stderr, "test: prep.buildGPUbuf failed\n");
    return 1;
  }
  if (prep.start({ prep.state.size() }) || prep.wait()) {
    fprintf(stderr, "test: prep.start or prep.wait failed\n");
    return 1;
  }
  Sha1Hash shaout;
  memcpy(shaout.result, prep.result.at(0).hash, sizeof(shaout.result));

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

// Use an idealized GPU where 1 worker can do 1024 iterations in 0.2 sec.
// (That's a really slow GPU. The load will be tuned from there.)
//
// This function counteracts buildGPUbuf() by estimating how long the kernel
// will run and only giving it about 0.2 sec of work. If numWorkers is bigger,
// the amount of work per worker can be bigger while still fitting in 0.2 sec.
static size_t getCtimeCountFor(size_t numWorkers, long long atime_work) {
  if (!atime_work) {
    fprintf(stderr, "getCtimeCountFor: atime_work=0 will divide by zero\n");
    exit(1);
  }
  float eachWork = float(atime_work) / numWorkers;
  long long r = (long long)(768.0f / eachWork);
  return (r <= 0) ? 1 : (size_t) r;
}

int findOnGPU(OpenCLdev& dev, OpenCLprog& prog, const CommitMessage& commit,
              long long atime_hint, long long ctime_hint) {
  if (testGPUsha1(dev, prog, commit)) {
    fprintf(stderr, "testGPUsha1 failed\n");
    return 1;
  }

  if (atime_hint < commit.atime()) {
    if (atime_hint) {
      fprintf(stderr, "invalid atime_hint %lld (must be at least %lld)\n",
              atime_hint, commit.atime());
    }
    atime_hint = commit.atime();
  }
  if (ctime_hint < commit.ctime()) {
    if (ctime_hint) {
      fprintf(stderr, "invalid ctime_hint %lld (must be at least %lld)\n",
              ctime_hint, commit.ctime());
    }
    ctime_hint = commit.ctime();
  }
  OpenCLqueue q(dev);
  if (q.open()) {
    fprintf(stderr, "q.open failed\n");
    return 1;
  }

  fprintf(stderr, "orig ctime=%lld\n", commit.ctime());
  auto t0 = Clock::now();

  size_t prep_max = 2;
  std::vector<CPUprep> prep;
  prep.reserve(prep_max);
  std::vector<OpenCLprog> progCopies;
  progCopies.reserve(prep_max - 1);

  for (size_t i = 0; i < prep_max; i++) {
    OpenCLprog* chosenProg = NULL;
    if (i == 0) {
      chosenProg = &prog;
    } else {
      progCopies.emplace_back(prog.code, dev);
      if (progCopies.back().copyFrom(prog, prog.funcName.c_str())) {
        fprintf(stderr, "progCopies.copyFrom() failed\n");
        return 1;
      }
      chosenProg = &progCopies.back();
    }
    prep.emplace_back(dev, *chosenProg, q, commit, atime_hint, ctime_hint);
  }
  size_t prep_i = 0;

  // Set context for the ping-ponging CPUprep instances.
  size_t numWorkers = dev.info.maxCU*dev.info.maxWG/2;
  size_t maxWorkers = dev.info.maxCU*dev.info.maxWG*4;
  size_t ctimeCount = getCtimeCountFor(numWorkers, ctime_hint - atime_hint);
  for (size_t i = 0; i < prep.size(); i++) {
    prep.at(i).setCtimeCount(ctimeCount);
    prep.at(i).state.resize(numWorkers);
    if (prep.at(i).allocState(maxWorkers)) {
      return 1;
    }
  }

  if (prep.at(prep_i).buildGPUbuf()) {
    fprintf(stderr, "first buildGPUbuf failed\n");
    return 1;
  }
  if (prep.at(prep_i).start({ prep.at(prep_i).state.size() })) {
    fprintf(stderr, "first start failed\n");
    return 1;
  }

  long long last_work = 0;
  bool startedWorkSizing = false;
  size_t good = 0;
  while (!good) {
    // Auto-tune the workCount, etc.
    // theP now has profiling info (unless this is the very first loop).
    auto& theP = prep.at(prep_i);
    auto& siblingP = prep.at((prep_i + 1) % prep_max);
    if (theP.validTiming() && siblingP.validTiming()) {
      float work = theP.getWorkRate();
      float prev_work = siblingP.getWorkRate();
      float f = 1.0f;
      if (!startedWorkSizing) {
        // Start walking up the capacity of the GPU with a larger batch.
        f = 2.0f;
      } else if (work > prev_work) {
        // Try a larger batch, and see how the GPU responds.
        f = 2.0f;
      } else {
        // This size did no good. Walk back one step and stop.
        for (size_t i = 0; i < prep.size(); i++) {
          prep.at(i).wantValidTime = 0;  // will set validTiming() false.
        }
        f = 0.5;
      }
      startedWorkSizing = true;

      if (f != 1.0f) {
        numWorkers = (size_t) (numWorkers * f);
        ctimeCount = getCtimeCountFor(numWorkers, ctime_hint - atime_hint);
        if (0 && startedWorkSizing) {
          fprintf(stderr, "w=%9.0f p=%9.0f (%.3f) f=%.1f x%zu for %zu\n",
                  work, prev_work, startedWorkSizing ? work/prev_work : 100,
                  f, numWorkers, (prep_i + 1) % prep_max);
        }
      } else if (0) {
        fprintf(stderr, "w=%9.0f p=%9.0f (%.3f) f=%.1f\n",
                work, prev_work, work/prev_work, f);
      }
    }

    // Update siblingP to do the work coming up after theP.
    siblingP.copyCountersFrom(theP);
    siblingP.markAllCtimeDone();

    // Build the batch of work.
    siblingP.setCtimeCount(ctimeCount);
    siblingP.state.resize(numWorkers);
    if (siblingP.buildGPUbuf()) {
      fprintf(stderr, "siblingP.buildGPUbuf failed\n");
      return 1;
    }

    // Kick off siblingP early, so the GPU stays full
    if (siblingP.start({ siblingP.state.size() })) {
      fprintf(stderr, "siblingP.start failed\n");
      return 1;
    }

    // Wait for GPU to finish theP (this also copies results to the CPU)
    if (theP.wait()) {
      fprintf(stderr, "theP.wait failed\n");
      return 1;
    }

    // Report stats
    auto t1 = Clock::now();
    std::chrono::duration<float> sec_duration = t1 - t0;
    float sec = sec_duration.count();  // seconds elapsed, can be >1 loop

    if (1 || sec > 1e-2) {
      t0 = t1;
      long long total_work = 0;
      for (size_t i = 0; i < prep.size(); i++) {
        total_work += prep.at(i).getWorkCount();
      }
      float r = float(total_work - last_work)/sec * 1e-6;
      if (r > 900) {
        r = 0;
      }
      last_work = total_work;
      fprintf(stderr, "%.1fs %6.3fM/s ct=%lld + %2lld x%zu\n",
              sec, r, theP.getCFirst(0), theP.getCEnd(0) - theP.getCFirst(0),
              theP.state.size());
    }

    for (size_t i = 0; i < theP.state.size(); i++) {
      if (theP.result.at(i).matchLen == MIN_MATCH_LEN) {
        continue;
      }
      uint64_t matchCount = theP.result.at(i).matchCount;

      // Reproduce the results on the CPU. Dump the results.
      CommitMessage noodle(commit);
      noodle.set_atime(theP.getAEnd(i) - matchCount);
      noodle.set_ctime(theP.getCEnd(0) - theP.result.at(i).matchCtimeCount);
      fprintf(stderr, "%zu match=%u bytes  atime=%lld  ctime=%lld\n",
              i, theP.result.at(i).matchLen, noodle.atime(),
              noodle.ctime());

      Sha1Hash shaout;
      Blake2Hash b2h;
      noodle.hash(shaout, b2h);
      char shabuf[1024];
      if (shaout.dump(shabuf, sizeof(shabuf))) {
        fprintf(stderr, "shaout.dump failed in findOnGPU\n");
        return 1;
      }
      fprintf(stderr, "%zu sha1: \e[1;31m%.*s\e[0m%s\n", i,
              theP.result.at(i).matchLen * 2, shabuf,
              &shabuf[theP.result.at(i).matchLen * 2]);
      char b2hbuf[1024];
      if (b2h.dump(b2hbuf, sizeof(b2hbuf))) {
        fprintf(stderr, "b2h.dump failed in findOnGPU\n");
        return 1;
      }
      shabuf[theP.result.at(i).matchLen * 2] = 0;
      char* b2hpos = strstr(b2hbuf, shabuf);
      if (b2hpos) {
        good++;
      }
      int b2hlen = b2hpos ? (b2hpos - b2hbuf) : strlen(b2hbuf);
      int b2hMatchLen = theP.result.at(i).matchLen * 2;
      if (b2hlen + b2hMatchLen > (int)strlen(b2hbuf)) {
        b2hMatchLen = strlen(b2hbuf) - b2hlen;
      }
      fprintf(stderr, "%zu blake2: %.*s\e[1;31m%.*s\e[0m%s\n", i, b2hlen,
              b2hbuf, b2hMatchLen, &b2hbuf[b2hlen],
              &b2hbuf[b2hlen + b2hMatchLen]);
    }

    prep_i = (prep_i + 1) % prep_max;
  }

  if (q.finish()) {
    fprintf(stderr, "q.finish failed\n");
    return 1;
  }
  return 0;
}

}  // namespace git-mine
