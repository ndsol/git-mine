#include "hashapi.h"
#include "ocl-device.h"
#include "ocl-program.h"
#include "ocl-sha1.h"

namespace gitmine {

int testOpenCL2(OpenCLdev& dev, OpenCLprog& p) {
  size_t count = 1024;
  std::vector<int> numbers(count);
  int factor = 2;

  OpenCLmem inbuf(dev);
  if (inbuf.create(CL_MEM_READ_ONLY, sizeof(numbers[0]) * numbers.size()) ||
      p.setArg(0, inbuf)) {
    return 1;
  }
  OpenCLmem outbuf(dev);
  if (outbuf.create(CL_MEM_WRITE_ONLY, sizeof(numbers[0]) * numbers.size()) ||
      p.setArg(1, outbuf) ||
      p.setArg(2, factor)) {
    return 1;
  }

  OpenCLqueue q(dev);
  if (q.open() || q.writeBuffer(inbuf, numbers)) {
    return 1;
  }
  OpenCLevent completeEvent;
  std::vector<size_t> global_work_size{ numbers.size() };
  if (q.NDRangeKernel(p, global_work_size.size(), NULL, 
                      global_work_size.data(), NULL, &completeEvent.handle)) {
    return 1;
  }
  completeEvent.waitForSignal();

  std::vector<int> donenumbers(numbers.size());
  if (q.readBuffer(outbuf, donenumbers)) {
    return 1;
  }
  fprintf(stderr, "checking:\n");
  for (size_t i = 0; i < numbers.size(); i++) {
    if (numbers.at(i) * factor != donenumbers.at(i)) {
      fprintf(stderr, "after  [%zu]   %d -> %d\n", i, numbers.at(i),
              donenumbers.at(i));
    }
  }
  return q.finish();
}

int testOpenCL(OpenCLdev& dev, const CommitMessage& commit) {
  FILE* f = fopen("/usr/local/google/home/dsp/restore/git-mine/factor.cl", "r");
  if (!f) {
    fprintf(stderr, "Unable to read OpenCL source: %d %s\n", errno,
            strerror(errno));
    return 1;
  }
  size_t codeLen = 16*1024*1024;
  char* codeBuf = (char*)malloc(codeLen);
  if (!codeBuf) {
    fprintf(stderr, "malloc(%zu) failed\n", codeLen);
    return 1;
  }
  size_t rresult = fread(codeBuf,1, codeLen, f);
  if (rresult >= codeLen || rresult == 0) {
    fprintf(stderr, "fread OpenCL source failed\n");
    return 1;
  }
  fclose(f);
  codeBuf[rresult] = 0;
  OpenCLprog p(codeBuf, dev);
  if (p.open("simple_demo")) {
    return 1;
  }
  dev.unloadPlatformCompiler();
  free(codeBuf);
  (void)commit;
  return testOpenCL2(dev, p);
}

int findHash2(OpenCLdev& dev, OpenCLprog& p, const CommitMessage& commit) {
  std::string s = commit.parent + commit.author +
      commit.author_time + commit.author_tz + commit.committer +
      commit.committer_time + commit.committer_tz + commit.log;
  Sha1Hash sha;
  sha.update(commit.header.data(), commit.header.size());
  sha.update(s.c_str(), s.size());
  sha.flush();
  char shabuf[1024];
  if (sha.dump(shabuf, sizeof(shabuf))) {
    return 1;
  }

  std::vector<char> buf(commit.header.data(),
                        commit.header.data() + commit.header.size());
  buf.insert(buf.end(), s.c_str(), s.c_str() + s.size());

  Sha1Hash gpusha;
  if (testGPUsha1(dev, p, buf, gpusha)) {
    fprintf(stderr, "testGPUsha1 failed\n");
    return 1;
  }
  if (memcmp(gpusha.result, sha.result, sizeof(sha.result))) {
    fprintf(stderr, "CPU sha1: %s\n", shabuf);
    if (gpusha.dump(shabuf, sizeof(shabuf))) {
      return 1;
    }
    fprintf(stderr, "GPU sha1: %s - mismatch!\n", shabuf);
    return 1;
  }

  Blake2Hash b2h;
  b2h.update(buf.data(), buf.size());
  b2h.flush();
  if (b2h.dump(shabuf, sizeof(shabuf))) {
    return 1;
  }
  fprintf(stderr, "CPU blake2: %s\n", shabuf);

  if (findOnGPU(dev, p, buf, gpusha)) {
    fprintf(stderr, "findOnGPU failed\n");
    return 1;
  }
  return 0;
}

int findHash(OpenCLdev& dev, const CommitMessage& commit) {
  FILE* f = fopen("/usr/local/google/home/dsp/restore/git-mine/sha1.cl", "r");
  if (!f) {
    fprintf(stderr, "Unable to read OpenCL source: %d %s\n", errno,
            strerror(errno));
    return 1;
  }
  size_t codeLen = 16*1024*1024;
  char* codeBuf = (char*)malloc(codeLen);
  if (!codeBuf) {
    fprintf(stderr, "malloc(%zu) failed\n", codeLen);
    return 1;
  }
  size_t rresult = fread(codeBuf,1, codeLen, f);
  if (rresult >= codeLen || rresult == 0) {
    fprintf(stderr, "fread OpenCL source failed\n");
    return 1;
  }
  fclose(f);
  codeBuf[rresult] = 0;
  OpenCLprog p(codeBuf, dev);
  if (p.open("main")) {
    fprintf(stderr, "open(main) failed\n");
    return 1;
  }
  dev.unloadPlatformCompiler();
  free(codeBuf);
  return findHash2(dev, p, commit);
}

int runOCL(const CommitMessage& commit) {
  std::vector<cl_platform_id> platforms;
  if (getPlatforms(platforms)) {
    return 1;
  }
  if (platforms.size() < 1) {
    fprintf(stderr, "clGetPlatformIDs: no OpenCL hardware found.\n");
    return 1;
  }
  if (platforms.size() > 1) {
    fprintf(stderr, "TODO: add a way to pick which platform\n");
    // Use this code in a for loop:
    //fprintf(stderr, "  platform [%zu] (%p):\n", i, platforms.at(i));
    return 1;
  }
  for (size_t i = 0; i < platforms.size(); i++) {
    std::vector<cl_device_id> devs;
    if (getDeviceIds(platforms.at(i), devs)) {
      return 1;
    }

    size_t best = 0;
    float bestScore = 0.0f;
    for (size_t j = 0; j < devs.size(); j++) {
      OpenCLdev dev(platforms.at(i), devs.at(j));
      if (dev.probe()) {
        return 1;
      }
      if (j == 0 || dev.score > bestScore) {
        best = j;
        bestScore = dev.score;
      }
    }
    OpenCLdev dev(platforms.at(i), devs.at(best));
    if (dev.probe()) {
      return 1;
    }
    fprintf(stderr, "Selected OpenCL:\n");
    dev.dump();

    // ctxProps is a list terminated with a "0, 0" pair.
    const cl_context_properties ctxProps[] = {
      CL_CONTEXT_PLATFORM,
      reinterpret_cast<cl_context_properties>(platforms.at(i)),
      0, 0,
    };
    if (dev.openCtx(ctxProps) || findHash(dev, commit)) {
      return 1;
    }
  }
  return 0;
}

}  // namespace git-mine

int main(int argc, char ** argv) {
  if (argc != 1) {
    // This utility must be called from a post-commit hook
    // with $GIT_TOPLEVEL as the only argument.
    fprintf(stderr, "Usage: %s\n", argv[0]);
    return 1;
  }

  CommitMessage commit;
  {
    FILE* f = stdin;
    CommitReader reader(argv[0]);
    if (reader.read_from(f, &commit)) {
      return 1;
    }
    Sha1Hash sha;
    Blake2Hash b2h;
    if (commit.hash(sha, b2h)) {
      return 1;
    }
    char shabuf[1024];
    if (sha.dump(shabuf, sizeof(shabuf))) {
      return 1;
    }
    fprintf(stderr, "sha1:   %s\n", shabuf);
    char buf[1024];
    if (b2h.dump(buf, sizeof(buf))) {
      return 1;
    }
    fprintf(stderr, "blake2: %s\n", buf);
    static const char* wantsha = "68d1800069d0d0f098d151560a5c62049113da1f";
    if (strcmp(shabuf, wantsha)) {
      fprintf(stderr, "sha1 want: %s\n", wantsha);
      fprintf(stderr, "BUG BUG BUG!\n");
      return 1;
    }
    if (strcmp(buf, "22e4065be020830611561aafe5420209d050f4ebe5de22cb1fe7bc3ddd272f6fe974c826a1a7ee39fe016eaea8c9702c8d50fa303baafa3ca8e6041ef7dc8173")) {
      fprintf(stderr, "blake2 want 22e4065b...\n");
      fprintf(stderr, "BUG BUG BUG!\n");
      return 1;
    }
  }

  return gitmine::runOCL(commit);
}
