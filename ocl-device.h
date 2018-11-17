#pragma once

#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>

#ifdef __APPLE__
#include "OpenCL/opencl.h"
#else
#include "CL/cl.h"
#endif

namespace gitmine {

const char* clerrstr(cl_int v);

int getPlatforms(std::vector<cl_platform_id>& platforms);

int getDeviceIds(cl_platform_id platform, std::vector<cl_device_id>& devs);

int getDeviceInfoAsBuffer(cl_device_id devId, cl_device_info field, void** buf,
                          size_t* len);

template<typename T>
int getDeviceInfo(cl_device_id devId, cl_device_info field, T& out) {
  void* r;
  size_t len;
  if (getDeviceInfoAsBuffer(devId, field, &r, &len)) return 1;
  if (len != sizeof(out)) {
    fprintf(stderr, "getDeviceInfo(%d): size %zu, want %zu\n", (int)field,
            len, sizeof(out));
    return 1;
  }
  memcpy(&out, r, sizeof(out));
  free(r);
  return 0;
}

template<>
inline int getDeviceInfo(cl_device_id devId, cl_device_info field, std::string& out) {
  void* r;
  size_t len;
  if (getDeviceInfoAsBuffer(devId, field, &r, &len)) return 1;
  out.assign(reinterpret_cast<const char*>(r), len);
  free(r);
  return 0;
}


class OpenCLdev {
public:
  OpenCLdev(cl_platform_id platId, cl_device_id devId)
      : platId(platId), devId(devId), ctx(NULL) {}

  virtual ~OpenCLdev() {
    if (ctx) {
      clReleaseContext(ctx);
      ctx = NULL;
    }
  }

  const cl_platform_id platId;
  const cl_device_id devId;

  void dump() {
    fprintf(stderr, "  %s: %6.1fGB / %lluKB. CU=%u WG=%zu (v%s)\n",
            info.name.c_str(),
            ((float) info.globalMemSize / 1048576.0f) / 1024.0f,
            (unsigned long long) info.localMemSize / 1024,
            info.maxCU, info.maxWG, info.driver.c_str());
    if (0) fprintf(stderr, "  vendor=%s OpenCL=\"%s\"\n",
                   info.vendor.c_str(), info.openclver.c_str());
  }

  void unloadPlatformCompiler();

  // probe populates score and DevInfo info.
  int probe();

  float score;

  struct DevInfo {
    cl_bool avail;
    cl_ulong globalMemSize;
    cl_ulong localMemSize;
    cl_uint maxCU;
    size_t maxWG;
    cl_uint maxWI;
    std::string name;
    std::string vendor;
    std::string openclver;
    std::string driver;
  } info;

  // openCtx is a wrapper around clCreateContext.
  int openCtx(const cl_context_properties* props) {
    cl_int v = CL_SUCCESS;
    ctx = clCreateContext(props, 1 /*numDevs*/, &devId, OpenCLdev::oclErrorCb,
                          this /*user_data*/, &v);
    if (v != CL_SUCCESS) {
      fprintf(stderr, "%s failed: %d %s\n", "clCreateContext", v, clerrstr(v));
      return 1;
    }
    return 0;
  }

  cl_context getContext() const { return ctx; }

private:
  cl_context ctx;

  static void oclErrorCb(const char *errMsg,
      /* binary and binary_size are implementation-specific data */
      const void */*binary*/, size_t /*binary_size*/, void *user_data);
};

}  // namespace git-mine
