#pragma once

#include "ocl-device.h"

namespace gitmine {


class OpenCLprog {
public:
  OpenCLprog(const char* code, OpenCLdev& dev)
      : code(code), dev(dev), prog(NULL), kern(NULL) {}
  virtual ~OpenCLprog() {
    if (kern) {
      clReleaseKernel(kern);
      kern = NULL;
    }
    if (prog) {
      clReleaseProgram(prog);
      prog = NULL;
    }
  }

  int open(const char* mainFuncName);

  const char* const code;
  OpenCLdev& dev;

  char* getBuildLog() {
    return reinterpret_cast<char*>(getProgramBuildInfo(CL_PROGRAM_BUILD_LOG));
  }

  template<typename T>
  int setArg(cl_uint argIndex, const T& arg) {
    if (!kern) {
      fprintf(stderr, "setArg(%u) before open\n", argIndex);
      return 1;
    }
    cl_int v = clSetKernelArg(kern, argIndex, sizeof(arg),
                              reinterpret_cast<const void*>(&arg));
    if (v != CL_SUCCESS) {
      fprintf(stderr, "%s(%u) failed: %d %s\n", "clSetKernelArg", argIndex, v,
              clerrstr(v));
      return 1;
    }
    return 0;
  }

  cl_kernel getKern() const { return kern; }

private:
  void* getProgramBuildInfo(cl_program_build_info field);

  cl_program prog;
  cl_kernel kern;
};

class OpenCLevent {
public:
  OpenCLevent() {}
  virtual ~OpenCLevent() {
    clReleaseEvent(handle);
  }

  void waitForSignal() {
    clWaitForEvents(1, &handle);
  }

  cl_event handle;
};

class OpenCLqueue {
public:
  OpenCLqueue(OpenCLdev& dev) : dev(dev), handle(NULL) {}

  virtual ~OpenCLqueue() {
    if (handle) {
      clReleaseCommandQueue(handle);
      handle = NULL;
    }
  }

  int open(const cl_queue_properties* props = NULL) {
    if (handle) {
      fprintf(stderr, "validation: OpenCLqueue::open called twice\n");
      return 1;
    }
    int v;
    handle = clCreateCommandQueueWithProperties(
        dev.getContext(), dev.devId, props, &v);
    if (v != CL_SUCCESS) {
      fprintf(stderr, "%s failed: %d %s\n", "clCreateCommandQueue", v,
              clerrstr(v));
      return 1;
    }
    return 0;
  }

  // writeBuffer does a blocking write
  template<typename T>
  int writeBuffer(cl_mem hnd, const std::vector<T>& src, size_t offset = 0) {
    size_t srcSize = sizeof(src[0]) * src.size();
    cl_int v = clEnqueueWriteBuffer(handle, hnd, CL_TRUE /*blocking*/,
        offset * srcSize, srcSize, reinterpret_cast<const void*>(src.data()),
        0, NULL, NULL);
    if (v != CL_SUCCESS) {
      fprintf(stderr, "%s failed: %d %s\n", "clEnqueueWriteBuffer", v,
              clerrstr(v));
      return 1;
    }
    return 0;
  }

  // readBuffer does a blocking read
  template<typename T>
  int readBuffer(cl_mem hnd, std::vector<T>& dst, size_t offset = 0) {
    size_t dstSize = sizeof(dst[0]) * dst.size();
    cl_int v = clEnqueueReadBuffer(handle, hnd, CL_TRUE /*blocking*/,
        offset * dstSize, dstSize, reinterpret_cast<void*>(dst.data()),
        0, NULL, NULL);
    if (v != CL_SUCCESS) {
      fprintf(stderr, "%s failed: %d %s\n", "clEnqueueReadBuffer", v,
              clerrstr(v));
      return 1;
    }
    return 0;
  }

  OpenCLdev& dev;

  int NDRangeKernel(OpenCLprog& prog, cl_uint work_dim,
                    const size_t* global_work_offset,
                    const size_t* global_work_size,
                    const size_t* local_work_size,
                    cl_event* completeEvent,
                    const std::vector<cl_event>& waitList
                        = std::vector<cl_event>()) {
    cl_int v = clEnqueueNDRangeKernel(handle, prog.getKern(), work_dim,
                                      global_work_offset,
                                      global_work_size, local_work_size,
                                      waitList.size(), waitList.data(),
                                      completeEvent);
    if (v != CL_SUCCESS) {
      fprintf(stderr, "%s failed: %d %s\n", "clEnqueueNDRangeKernel", v,
              clerrstr(v));
      return 1;
    }
    return 0;
  }

  int finish() {
    cl_int v = clFinish(handle);
    if (v != CL_SUCCESS) {
      fprintf(stderr, "%s failed: %d %s\n", "clFinish", v, clerrstr(v));
      return 1;
    }
    return 0;
  }

private:
  cl_command_queue handle;
};

class OpenCLmem {
public:
  OpenCLmem(OpenCLdev& dev) : dev(dev), handle(NULL) {}
  virtual ~OpenCLmem() {
    if (handle) {
      clReleaseMemObject(handle);
      handle = NULL;
    }
  }
  OpenCLdev& dev;

  int create(cl_mem_flags flags, size_t size);

  template<typename T>
  int createInput(OpenCLqueue& q, std::vector<T>& in, size_t copies = 1) {
    if (create(CL_MEM_READ_ONLY, sizeof(in[0]) * in.size() * copies)) {
      fprintf(stderr, "createInput failed\n");
      return 1;
    }
    if (copies == 1) {
      if (q.writeBuffer(getHandle(), in)) {
        fprintf(stderr, "createInput: writeBuffer failed\n");
        return 1;
      }
    }
    return 0;
  }

  template<typename T>
  int createIO(OpenCLqueue& q, std::vector<T>& in, size_t copies = 1) {
    if (create(CL_MEM_READ_WRITE, sizeof(in[0]) * in.size() * copies)) {
      fprintf(stderr, "createIO failed\n");
      return 1;
    }
    if (copies == 1) {
      if (q.writeBuffer(getHandle(), in)) {
        fprintf(stderr, "createIO: writeBuffer failed\n");
        return 1;
      }
    }
    return 0;
  }

  template<typename T>
  int createOutput(std::vector<T>& out) {
    // The OpenCLqueue::readBuffer() call is done using copyTo, below.
    return create(CL_MEM_WRITE_ONLY, sizeof(out[0]) * out.size());
  }

  template<typename T>
  int copyTo(OpenCLqueue& q, std::vector<T>& out) {
    return q.readBuffer(getHandle(), out);
  }

  cl_mem getHandle() const { return handle; }

private:
  cl_mem handle;
};

template<>
inline int OpenCLprog::setArg(cl_uint argIndex, const OpenCLmem& mem) {
  return setArg(argIndex, mem.getHandle());
}

}  // namespace git-mine
