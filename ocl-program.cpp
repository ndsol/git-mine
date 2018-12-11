#include "ocl-program.h"

namespace gitmine {

int OpenCLmem::create(cl_mem_flags flags, size_t size) {
  if (handle) {
    fprintf(stderr, "validation: OpenCLmem::create called twice\n");
    return 1;
  }
  cl_int v;
  handle = clCreateBuffer(dev.getContext(), flags, size, NULL, &v);
  if (v != CL_SUCCESS) {
    fprintf(stderr, "%s failed: %d %s\n", "clCreateBuffer", v, clerrstr(v));
    return 1;
  }
  return 0;
}

static int printBuildLog(char* log) {
  if (!log) {
    return 0;
  }
  if (strspn(log, "\r\n") != strlen(log)) {
    fprintf(stderr, "%s", log);
    if (strlen(log) && log[strlen(log) - 1] != '\n') {
      fprintf(stderr, "\n");
    }
  }
  free(log);
  return 0;
}

int OpenCLprog::open(const char* mainFuncName, std::string buildargs /*= ""*/) {
  if (prog) {
    fprintf(stderr, "validation: OpenCLprog::open called twice\n");
    return 1;
  }
  funcName = mainFuncName;
  const char* pCode = code;
  cl_int v;
  prog = clCreateProgramWithSource(dev.getContext(), 1, &pCode, NULL, &v);
  if (v != CL_SUCCESS) {
    fprintf(stderr, "%s failed: %d %s\n", "clCreateProgramWithSource", v,
            clerrstr(v));
    return 1;
  }

  v = clBuildProgram(prog, 1, &dev.devId, buildargs.c_str(), NULL, NULL);
  if (v != CL_SUCCESS) {
    fprintf(stderr, "%s failed: %d %s\n", "clBuildProgram", v, clerrstr(v));
    (void)printBuildLog(getBuildLog());
    return 1;
  }
  if (printBuildLog(getBuildLog())) {
    return 1;
  }

  kern = clCreateKernel(prog, mainFuncName, &v);
  if (v != CL_SUCCESS) {
    fprintf(stderr, "%s failed: %d %s\n", "clCreateKernel", v, clerrstr(v));
    return 1;
  }
  return 0;
}

void* OpenCLprog::getProgramBuildInfo(cl_program_build_info field) {
  size_t len = 0;
  cl_int v = clGetProgramBuildInfo(prog, dev.devId, field, 0, NULL, &len);
  if (v != CL_SUCCESS) {
    fprintf(stderr, "%s failed: %d %s\n", "clGetProgramBuildInfo", v,
            clerrstr(v));
    return NULL;
  }
  void* mem = malloc(len);
  if (!mem) {
    fprintf(stderr, "%s malloc failed\n", "clGetProgramBuildInfo");
    return NULL;
  }
  v = clGetProgramBuildInfo(prog, dev.devId, field, len, mem, NULL);
  if (v != CL_SUCCESS) {
    fprintf(stderr, "%s failed: %d %s\n", "clGetProgramBuildInfo", v,
            clerrstr(v));
    free(mem);
    return NULL;
  }
  return mem;
}

}  // namespace git-mine
