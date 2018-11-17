#include "ocl-device.h"

namespace gitmine {

const char* clerrstr(cl_int v) {
  if (v == -1001) {
    return "-1001: try apt-get install nvidia-opencl-dev";
  }
  switch (v) {
#define stringify(d) #d
#define case_to_string(d) case d: return #d
    case_to_string(CL_DEVICE_NOT_FOUND);
    case_to_string(CL_DEVICE_NOT_AVAILABLE);
    case_to_string(CL_COMPILER_NOT_AVAILABLE);
    case_to_string(CL_MEM_OBJECT_ALLOCATION_FAILURE);
    case_to_string(CL_OUT_OF_RESOURCES);
    case_to_string(CL_OUT_OF_HOST_MEMORY);
    case_to_string(CL_PROFILING_INFO_NOT_AVAILABLE);
    case_to_string(CL_MEM_COPY_OVERLAP);
    case_to_string(CL_IMAGE_FORMAT_MISMATCH);
    case_to_string(CL_IMAGE_FORMAT_NOT_SUPPORTED);
    case_to_string(CL_BUILD_PROGRAM_FAILURE);
    case_to_string(CL_MAP_FAILURE);
    case_to_string(CL_MISALIGNED_SUB_BUFFER_OFFSET);
    case_to_string(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
    case_to_string(CL_COMPILE_PROGRAM_FAILURE);
    case_to_string(CL_LINKER_NOT_AVAILABLE);
    case_to_string(CL_LINK_PROGRAM_FAILURE);
    case_to_string(CL_DEVICE_PARTITION_FAILED);
    case_to_string(CL_KERNEL_ARG_INFO_NOT_AVAILABLE);

    case_to_string(CL_INVALID_VALUE);
    case_to_string(CL_INVALID_DEVICE_TYPE);
    case_to_string(CL_INVALID_PLATFORM);
    case_to_string(CL_INVALID_DEVICE);
    case_to_string(CL_INVALID_CONTEXT);
    case_to_string(CL_INVALID_QUEUE_PROPERTIES);
    case_to_string(CL_INVALID_COMMAND_QUEUE);
    case_to_string(CL_INVALID_HOST_PTR);
    case_to_string(CL_INVALID_MEM_OBJECT);
    case_to_string(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
    case_to_string(CL_INVALID_IMAGE_SIZE);
    case_to_string(CL_INVALID_SAMPLER);
    case_to_string(CL_INVALID_BINARY);
    case_to_string(CL_INVALID_BUILD_OPTIONS);
    case_to_string(CL_INVALID_PROGRAM);
    case_to_string(CL_INVALID_PROGRAM_EXECUTABLE);
    case_to_string(CL_INVALID_KERNEL_NAME);
    case_to_string(CL_INVALID_KERNEL_DEFINITION);
    case_to_string(CL_INVALID_KERNEL);
    case_to_string(CL_INVALID_ARG_INDEX);
    case_to_string(CL_INVALID_ARG_VALUE);
    case_to_string(CL_INVALID_ARG_SIZE);
    case_to_string(CL_INVALID_KERNEL_ARGS);
    case_to_string(CL_INVALID_WORK_DIMENSION);
    case_to_string(CL_INVALID_WORK_GROUP_SIZE);
    case_to_string(CL_INVALID_WORK_ITEM_SIZE);
    case_to_string(CL_INVALID_GLOBAL_OFFSET);
    case_to_string(CL_INVALID_EVENT_WAIT_LIST);
    case_to_string(CL_INVALID_EVENT);
    case_to_string(CL_INVALID_OPERATION);
    case_to_string(CL_INVALID_GL_OBJECT);
    case_to_string(CL_INVALID_BUFFER_SIZE);
    case_to_string(CL_INVALID_MIP_LEVEL);
    case_to_string(CL_INVALID_GLOBAL_WORK_SIZE);
    case_to_string(CL_INVALID_PROPERTY);
    case_to_string(CL_INVALID_IMAGE_DESCRIPTOR);
    case_to_string(CL_INVALID_COMPILER_OPTIONS);
    case_to_string(CL_INVALID_LINKER_OPTIONS);
    case_to_string(CL_INVALID_DEVICE_PARTITION_COUNT);
    case_to_string(CL_INVALID_PIPE_SIZE);
    case_to_string(CL_INVALID_DEVICE_QUEUE);

#undef case_to_string
    default: return "(unknown)";
  }
}

int getPlatforms(std::vector<cl_platform_id>& platforms) {
  cl_uint platformIdMax = 0;
  cl_int v = clGetPlatformIDs(0, NULL, &platformIdMax);
  if (v != CL_SUCCESS) {
    fprintf(stderr, "%s failed: %d %s\n", "clGetPlatformIDs", v, clerrstr(v));
    return 1;
  }

  platforms.resize(platformIdMax);
  v = clGetPlatformIDs(platformIdMax, platforms.data(), NULL);
  if (v != CL_SUCCESS) {
    fprintf(stderr, "%s failed: %d %s\n", "clGetPlatformIDs", v, clerrstr(v));
    return 1;
  }
  return 0;
}

int getDeviceIds(cl_platform_id platform, std::vector<cl_device_id>& devs) {
  cl_uint devMax = 0;
  cl_int v = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &devMax);
  if (v != CL_SUCCESS) {
    fprintf(stderr, "%s failed: %d %s\n", "clGetDeviceIDs", v, clerrstr(v));
    return 1;
  }
  devs.resize(devMax);
  v = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devMax, devs.data(), NULL);
  if (v != CL_SUCCESS) {
    fprintf(stderr, "%s failed: %d %s\n", "clGetDeviceIDs", v, clerrstr(v));
    return 1;
  }
  return 0;
}

int getDeviceInfoAsBuffer(cl_device_id devId, cl_device_info field, void** buf,
                          size_t* len) {
  size_t llen = 0;
  cl_int v = clGetDeviceInfo(devId, field, 0, NULL, &llen);
  if (v != CL_SUCCESS) {
    fprintf(stderr, "%s(%d) failed: %d %s\n", "clGetDeviceInfo", (int)field,
            v, clerrstr(v));
    return 1;
  }
  *buf = malloc(llen);
  if (!*buf) {
    fprintf(stderr, "getDeviceInfoAsBuffer(%d): malloc(%zu) failed\n",
            (int)field, llen);
    return 1;
  }
  v = clGetDeviceInfo(devId, field, llen, *buf, NULL);
  if (v != CL_SUCCESS) {
    fprintf(stderr, "%s(%d) failed: %d %s\n", "clGetDeviceInfo", (int)field,
            v, clerrstr(v));
    free(*buf);
    *buf = NULL;
    return 1;
  }
  *len = llen;
  return 0;
}

void OpenCLdev::unloadPlatformCompiler() {
  clUnloadPlatformCompiler(platId);
}

int OpenCLdev::probe() {
  if (getDeviceInfo(devId, CL_DEVICE_AVAILABLE, info.avail)) {
    return 1;
  }
  if (!info.avail) {
    fprintf(stderr, "!CL_DEVICE_AVAILABLE\n");
    return 1;
  }
  if (getDeviceInfo(devId, CL_DEVICE_COMPILER_AVAILABLE, info.avail)) {
    return 1;
  }
  if (!info.avail) {
    fprintf(stderr, "!CL_DEVICE_COMPILER_AVAILABLE\n");
    return 1;
  }
  if (getDeviceInfo(devId, CL_DEVICE_GLOBAL_MEM_SIZE, info.globalMemSize) ||
      getDeviceInfo(devId, CL_DEVICE_LOCAL_MEM_SIZE, info.localMemSize) ||
      getDeviceInfo(devId, CL_DEVICE_MAX_COMPUTE_UNITS, info.maxCU) ||
      getDeviceInfo(devId, CL_DEVICE_MAX_WORK_GROUP_SIZE, info.maxWG) ||
      getDeviceInfo(devId, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, info.maxWI)) {
    return 1;
  }
  if (getDeviceInfo(devId, CL_DEVICE_NAME, info.name) ||
      getDeviceInfo(devId, CL_DEVICE_VENDOR, info.vendor) ||
      getDeviceInfo(devId, CL_DEVICE_VERSION, info.openclver) ||
      getDeviceInfo(devId, CL_DRIVER_VERSION, info.driver)) {
    return 1;
  }

  score = info.globalMemSize / 1048576;
  score *= info.maxCU;
  score *= info.maxWG;
  return 0;
}

void OpenCLdev::oclErrorCb(const char *errMsg,
    /* binary and binary_size are implementation-specific data */
    const void */*binary*/, size_t /*binary_size*/, void *user_data) {
  (void)user_data;
  fprintf(stderr, "oclErrorCb: \"%s\"\n", errMsg);
}

}  // namespace git-mine
