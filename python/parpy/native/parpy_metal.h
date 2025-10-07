#pragma once

#include "Metal/Metal.hpp"
#include <cstdint>
#include <string>
#include <vector>

#define parpy_metal_check_error(e) if (e != 0) return 1;

struct metal_buffer {
  MTL::Buffer *buf;
  int64_t offset = 0;
};

// Functions used by the ParPy library when initializing, synchronizing with
// running GPU code, and operating on buffers.
extern "C" void parpy_init(int64_t);
extern "C" int32_t parpy_sync();
extern "C" metal_buffer *parpy_alloc_buffer(int64_t);
extern "C" void *parpy_ptr_buffer(metal_buffer*);
extern "C" void parpy_buffer_set_offset(metal_buffer*, int64_t);
extern "C" int32_t parpy_memcpy(void*, void*, int64_t, int64_t);
extern "C" int32_t parpy_memset(void*, int64_t, int8_t);
extern "C" int32_t parpy_free_buffer(metal_buffer*);
extern "C" const char *parpy_get_error_message();

// The below functions are to be used in the generated kernel code from C++. We
// wrap these in a namespace to avoid risk of name conflicts.
namespace parpy_metal {
  const char *error_message = nullptr;

  MTL::Library *load_library(const char*);
  MTL::Function *get_fun(MTL::Library*, const char*);
  int32_t alloc(metal_buffer**, int64_t);
  void free(metal_buffer*);
  void copy(void*, void*, int64_t, int64_t);
  int32_t launch_kernel(
      MTL::Function*, std::vector<metal_buffer*>, int64_t, int64_t, int64_t,
      int64_t, int64_t, int64_t);
  void submit_work();
  void sync();
}
