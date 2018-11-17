.phony: all clean

TARGET=git-mine
SRCS+=git-mine.cpp
SRCS+=blake2b-ref.c
SRCS+=hashapi.cpp
HDRS+=hashapi.h
HDRS+=blake2.h
HDRS+=blake2-impl.h
FLAGS=-O2 -g -Wall -Wextra
CFLAGS+=$(FLAGS)
CXXFLAGS+=$(FLAGS) -std=c++11
LDFLAGS+=-lssl -lcrypto -lpthread

$(shell mkdir -p .o)

all: $(TARGET)

OBJS=$(foreach OBJ,$(patsubst %.cpp,%.o,$(patsubst %.c,%.o,$(SRCS))),.o/$(OBJ))

clean:
	rm -rf $(TARGET) .o

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(LDFLAGS) $^

OCL=git-mine-ocl
OCL_SRCS+=git-mine-ocl.cpp
OCL_SRCS+=ocl-device.cpp
OCL_SRCS+=ocl-program.cpp
OCL_SRCS+=ocl-sha1.cpp
OCL_SRCS+=blake2b-ref.c
OCL_SRCS+=hashapi.cpp
HDRS+=ocl-device.h
HDRS+=ocl-program.h
HDRS+=ocl-sha1.h

OCL_OBJS=$(foreach OBJ,$(patsubst %.cpp,%.o,$(patsubst %.c,%.o,$(OCL_SRCS))),.o/$(OBJ))

$(OCL): $(OCL_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(LDFLAGS) -lOpenCL $^

define SRC_MACRO
.o/$(patsubst %.cpp,%.o,$(patsubst %.c,%.o,$(1))): $(1) $(HDRS)
	$(CXX) $(CXXFLAGS) -o .o/$(patsubst %.cpp,%.o,$(patsubst %.c,%.o,$(1))) -c $(1)
endef

# use $(sort $()) to remove duplicate sources
$(foreach SRC,$(sort $(SRCS) $(OCL_SRCS)),$(eval $(call SRC_MACRO,$(SRC))))
