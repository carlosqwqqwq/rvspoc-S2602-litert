# Copyright 2026 The LiteRT RVV contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Official RISC-V GNU Toolchain naming. Override RISCV_TOOLCHAIN_PREFIX and
# RISCV_SYSROOT when the compiler is installed somewhere else.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

if(NOT DEFINED RISCV_TOOLCHAIN_PREFIX)
  set(RISCV_TOOLCHAIN_PREFIX riscv64-unknown-linux-gnu)
endif()
if(NOT DEFINED RISCV_SYSROOT)
  set(RISCV_SYSROOT /opt/riscv/sysroot)
endif()

set(CMAKE_C_COMPILER ${RISCV_TOOLCHAIN_PREFIX}-gcc)
set(CMAKE_CXX_COMPILER ${RISCV_TOOLCHAIN_PREFIX}-g++)
set(CMAKE_ASM_COMPILER ${RISCV_TOOLCHAIN_PREFIX}-gcc)
set(CMAKE_AR ${RISCV_TOOLCHAIN_PREFIX}-ar)
set(CMAKE_RANLIB ${RISCV_TOOLCHAIN_PREFIX}-ranlib)
set(CMAKE_STRIP ${RISCV_TOOLCHAIN_PREFIX}-strip)

set(CMAKE_SYSROOT ${RISCV_SYSROOT})
set(CMAKE_FIND_ROOT_PATH ${RISCV_SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

set(CMAKE_C_COMPILER_TARGET riscv64-unknown-linux-gnu)
set(CMAKE_CXX_COMPILER_TARGET riscv64-unknown-linux-gnu)
