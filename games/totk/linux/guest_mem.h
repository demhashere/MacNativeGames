/* Guest memory regions backed by memfds (linux/guest_mem.c). */
#pragma once
#include <stdint.h>
#include <stddef.h>
/* PROT_NONE reservation of size bytes at exactly `at` (NULL if anything is mapped there); commit with mprotect. */
uint8_t *guest_mem_reserve_fixed(uintptr_t at, uint64_t size, const char *name);
/* read-write memory anywhere (the code modules; MAP_FAILED on failure) */
void *guest_mem_map_anywhere(size_t size, const char *name);
void guest_mem_release(void *at, uint64_t size);
/* make [at, at+size) inaccessible and give its pages back (zero on the next commit) */
int guest_mem_decommit(void *at, uint64_t size);
/* the memfd and file offset behind addr, and how many bytes of the region follow it; 0 if addr is no guest memory */
int guest_mem_lookup(uintptr_t addr, int *fd, uint64_t *off, uint64_t *run);
/* map the pages behind src a second time at dst (MAP_FIXED, shared) */
int guest_mem_alias(uintptr_t dst, uintptr_t src, uint64_t size, int prot);
/* give [dst, dst+size) its own region's pages back, zeroed (undoes guest_mem_alias) */
int guest_mem_unalias(uintptr_t dst, uint64_t size, int prot);
