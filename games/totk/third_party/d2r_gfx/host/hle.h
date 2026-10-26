#pragma once
#ifndef MNG_DATA_SECTION
#ifdef __APPLE__
#define MNG_DATA_SECTION(name) "__DATA," #name
#else
#define MNG_DATA_SECTION(name) #name   /* ELF: found through the linker's __start_ / __stop_ symbols */
#endif
#endif
#include "dm.h"
typedef struct {const char *name;void *fn;const char *why;} hle_force_ent;
#define HLE_FORCE(mangled,reason,ret,cname,...) static ret cname(__VA_ARGS__); __attribute__((used,section(MNG_DATA_SECTION(totkgfx)))) static const hle_force_ent ent_##cname={mangled,(void *)cname,reason}; static ret cname(__VA_ARGS__)
