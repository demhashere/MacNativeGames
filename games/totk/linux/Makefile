# Breath of the Wild (and the shared TOTK runtime) for Linux arm64: Asahi Linux on Apple silicon, Vulkan (Honeykrisp).
# Run from games/totk:  make -f linux/Makefile VERSION=botw190 [GEN=build/botw190] [OUT=build/linux-botw190]
# VERSION: botw190 (base NSP + 1.9.0 update) or botw160. GEN holds the headers the build generates from your game files
# (runtime_tables.h, romfs_config.h, aoc_config.h; shared_fonts_config.h in build/); GEN=build/linux-dummy compiles
# without game data (link check only).
VERSION ?= botw190
GEN ?= build/$(VERSION)
FONTS_H ?= $(if $(filter build/linux-dummy,$(GEN)),build/linux-dummy,build)
OUT ?= build/linux-$(VERSION)
CC := clang
OPT ?= -O2
# The runtime runs on this machine only: tune for its cores (Apple M1/M2 under Asahi report apple-mN to clang).
CPU ?= -mcpu=native
ifeq ($(VERSION),botw160)
GAMEDEF := -DGAME_BOTW=1 -DGAME_BOTW160=1
else ifeq ($(VERSION),botw190)
GAMEDEF := -DGAME_BOTW=1 -DGAME_BOTW190=1
else
$(error VERSION must be botw160 or botw190)
endif

COMMON := -g $(OPT) $(CPU) -fno-strict-aliasing -D_GNU_SOURCE -DTOTK_LINUX=1 $(GAMEDEF) \
          -include linux/compat/darwin_linux.h -Ilinux/compat -Ithird_party/d2r_gfx/gfx/nvn/include -I$(GEN) -I$(FONTS_H) -Ibuild \
          $(foreach p,sdl3 opus vpx vulkan,$(shell pkg-config --cflags $(p) 2>/dev/null))
WARN := -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -Wno-deprecated-declarations -Wno-missing-field-initializers \
        -Wno-sign-compare -Wno-unused-but-set-variable -Wno-format-truncation -Wno-format
CFLAGS := -std=gnu11 $(COMMON) $(WARN)
GFXFLAGS := -std=gnu11 $(COMMON) $(WARN) -Wno-unused-variable

# runtime/*.c as in games/totk/Makefile's botw targets; the Apple frontends, gamepad code and Metal probes are replaced
# by the files in linux/
RT_C := loader.c game_patches_botw.c aoc_romfs.c settings_service.c world_survey.c game_slot.c services.c romfs.c threads.c \
        hle_stack.c gfx_bridge.c account.c input.c hid_feedback.c motion_fusion.c autoload.c gfx_probe.c memory_probe.c \
        shared_font.c ipc.c audio_renderer.c audio_mixer.c audio_reverb.c audio_effects.c fs_host.c opus_codec.c movie_host.c \
        time_host.c applet_host.c swkbd_host.c
RT_S := gate.S hle_stack.S thread_enter.S
GFX_C := $(filter-out %/totk_sysload.c %/totk_vpool.c,$(wildcard third_party/d2r_gfx/gfx/nvn/*.c)) $(wildcard third_party/d2r_gfx/gfx/tex/*.c)
GFX_S := third_party/d2r_gfx/gfx/nvn/nvn_entry.S third_party/d2r_gfx/gfx/nvn/nvn_thunks.S
LX_C := $(wildcard linux/*.c) $(wildcard linux/vk/*.c)
LX_CXX := linux/vk/vk_vma.cpp
CXX := clang++

OBJ := $(addprefix $(OUT)/rt/,$(RT_C:.c=.o)) $(addprefix $(OUT)/asm/,$(RT_S:.S=.o)) \
       $(patsubst third_party/d2r_gfx/gfx/%.c,$(OUT)/gfx/%.o,$(GFX_C)) $(patsubst third_party/d2r_gfx/gfx/nvn/%.S,$(OUT)/asm/%.o,$(GFX_S)) \
       $(patsubst linux/%.c,$(OUT)/lx/%.o,$(LX_C)) $(patsubst linux/%.cpp,$(OUT)/lx/%.o,$(LX_CXX))
HDR := $(wildcard runtime/*.h) $(wildcard third_party/d2r_gfx/gfx/nvn/*.h) $(wildcard third_party/d2r_gfx/gfx/nvn/include/*.h) \
       $(wildcard third_party/d2r_gfx/gfx/tex/*.h) $(wildcard third_party/d2r_gfx/host/*.h) $(wildcard linux/compat/*.h linux/compat/*/*.h) \
       $(wildcard linux/*.h linux/vk/*.h) $(wildcard $(GEN)/*.h)
LIBS := $(foreach p,sdl3 opus vpx vulkan,$(shell pkg-config --libs $(p) 2>/dev/null)) -lcrypto -lpthread -lm -ldl -lstdc++

EXE := $(OUT)/botw-native
.PHONY: all objects clean
all: $(EXE)
objects: $(OBJ)

$(EXE): $(OBJ)
	$(CC) $(CPU) -g -fuse-ld=lld -rdynamic $(OBJ) $(LIBS) -o $@

$(OUT)/rt/%.o: runtime/%.c $(HDR) linux/Makefile
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -Iruntime -c $< -o $@
$(OUT)/gfx/%.o: third_party/d2r_gfx/gfx/%.c $(HDR) linux/Makefile
	@mkdir -p $(dir $@)
	$(CC) $(GFXFLAGS) -c $< -o $@
$(OUT)/lx/%.o: linux/%.c $(HDR) linux/Makefile
	@mkdir -p $(dir $@)
	$(CC) $(GFXFLAGS) -Iruntime -Ilinux -Ithird_party/d2r_gfx/gfx/nvn -c $< -o $@
$(OUT)/lx/%.o: linux/%.cpp linux/Makefile
	@mkdir -p $(dir $@)
	$(CXX) -std=c++17 -O2 $(CPU) -g -w $(shell pkg-config --cflags vulkan 2>/dev/null) -c $< -o $@
# the built-in shaders (masked clears) as SPIR-V arrays
linux/vk/vk_builtin_spv.h: linux/vk/builtin/clear.vert linux/vk/builtin/clear.frag linux/vk/builtin/fsr_easu.frag linux/vk/builtin/fsr.vert linux/vk/builtin/fsr_rcas.frag linux/vk/builtin/sgsr1.frag linux/vk/builtin/ffx/ffx_a.h linux/vk/builtin/ffx/ffx_fsr1.h linux/vk/builtin/gen.py
	python3 linux/vk/builtin/gen.py $@
# Mach-O assembly converted to ELF (linux/tools/macho_asm_to_elf.py)
$(OUT)/asm/%.S: runtime/%.S linux/tools/macho_asm_to_elf.py
	@mkdir -p $(dir $@)
	python3 linux/tools/macho_asm_to_elf.py $< $@
$(OUT)/asm/%.S: third_party/d2r_gfx/gfx/nvn/%.S linux/tools/macho_asm_to_elf.py
	@mkdir -p $(dir $@)
	python3 linux/tools/macho_asm_to_elf.py $< $@
$(OUT)/asm/%.o: $(OUT)/asm/%.S
	$(CC) $(CPU) -c $< -o $@
.PRECIOUS: $(OUT)/asm/%.S

clean:
	rm -rf $(OUT)

# Offscreen smoke test of the Vulkan backend (no game data): make -f linux/Makefile GEN=build/linux-dummy OUT=build/linux-test smoke
SMOKE_OBJ := $(patsubst linux/%.c,$(OUT)/lx/%.o,$(wildcard linux/vk/*.c)) $(OUT)/lx/vk/vk_vma.o $(OUT)/lx/guest_mem.o \
             $(OUT)/gfx/nvn/nvn_dirty.o $(OUT)/gfx/nvn/nvn_tex.o $(OUT)/gfx/tex/nvn_deswizzle.o
.PHONY: smoke
smoke: $(OUT)/vk-smoke
	$(OUT)/vk-smoke
	@if TOTK_SMOKE_NEG=1 $(OUT)/vk-smoke >/dev/null; then echo "negative control passed unexpectedly"; exit 1; else echo "negative control failed as expected"; fi
$(OUT)/vk-smoke: $(OUT)/lx/tests/vk_smoke.o $(SMOKE_OBJ)
	$(CC) $(CPU) -g -fuse-ld=lld $^ $(shell pkg-config --libs vulkan) -lcrypto -lpthread -lm -lstdc++ -o $@
# Presentation times and the vblank clock (vk_present.c) under a headless Weston, which shows nothing
.PHONY: pttest
pttest: $(OUT)/pt-test
	@rm -f "$$XDG_RUNTIME_DIR/wl-pt-test" "$$XDG_RUNTIME_DIR/wl-pt-test.lock"
	@weston --backend=headless --renderer=gl --socket=wl-pt-test --width=640 --height=360 --idle-time=0 >$(OUT)/pt-weston.log 2>&1 & w=$$!; sleep 1; \
	 WAYLAND_DISPLAY=wl-pt-test SDL_VIDEO_DRIVER=wayland $(OUT)/pt-test; r=$$?; kill $$w; exit $$r
$(OUT)/pt-test: $(OUT)/lx/tests/pt_test.o $(SMOKE_OBJ)
	$(CC) $(CPU) -g -fuse-ld=lld $^ $(shell pkg-config --libs vulkan sdl3) -lcrypto -lpthread -lm -lstdc++ -o $@
# The frame hold's decisions (third_party/d2r_gfx/gfx/nvn/nvn_fhold.h), with a negative control
.PHONY: fhtest
fhtest: $(OUT)/fhold-test
	$(OUT)/fhold-test
	@if TOTK_FH_NEG=1 $(OUT)/fhold-test >/dev/null; then echo "negative control passed unexpectedly"; exit 1; else echo "negative control failed as expected"; fi
$(OUT)/fhold-test: linux/tests/fhold_test.c third_party/d2r_gfx/gfx/nvn/nvn_fhold.h
	@mkdir -p $(dir $@)
	$(CC) -std=gnu11 -O2 -g $(CPU) -Ithird_party/d2r_gfx/gfx/nvn $< -lm -o $@
# The job-queue wait cave (runtime/jq_wait.h) run natively, with its negative control
.PHONY: jqtest
jqtest: $(OUT)/jq-wait-test
	$(OUT)/jq-wait-test
	@if TOTK_JQ_NEG=1 $(OUT)/jq-wait-test >/dev/null; then echo "negative control passed unexpectedly"; exit 1; else echo "negative control failed as expected"; fi
$(OUT)/jq-wait-test: linux/tests/jq_wait_test.c runtime/jq_wait.h
	@mkdir -p $(dir $@)
	$(CC) -std=gnu11 -O2 -g $(CPU) -Iruntime $< -lpthread -o $@
