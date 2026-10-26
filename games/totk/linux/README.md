# Breath of the Wild on Linux (Asahi Linux, Apple silicon)

The same native port as the Mac app (the game's ARM64 code runs directly; services, audio, input and the NVN graphics API
are implemented on the host), with the Metal pieces replaced:

| Mac | Linux |
| --- | --- |
| `runtime/frontend.m`, `gamepad.m` (AppKit, GameController) | `frontend_sdl.c`, `gamepad_sdl.c` (SDL3) |
| AudioToolbox output | SDL3 audio stream (`runtime/audio_mixer.c`) |
| `nvn_mtl.m` (NVN on Metal 4) | `vk/` (NVN on Vulkan 1.4: shader objects, dynamic rendering) |
| Mach VM, `TPIDRRO_EL0`, Mach-O sections | `guest_mem.c` (memfd), `platform.c` (TLS slots), `compat/` |
| translator `prebuilt/macos-arm64/totk_xlate` | the same sources published for linux-arm64 (NativeAOT) |

Requirements: Asahi Linux (16K pages, Honeykrisp / Mesa 26+), clang, lld, make, git, rsync, Python 3.14, and
`sudo dnf install SDL3-devel vulkan-loader-devel opus-devel libvpx-devel VulkanMemoryAllocator-devel`.
`/dev/udmabuf` must be accessible to your user (it is by default on Fedora Asahi Remix).

## What you need from your Switch

- the base game **Breath of the Wild** (title 01007EF00011E000) as an NSP or NSZ (or a 1.6.0 card image as XCI),
- the **1.9.0 update** NSP or NSZ (title 01007EF00011E800, version 1114112) — or the 1.6.0 update for a 1.6.0 build,
- optional: the two **DLC packs** as NSPs (01007EF00011F001, 01007EF00011F002; 1.9.0 builds only),
- your **prod.keys**, and either your **firmware** (the folder of NCAs, for the system shared fonts) or the six
  `.bfttf` shared fonts from an earlier build (`--shared-fonts`).

Every file is checked against pinned hashes (`games/botw/assets/expected-v1.9.0.json`); another dump of the same
version passes, a damaged one or another version stops the build at the first wrong file.

## Build and install

    python3 games/totk/linux/build_linux.py ~/Games/BotW/base.nsp -k ~/keys/prod.keys \
        --update ~/Games/BotW/update-1.9.0.nsp [--dlc ~/Games/BotW/dlc1.nsp --dlc ~/Games/BotW/dlc2.nsp] \
        --firmware ~/Games/firmware

It needs about 22 GB in the work folder (`BotW_build` next to the game file; `--work` to move it; 36 GB for an NSZ base
game, which NSZ's decompressor expands there first, checking every rebuilt NCA against its hash) and installs into
`~/Applications/botw` (`--out`): `botw` (the launcher), `bin/` (the game, the shader translator, SPIRV-Cross) and
`game/` (the rewritten code, the RomFS image, DLC, fonts; reflinked on btrfs, so no second copy). Saves, the shader cache
and the log go to `~/.local/share/MacNativeGames/BotW`.

The build also translates every shader of the game ahead of time (`build/shader-seed/<version>` in the work folder: the
RomFS shaders plus the variants in `games/botw/assets/shader_list.tsv`, with their SPIR-V) and copies the missing ones
into the shader cache, so new areas do not stutter on first visit. A second build reuses the seed.

## Picture shape and mouse

The launcher starts with the picture filling the window (`TOTK_ASPECT=display`): the camera sees a view of the window's
shape, the HUD and menus keep their shape in the middle 16:9. Key **4** cycles 16:9 -> fit to the window -> 21:9
(`TOTK_ASPECT=<w>:<h>` takes 21:9's place and starts on it, `TOTK_ASPECT=0` starts at 16:9; `TOTK_STRETCH=1` stretches
the 16:9 picture instead). Mouse / trackpad look is on from the first start: moving it turns the camera (the right stick,
as the Mac's mouse mode), left click attacks, right click draws the bow; key **7** turns it off and on (kept in
`mouse.txt` next to `controls.txt`), and the window lets the pointer go when it loses focus. `TOTK_MOUSE_SENSITIVITY`
1-100 (50), `TOTK_MOUSE_INVERT_Y=1`. F11: full screen.

The game sets its camera back to 16:9 every frame in play; a patch in its projection-matrix update
(runtime/game_patches_botw.c, `uw_cave`) gives the main camera the wider ratio there, so no frame comes out 16:9
stretched (the flash every 15-30 s). Screens with no scene under the UI (loading) get black sides.
`TOTK_ASPECT_LOG=1` logs each reset. The launcher opens a native Wayland window (`SDL_VIDEO_DRIVER=wayland,x11`):
SDL 3 otherwise picks X11, which on niri goes through xwayland-satellite (about 45 fps instead of 55-58).
The window opens maximized (`TOTK_WINDOW_MAXIMIZED=0`: at 1920x1080 scaled to the screen, which niri tiles as a column).

## Frame rate, frame pacing and resolution

- **Frame hold** (default on; `TOTK_FRAME_HOLD=0` off, `TOTK_FRAME_HOLD=strict`; **F7**, or previous track without Fn,
  cycles on -> strict -> off while playing). Each frame stays on screen for whole refreshes, paced to the display's
  vblanks (the compositor's presentation times, `VK_EXT_present_timing`), and the game's time step is then exactly that
  long, so motion is even: 60 fps when a frame fits, as fast as it can between about 44 and 60, and an even 30 (or 20)
  when three frames in a row are slower than that, trying 60 again after a second of frames that would fit (longer after
  each failed try). Strict holds only 60 or an even 30. Off is the uncapped mode as before (up to 120 frames a second,
  the display showing the newest). The log has a `frame hold` line every 30 s (frames held 1 / 2 / 3 refreshes, how many
  on time, steps) and one at each change. After nxnative's frame step (`dm/gfx/nvn/nvn_fstep.h`).
- **Render scale** (`TOTK_RENDER_SCALE=0.5`-`1`, or **Fn+F6** cycling the game's own -> 1 -> 0.85 -> 0.75 -> 0.67 -> 0.5):
  fixes the scene's resolution at that fraction of 1920x1080 and turns the game's dynamic resolution off; 1 holds full
  resolution at all times. The HUD, the shadow maps and the 1920x1080 output keep their size. Unset: the game's own,
  as before. From nxnative's `DM_RENDER_SCALE` (a patch where the game applies its dynamic-resolution scale).
- **Job-queue wait** (`TOTK_JQ_WAIT=0`: as shipped): a worker thread waiting for the main thread's item of a job no longer
  takes and releases the queue's lock at full speed, which kept the main thread from it for 30-210 ms at a time
  (nxnative measured those hitches every 1-10 s on an M2 Max); it watches the queue without the lock and sleeps after
  2 ms. Every 30 s the log has a `job-queue waits` line (how many, how long in all, the longest).
A resize no longer waits for the device to go idle: on Wayland that hung the present thread when niri resized the
window, because the compositor keeps its last buffer until a new frame comes (vk_present.c retire_old).

## Tests without game data

    make -f linux/Makefile GEN=build/linux-dummy OUT=build/linux-test smoke

replays hand-written NVN command records through the Vulkan backend (clears, uploads from block-linear guest memory,
dirty-page re-uploads, copies, inline uniform updates, texel reads, two draws with hand-assembled Maxwell shaders
through the translator, the six layers of an array level drawn as one layered pass, a draw sampling what an earlier
draw of its pass drew, lazy clears with their hazards, and the F9 frame ring), with a negative control. `TOTK_VK_VALIDATION=1` runs it under the Khronos validation layer;
it needs `totk_xlate` and `spirv-cross` next to `vk-smoke`. Also with `GEN=build/linux-dummy OUT=build/linux-test`:

- `jqtest`: the job-queue wait's code run natively (its branches out lead to stubs), with a negative control;
- `fhtest`: the frame hold's decisions on made-up frame costs, with a negative control;
- `pttest`: presentation times and the vblank clock with a real compositor that shows nothing (it starts a headless
  Weston): frames paced one and two refreshes apart must reach the screen that far apart.

## Environment switches of the Vulkan backend

- `TOTK_VK_VALIDATION=1`: the Khronos validation layer (slow; for debugging).
- `TOTK_VK_HOST_COPY=0`: uploads through staging copies instead of host image copies.
- `TOTK_PRESENT_MODE=fifo`: FIFO instead of mailbox presentation.
- `TOTK_SHADER_CACHE=<dir>`: the shader cache (the launcher sets it).
- `TOTK_SYNC_LAG=0` (launcher default 1): the game's once-a-frame wait for its newest submission waits for it again
  instead of the previous frame's, so CPU and GPU take turns (~39 instead of ~53 fps outdoors on an M2).
- `TOTK_VK_WB_WAIT=1`: the game's invalidate of its small read-back renderings waits for them again, instead of their
  bytes being written into memory when their submission completes.
- `TOTK_VK_ASYNC_SUBMIT=0`: queue submissions on the replay thread instead of a submit thread.
- `TOTK_VK_CHUNK_DRAWS=<n>` (400), `TOTK_VK_CHUNK_FIRST=<n>` (150): a submission is cut into GPU segments after this many
  draws (the first one earlier), so the GPU starts while the game is still recording.
- `TOTK_VK_DEPTH_LAZY=0`: attaches depth to every pass instead of only to passes that draw with it.
- `TOTK_VK_SPLIT_ALL=1`: ends a pass whenever one of its attachments is sampled (by default only when the pass cleared
  it or drew it with a draw that samples nothing, nvn_mtl.m att_unstored_read); `TOTK_ATT_STORE=0`: only after a clear
  (the rule before; such draws then read what the pass loaded, not what was drawn).
- `TOTK_LAYER_MERGE=0`: draws every layer of a cube / array level in a pass of its own instead of one layered pass for
  the run of passes that render the layers of one level (each draw writes its layer through its vertex shader's layer
  variant; nvn_mtl.m's layered passes).
- `TOTK_HK_BATCH=0`: Honeykrisp's default of one kernel submission per render pass (the backend sets
  `HK_PERFTEST=batch`, up to 64 a submission).
- `TOTK_VK_CUSTOM_BORDER=1`: exact custom sampler border colours. Honeykrisp then emulates them in every sample of every
  shader object (23% more fragment shader instructions on Breath of the Wild's shaders); by default the backend sets
  `HK_PERFTEST=noborder` and such samplers clamp to the edge texel (the grass interaction map's 0.5 grey border).
- `TOTK_VK_LAZY_CLEAR=0`: clears with no draw after them open a pass of their own when the game sets other targets, as
  before. By default they wait for the next pass that attaches the same texture level and layer, which clears it as its
  load action (the four clear-only passes before the G-buffer pass); the rest land in passes of their own before
  anything could see them (that pass, a draw sampling one, a copy, texture clear, dispatch or upload touching one, an
  event, the end of the submission). The passes-per-frame line counts both.
- `TOTK_VK_PRESENT_TIMING=0`: no presentation times (the frame hold then paces by its own clock).
- `TOTK_VK_UBOP_COPY=0`: compute shaders read their uniform buffers from memory while they run, instead of a copy made
  at the submit call (with `TOTK_SYNC_LAG` they then read what the game already wrote for the next frame: shadows jumped
  for a frame every few seconds).
- `TOTK_CORE_DUMP=1`: a core dump when the game crashes (off by default: dumping its memory took minutes and gigabytes;
  the log has the call stack).
- `TOTK_POOL_TLS=0`: every guest address lookup takes the pool lock (by default each thread keeps its last pools).
- `TOTK_GPU_TIME_EVERY=<n>` (4): the GPU time of every n-th frame is measured for the timing lines (each measurement
  adds compute commands Honeykrisp serialises with the render passes); 1: every frame.
- `TOTK_FRAME_RING=0`: no frame ring. By default every presented frame is also kept at 480x270 for the last 96
  frames; F9 (or next track) writes them to `~/.local/share/MacNativeGames/BotW/logs/frames-<date>/` (`<n>-<ms>-f<game
  frame>.ppm`) with `passes.txt`, the passes and dispatches of the last 16 game frames, for showing a flicker that lasts
  a frame or two and what that frame drew differently (`TOTK_FRAME_RING_DIR=<dir>`: elsewhere).
- `TOTK_VK_PASSTIME_AT=<s>` (default 120; -1: never), and F8 in the window (or play / pause, the key's media function on a
  Mac keyboard): GPU time of every pass of one frame, the 25 longest and then all of them in order, with what ended each
  (`[passtime]`). `TOTK_VK_PASSDUMP=1`: a list of the passes of one frame.
- `TOTK_VK_CAPTURE=<dir>`, `TOTK_VK_CAPTURE_EVERY=<s>`: writes the presented frame every s seconds as PPM (for offscreen
  tests, see `tools/boot-test.sh`); `TOTK_VK_CAPTURE_BURST=<s>` also 24 frames in a row from second s (flicker checks).
- `TOTK_VK_WB_LOG=1`, `TOTK_VK_TRACE_FS=<file>`: logs of texture write-backs and of shader inputs (debugging).

Every 600 frames the log has a line with the passes per frame, what ended them, the extra GPU segments and the waits for
read-backs, and one with the commonest pass shapes (size, attachments, NVN target, one layer of a cube / array level,
draws) and the layered passes made so far.
