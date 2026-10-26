#!/bin/sh
# Breath of the Wild @DISPLAY@ (MacNativeGames on Linux, Vulkan): the folders the game uses, then the game itself, as
# app/launcher.m does on the Mac. Saves, the shader cache and the log live in ~/.local/share/MacNativeGames/BotW.
# Any TOTK_* variable already set is kept (e.g. TOTK_PROFILE_DIR=/tmp/save-copy botw runs on a copy of the saves).
here=$(dirname "$(readlink -f "$0")")
data=${XDG_DATA_HOME:-$HOME/.local/share}/MacNativeGames/BotW
export TOTK_PROFILE_DIR="${TOTK_PROFILE_DIR:-$data/profile}"
export TOTK_GAME_SLOT_DIR="${TOTK_GAME_SLOT_DIR:-$data/game-slot}"
export TOTK_SHADER_CACHE="${TOTK_SHADER_CACHE:-$data/shader-cache/rt-totk-native-v7-unverified}"
export TOTK_CONTROLS="${TOTK_CONTROLS:-$data/controls.txt}"
export TOTK_INPUT_FILE="${TOTK_INPUT_FILE:-$data/input.txt}"
export TOTK_ROMFS_MODS="${TOTK_ROMFS_MODS:-$data/mods/@VERSION@}"
export TOTK_ROMFS_PROOF_DIR="$data"
export TOTK_SHARED_FONTS=extracted/firmware/shared-fonts
export TOTK_GPU_ALLOW_SUBMIT=1
# the picture fills the window (key 4 cycles 16:9 -> fit to the window -> 21:9); TOTK_ASPECT=0 starts at 16:9
export TOTK_ASPECT="${TOTK_ASPECT-display}"
# the GPU renders frame N while the game prepares N + 1 (TOTK_SYNC_LAG=0: the game waits for each frame as it asks)
export TOTK_SYNC_LAG="${TOTK_SYNC_LAG-1}"
# a native Wayland window (SDL 3 picks X11 through xwayland-satellite on niri, a copy more per frame); X11 if none
[ -n "$SDL_VIDEODRIVER$SDL_VIDEO_DRIVER" ] || export SDL_VIDEO_DRIVER=wayland,x11
export TOTK_SLOT_OWNER="${TOTK_SLOT_OWNER:-user}" TOTK_SLOT_PURPOSE="${TOTK_SLOT_PURPOSE:-player}"
unset TOTK_AUTOLOAD
# no core dump on a crash (the game logs its call stack; dumping its memory took minutes and gigabytes)
[ "$TOTK_CORE_DUMP" = 1 ] || ulimit -c 0
mkdir -p "$TOTK_PROFILE_DIR" "$TOTK_GAME_SLOT_DIR" "$TOTK_SHADER_CACHE" "$data/logs" || exit 1
# mouse / trackpad look on at the first start (key 7 turns it off; the choice is kept in mouse.txt)
[ -f "$data/mouse.txt" ] || echo on > "$data/mouse.txt"
: > "$TOTK_INPUT_FILE"
log="$data/logs/botw.log"
[ -f "$log" ] && mv -f "$log" "$data/logs/botw.previous.log"
cd "$here/game" || { echo "botw: $here/game is missing (run linux/build_linux.py again)" >&2; exit 1; }
# a terminal sees the log as it is written; started from a menu, it only goes to the file
if [ -t 1 ]; then
    "$here/bin/botw-native" rewritten/@VERSION@ boot 2>&1 | tee "$log"
else
    exec "$here/bin/botw-native" rewritten/@VERSION@ boot >"$log" 2>&1
fi
