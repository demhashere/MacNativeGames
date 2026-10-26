#!/bin/sh
# Boot check, offscreen (usage: boot-test.sh <empty result folder>; PASSDUMP_AT=<second> also dumps a frame's passes): the installed launcher with SDL's offscreen video driver (a Vulkan headless surface, so the
# swapchain path runs), no sound, a throwaway profile, frame captures every 5 s; A pressed every 6 s from 40 s, then close.
R=$1; mkdir -p "$R/profile" "$R/slot" "$R/shots"
export SDL_VIDEODRIVER=offscreen SDL_AUDIO_DRIVER=dummy TOTK_SLOT_OWNER=claude TOTK_SLOT_PURPOSE=first-boot-test
export TOTK_PROFILE_DIR="$R/profile" TOTK_GAME_SLOT_DIR="$R/slot" TOTK_INPUT_FILE="$R/input.txt"
[ "${CAPTURE:-1}" = 1 ] && export TOTK_VK_CAPTURE="$R/shots" TOTK_VK_CAPTURE_EVERY=5
[ -n "$PASSDUMP_AT" ] && mkdir -p "$R/passes" && export TOTK_VK_PASSDUMP="$R/passes"
$WRAP "$HOME/Applications/botw/botw" </dev/null >/dev/null 2>&1 &
pid=$!; t=0
while kill -0 $pid 2>/dev/null && [ $t -lt 230 ]; do
    sleep 1; t=$((t + 1))
    [ $t -ge 40 ] && [ $t -le 170 ] && [ $((t % 6)) -eq 4 ] && echo "key 38 120" >> "$R/input.txt"
    [ -n "$PASSDUMP_AT" ] && [ $t -eq "$PASSDUMP_AT" ] && echo "passdump 1" >> "$R/input.txt"
    [ $t -eq 200 ] && echo "close" >> "$R/input.txt"
done
if kill -0 $pid 2>/dev/null; then echo "still running at ${t}s: killed"; pkill -f 'botw-native rewritten' ; sleep 2; fi
wait $pid 2>/dev/null; echo "exit $? after ${t}s"
