#!/bin/sh
# the repo's game sources -> the build's work tree (build_app.prepare_work's rsync), botw-native rebuilt and installed
set -e
R=$HOME/src/MacNativeGames/games/totk; T=$HOME/Games/BotW_build/totk
rsync -a --exclude=/extracted --exclude=/rewritten --exclude=/build --exclude=/analysis --exclude=/.venv --exclude=/tooling --exclude=/user-data "$R/" "$T/"
cd "$T"
make -f linux/Makefile VERSION=botw190 GEN=build/botw190 OUT=build/linux-botw190 -j8 > build/rebuild.log 2>&1 || { grep -E 'error|warning' build/rebuild.log | head -20; tail -5 build/rebuild.log; exit 1; }
grep -E 'warning' build/rebuild.log | head -10 || true
install -m 755 build/linux-botw190/botw-native "$HOME/Applications/botw/bin/botw-native"
echo "rebuilt: $(grep -c '^clang' build/rebuild.log) compiles/links"
