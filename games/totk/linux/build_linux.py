#!/usr/bin/env python3
"""Breath of the Wild for Linux on Apple silicon (Asahi Linux, Vulkan / Honeykrisp): your own dumps -> an installed game.

    python3 games/totk/linux/build_linux.py <base.nsp|base.nsz|card.xci> -k <prod.keys> [--update <update.nsp|update.nsz>]
        [--dlc <pack.nsp> ...] (--firmware <firmware NCA folder> | --shared-fonts <folder>) [--work <folder>] [--out <folder>]

The same steps and checks as build_app.py's Breath of the Wild path (its functions are used as they are): the ExeFS out
of the base NSP + 1.9.0 update (or a 1.6.0 card image), the three code modules rewritten and checked against the pinned
hashes, the merged RomFS checked over every IVFC level, the add-on content packs, the shared fonts. Then, in place of the
Mac app: the shader translator for linux-arm64 (vendor/ryujinx + patches + Service.cs, NativeAOT), the pinned SPIRV-Cross,
the generated runtime headers, the game binary (linux/Makefile) and an installed folder (default ~/Applications/botw)
with the launcher `botw` in it. Large files are reflinked (btrfs) where possible, else copied.
The game's shaders are translated ahead (the startup seed of build_app.py, with the SPIR-V Vulkan loads) and imported
into the player's shader cache, so a first visit to a place does not stutter; anything else translates on first use.
A compressed base game or update (NSZ) is first expanded into the work folder by NSZ's decompressor, which checks every
rebuilt NCA against its hash; the base game takes about 14 GB more there.
"""
import argparse, json, os, platform, shutil, subprocess, sys, time
from pathlib import Path

LINUX = Path(__file__).resolve().parent
GAME = LINUX.parent
REPO = GAME.parents[1]
sys.path.insert(0, str(REPO))
import build_app as B   # noqa: E402

PAGE = 16384


def say(msg):
    B.say(msg)


def need_tool(name, hint):
    if not shutil.which(name):
        raise B.BuildError(f'{name} not found: {hint}')


def prerequisites():
    if platform.system() != 'Linux' or platform.machine() != 'aarch64':
        raise B.BuildError('this build is for Linux on arm64 (Asahi Linux on Apple silicon)')
    if os.sysconf('SC_PAGE_SIZE') != PAGE:
        raise B.BuildError(f'expected the 16K-page kernel of Asahi Linux (page size {os.sysconf("SC_PAGE_SIZE")})')
    if sys.version_info < (3, 14):
        raise B.BuildError('needs Python 3.14 or newer (the build reads zstd with the standard library)')
    for tool, hint in (('clang', 'sudo dnf install clang'), ('clang++', 'sudo dnf install clang'), ('ld.lld', 'sudo dnf install lld'),
                       ('make', 'sudo dnf install make'), ('git', 'sudo dnf install git'), ('rsync', 'sudo dnf install rsync'),
                       ('pkg-config', 'sudo dnf install pkgconf-pkg-config')):
        need_tool(tool, hint)
    missing = [pkg for pkg, rpm in (('sdl3', 'SDL3-devel'), ('vulkan', 'vulkan-loader-devel'), ('opus', 'opus-devel'), ('vpx', 'libvpx-devel'))
               if subprocess.run(['pkg-config', '--exists', pkg]).returncode]
    if missing:
        rpms = {'sdl3': 'SDL3-devel', 'vulkan': 'vulkan-loader-devel', 'opus': 'opus-devel', 'vpx': 'libvpx-devel'}
        raise B.BuildError(f'development packages missing: sudo dnf install {" ".join(rpms[m] for m in missing)}')
    if not Path('/usr/include/vk_mem_alloc.h').is_file():
        raise B.BuildError('the Vulkan Memory Allocator header is missing: sudo dnf install VulkanMemoryAllocator-devel')
    if not os.access('/dev/udmabuf', os.R_OK | os.W_OK):
        raise B.BuildError('/dev/udmabuf is not accessible: the GPU maps guest memory through it (check the udmabuf module and its ACL)')
    return Path(sys.executable)


# ---- hactool: build_app.build_hactool's patches, with glibc's fopencookie in place of BSD funopen ----
BASE_HOOK = r'''
typedef struct { FILE *file; off_t start, size, position; } mng_base_range;
static ssize_t mng_base_read(void *cookie, char *buffer, size_t count) {
    mng_base_range *range = cookie;
    off_t left = range->size - range->position;
    if ((off_t)count > left) count = (size_t)left;
    if (!count) return 0;
    if (fseeko(range->file, range->start + range->position, SEEK_SET)) return -1;
    size_t got = fread(buffer, 1, count, range->file);
    if (!got && ferror(range->file)) return -1;
    range->position += (off_t)got;
    return (ssize_t)got;
}
static int mng_base_seek(void *cookie, off64_t *offset, int whence) {
    mng_base_range *range = cookie;
    off_t base = whence == SEEK_SET ? 0 : whence == SEEK_CUR ? range->position : whence == SEEK_END ? range->size : -1;
    if (base < 0 || *offset < -base || *offset > range->size - base) return -1;
    range->position = base + *offset; *offset = range->position;
    return 0;
}
static int mng_base_close(void *cookie) {
    mng_base_range *range = cookie;
    int result = fclose(range->file); free(range); return result;
}
static FILE *mng_open_base(const char *path) {
    const char *start = getenv("MNG_BASE_RANGE_OFFSET"), *size = getenv("MNG_BASE_RANGE_SIZE");
    if (!start && !size) return fopen(path, "rb");
    if (!start || !size) return NULL;
    char *end1, *end2;
    unsigned long long a = strtoull(start, &end1, 10), n = strtoull(size, &end2, 10);
    if (!*start || !*size || *end1 || *end2 || a > INT64_MAX || !n || n > INT64_MAX - a) return NULL;
    mng_base_range *range = calloc(1, sizeof(*range));
    if (!range) return NULL;
    range->file = fopen(path, "rb"); range->start = (off_t)a; range->size = (off_t)n;
    if (!range->file) { free(range); return NULL; }
    if (fseeko(range->file, 0, SEEK_END) || ftello(range->file) < (off_t)(a + n)) { mng_base_close(range); return NULL; }
    cookie_io_functions_t io = {mng_base_read, NULL, mng_base_seek, mng_base_close};
    FILE *view = fopencookie(range, "rb", io);
    if (!view) mng_base_close(range);
    return view;
}
'''


def build_hactool(tree, logs):
    source = tree.parent / 'src/hactool'
    B.git_fetch(*B.HACTOOL, source, logs / 'fetch-hactool.log')
    main = source / 'main.c'
    text = main.read_text()
    old = 'FILE *titlekeyfile = open_key_file("title");'
    new = 'FILE *titlekeyfile = getenv("MNG_TITLE_KEYS") ? fopen(getenv("MNG_TITLE_KEYS"), "rb") : open_key_file("title");'
    if new not in text:
        if text.count(old) != 1:
            raise B.BuildError('pinned hactool title-key hook changed')
        text = text.replace(old, new)
    if '#define _GNU_SOURCE' not in text:
        text = '#define _GNU_SOURCE 1\n#define _FILE_OFFSET_BITS 64\n' + text
    if 'static FILE *mng_open_base(const char *path)' not in text:
        marker = 'int main(int argc, char **argv)'
        if text.count(marker) != 1 or text.count('nca_ctx.tool_ctx->base_file = fopen(optarg, "rb")') != 2:
            raise B.BuildError('pinned hactool bounded base-file hook changed')
        text = text.replace(marker, BASE_HOOK + '\n' + marker)
        text = text.replace('nca_ctx.tool_ctx->base_file = fopen(optarg, "rb")', 'nca_ctx.tool_ctx->base_file = mng_open_base(optarg)')
    main.write_text(text)
    nca = source / 'nca.c'
    text = nca.read_text()
    for old, new, what in (
            ('sb->relocation_header.offset + sb->relocation_header.size != sb->subsection_header.offset ||\n'
             '            sb->subsection_header.offset + sb->subsection_header.size != ctx->size',
             'sb->relocation_header.offset + sb->relocation_header.size != sb->subsection_header.offset ||\n'
             '            sb->subsection_header.offset + sb->subsection_header.size > ctx->size ||\n'
             '            (ctx->size - (sb->subsection_header.offset + sb->subsection_header.size)) % 0x200 != 0', 'BKTR layout check'),
            ('if (ctx->bktr_ctx.subsection_block->total_size != sb->subsection_header.offset) {',
             'if (ctx->bktr_ctx.subsection_block->total_size != sb->subsection_header.offset &&\n'
             '            (sb->relocation_header.size > sb->subsection_header.offset ||\n'
             '             ctx->bktr_ctx.subsection_block->total_size != sb->subsection_header.offset - sb->relocation_header.size)) {', 'BKTR total-size check')):
        if new not in text:
            if text.count(old) != 1:
                raise B.BuildError(f'pinned hactool {what} changed')
            text = text.replace(old, new)
    nca.write_text(text)
    config = source / 'config.mk'
    if not config.exists():
        shutil.copy2(source / 'config.mk.template', config)
    B.run(['make', f'-j{os.cpu_count() or 4}', 'CC=clang'], logs / 'hactool-build.log', cwd=source)
    return source / 'hactool'


def build_spirv_cross(tree, logs):
    """The pinned SPIRV-Cross (build_app.SPIRV_CROSS) compiled with clang++: the reflection is parsed from its MSL."""
    out = tree / 'build/spirv-cross/spirv-cross'
    if out.is_file():
        return out
    src = tree / 'tooling/spirv-cross'
    objs = out.parent / 'obj'
    objs.mkdir(parents=True, exist_ok=True)
    flags = [f for f in B.SPIRV_CROSS_FLAGS if f not in ('-arch', 'arm64')]
    import concurrent.futures
    def one(name):
        B.run(['clang++', *flags, '-c', src / name, '-o', objs / (name + '.o')], logs / 'spirv-cross' / (name + '.log'), quiet=False)
    (logs / 'spirv-cross').mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(os.cpu_count() or 4) as pool:
        list(pool.map(one, B.SPIRV_CROSS_SOURCES))
    B.run(['clang++', *[objs / (n + '.o') for n in B.SPIRV_CROSS_SOURCES], '-o', out], logs / 'spirv-cross' / 'link.log', quiet=False)
    return out


def dotnet(tree, logs):
    exe = tree / 'tooling/dotnet/dotnet'
    for candidate in (exe, GAME / 'tooling/dotnet/dotnet'):
        if candidate.is_file():
            r = subprocess.run([candidate, '--list-sdks'], capture_output=True, text=True)
            if any(line.startswith('10.') for line in r.stdout.splitlines()):
                return candidate
    say('3/4 shaders: fetching the .NET 10 SDK (only to compile the shader translator; the game does not need it)')
    target = tree / 'tooling/dotnet'
    script = tree / 'tooling/dotnet-install.sh'
    script.parent.mkdir(parents=True, exist_ok=True)
    B.run(['curl', '-fsSL', 'https://dot.net/v1/dotnet-install.sh', '-o', script], logs / 'dotnet-fetch.log')
    B.run(['bash', script, '--channel', '10.0', '--install-dir', target, '--no-path'], logs / 'dotnet-install.log')
    return target / 'dotnet'


def build_translator(tree, logs):
    """totk_xlate for linux-arm64 with NativeAOT (publish_native.sh's steps): no .NET runtime at play time."""
    out = tree / 'build/shader-service/totk_xlate'
    sources = [*(tree / 'tools/totk_xlate').glob('*.cs'), tree / 'tools/totk_xlate/totk_xlate.csproj', *(tree / 'tools/totk_xlate/patches').glob('*.patch')]
    stamp = out.with_name('sources.json')
    want = {str(p.relative_to(tree)): B.sha256(p) for p in sorted(sources)}
    try:
        if out.is_file() and json.loads(stamp.read_text()) == want:
            return out
    except (OSError, ValueError):
        pass
    say('3/4 shaders: compiling the shader translator for linux-arm64 (NativeAOT, a few minutes the first time)')
    work = tree / 'build/xlate-work'
    shutil.rmtree(work, ignore_errors=True)
    (work / 'xlate').mkdir(parents=True)
    B.run(['rsync', '-a', f'{REPO}/vendor/ryujinx/', f'{work / "ryujinx-src"}/'], logs / 'xlate-sources.log')
    for p in sorted((tree / 'tools/totk_xlate/patches').glob('*.patch')):
        with p.open('rb') as f:
            r = subprocess.run(['patch', '-s', '-d', work / 'ryujinx-src', '-p1', '--forward'], stdin=f, capture_output=True, text=True)
        if r.returncode:
            raise B.BuildError(f'translator patch {p.name} does not apply: {r.stdout + r.stderr}')
    for p in sources:
        if p.suffix in ('.cs', '.csproj'):
            shutil.copy2(p, work / 'xlate' / p.name)
    env = dict(os.environ, DOTNET_CLI_TELEMETRY_OPTOUT='1', DOTNET_NOLOGO='1')
    B.run([dotnet(tree, logs), 'publish', work / 'xlate/totk_xlate.csproj', '-c', 'Release', '-r', 'linux-arm64', '-p:PublishAot=true',
           '-p:InvariantGlobalization=true', f'-p:RyuSrc={work / "ryujinx-src"}', '-o', work / 'out'], logs / 'xlate-publish.log', env=env)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(work / 'out/totk_xlate', out)
    stamp.write_text(json.dumps(want))
    shutil.rmtree(work, ignore_errors=True)
    return out


def expand_nsz(tree, package, keys, logs, what):
    """An NSZ as the NSP it was made from, cached in the work folder: build_app.py's prepare_update_container, for the
    base game too. NSZ rebuilds every NCA and checks it against its hash, so the pinned checks see an uncompressed dump."""
    if B.input_kind(package) != 'nsz':
        return package
    folder = tree / 'extracted/containers'
    output, manifest = folder / f'{package.stem}.nsp', folder / f'{package.stem}.json'
    if B.source_matches(manifest, package) and output.is_file():
        return output
    folder.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    manifest.unlink(missing_ok=True)
    say(f'decompressing the NSZ {what} and verifying its NCA hashes')
    try:
        B.run([tree / '.venv/bin/nsz', '-D', '--keys', keys, '-o', folder, package], logs / f'decompress-{package.stem}.log', cwd=tree)
    except B.BuildError:
        output.unlink(missing_ok=True)   # a partly written NSP (up to 14 GB for the base game)
        raise
    if not output.is_file():
        raise B.BuildError(f'the NSZ decompressor did not produce an NSP: {output}')
    st = package.stat()
    manifest.write_text(json.dumps(dict(source=str(package.resolve()), source_size=st.st_size,
                                        source_mtime_ns=st.st_mtime_ns), indent=2) + '\n')
    return output


def step_headers(tree, version, logs):
    py = tree / '.venv/bin/python'
    B.run([py, 'scripts/prepare_runtime.py', '--version', version], logs / f'prepare-runtime-{version}.log', cwd=tree)
    B.run([py, 'scripts/prepare_romfs_backend.py', '--version', version], logs / f'prepare-romfs-{version}.log', cwd=tree)
    B.run([py, 'scripts/prepare_romfs_backend.py', '--version', version, '--aoc'], logs / f'prepare-aoc-{version}.log', cwd=tree)
    B.run([py, 'scripts/prepare_shared_fonts.py'], logs / 'prepare-fonts.log', cwd=tree)


def step_binary(tree, version, logs):
    say('4/4 game: compiling the runtime and the Vulkan graphics backend')
    B.run(['make', '-f', 'linux/Makefile', f'VERSION={version}', f'GEN=build/{version}', f'OUT=build/linux-{version}', f'-j{os.cpu_count() or 4}'],
          logs / f'binary-{version}.log', cwd=tree)
    return tree / f'build/linux-{version}/botw-native'


def place(src, dst):
    """dst becomes src's bytes: a reflink (btrfs, instant) when possible, else a copy; an unchanged dst is kept."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file() and dst.stat().st_size == src.stat().st_size and dst.stat().st_mtime_ns == src.stat().st_mtime_ns:
        return
    tmp = dst.with_name(dst.name + '.partial')
    tmp.unlink(missing_ok=True)
    r = subprocess.run(['cp', '--reflink=auto', '--preserve=timestamps', src, tmp], capture_output=True, text=True)
    if r.returncode:
        raise B.BuildError(f'copying {src} failed: {r.stderr.strip()}')
    tmp.replace(dst)


def step_seed(tree, botw, version, translator, spirv_cross, logs):
    """The startup shader seed (build_app.py step_shaders_botw: every shader in the game data, for its default bindings
    and for each binding state the shared shader list names), with each variant's SPIR-V. Reused while it is complete."""
    seed = tree / f'build/shader-seed/{version}'
    if (seed / 'seed-manifest.json').is_file() and next(seed.glob('*.spv'), None):
        return seed
    say('3/4 shaders: translating the game\'s shaders ahead (the startup seed, about 4 minutes)')
    B.run([tree / '.venv/bin/python', botw / 'scripts/prepare_shader_seed.py', '--romfs', tree / f'extracted/{version}/romfs',
           '--output', seed, '--translator', translator, '--spirv-cross', spirv_cross, '--version', version,
           '--modules', tree / f'rewritten/{version}', '--shader-list', REPO / 'games/botw/assets/shader_list.tsv'],
          logs / 'prepare-shader-seed.log', cwd=tree, env=dict(os.environ, TOTK_SEED_SPIRV='1'))
    return seed


def import_seed(seed, cache):
    """The seed into the player's shader cache, as the Mac runtime imports its seed at the first start: the files the cache
    lacks, and each shader's variant index merged (the cache's own rows first, then the seed's new ones)."""
    cache.mkdir(parents=True, exist_ok=True)
    copied = merged = 0
    for src in seed.iterdir():
        dst = cache / src.name
        if src.name.endswith('.variants'):
            rows = json.loads(src.read_text())
            have = json.loads(dst.read_text()) if dst.is_file() else []
            names = {r.get('variant') for r in have}
            new = [r for r in rows if r.get('variant') not in names]
            if new:
                tmp = dst.with_name(dst.name + '.partial'); tmp.write_text(json.dumps(have + new, separators=(',', ':'))); tmp.replace(dst)
                merged += 1
        elif src.name.endswith(('_b.json', '_b.metal', '.spv')) and not dst.exists():
            place(src, dst); copied += 1
    say(f'3/4 shaders: {copied} prepared shader files and {merged} variant indexes into {cache}')


def step_install(tree, version, binary, translator, spirv_cross, out, logs):
    expected = B.VERSIONS[version]
    say(f'4/4 game: installing into {out}')
    game = out / 'game'
    files = [f'rewritten/{version}/main.patched.bin', f'rewritten/{version}/sdk.patched.bin', f'rewritten/{version}/subsdk0.patched.bin',
             f'extracted/{version}/romfs/runtime-index.bin', f'extracted/{version}/romfs/romfs.bin', f'extracted/{version}/romfs/index.json']
    for index in expected.get('aoc', {}):
        if (tree / f'extracted/{version}/aoc/{index}/index.json').is_file():
            files += [f'extracted/{version}/aoc/{index}/romfs.bin', f'extracted/{version}/aoc/{index}/runtime-index.bin', f'extracted/{version}/aoc/{index}/index.json']
    for rel in files:
        place(tree / rel, game / rel)
    stale = game / f'extracted/{version}/aoc'
    if stale.is_dir():   # packs not given this time go (the runtime takes the packs it finds)
        for p in stale.iterdir():
            if p.name not in expected.get('aoc', {}) or not (tree / f'extracted/{version}/aoc/{p.name}/index.json').is_file():
                shutil.rmtree(p)
    fonts = tree / 'extracted/firmware/shared-fonts'
    shutil.rmtree(game / 'extracted/firmware/shared-fonts', ignore_errors=True)
    shutil.copytree(fonts, game / 'extracted/firmware/shared-fonts')
    for src, name in ((binary, 'botw-native'), (translator, 'totk_xlate'), (spirv_cross, 'spirv-cross')):
        place(src, out / 'bin' / name)
        os.chmod(out / 'bin' / name, 0o755)
    launcher = out / 'botw'
    launcher.write_text((LINUX / 'botw.sh').read_text().replace('@VERSION@', version).replace('@DISPLAY@', expected['display_version']))
    os.chmod(launcher, 0o755)
    (out / 'build-info.json').write_text(json.dumps(dict(version=version, display_version=expected['display_version'],
                                                         built=time.strftime('%Y-%m-%d %H:%M:%S'), source=str(REPO)), indent=2) + '\n')


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('game', type=Path, help='the base game NSP or NSZ (with --update) or a 1.6.0 card image (XCI)')
    p.add_argument('-k', '--keys', type=Path, required=True, help='prod.keys dumped from your Switch')
    p.add_argument('--update', type=Path, help='the 1.9.0 (or 1.6.0) update NSP / NSZ')
    p.add_argument('--dlc', type=Path, action='append', help='an add-on content NSP (1.9.0; repeat for both packs)')
    p.add_argument('--firmware', type=Path, help='a folder with your firmware NCAs (for the system shared fonts)')
    p.add_argument('--shared-fonts', type=Path, help='a folder with the six shared-font .bfttf files from an earlier build')
    p.add_argument('--work', type=Path, help='work folder (default: BotW_build next to the game file; about 22 GB, '
                                            '36 GB for a compressed base game)')
    p.add_argument('--out', type=Path, default=Path.home() / 'Applications/botw', help='install folder (default ~/Applications/botw)')
    return p.parse_args(argv)


def main(argv):
    a = parse_args(argv)
    game, keys = a.game.expanduser().resolve(), a.keys.expanduser().resolve()
    if not game.is_file():
        raise B.BuildError(f'no such file: {game}')
    if not keys.is_file():
        raise B.BuildError(f'no such file: {keys}')
    out = a.out.expanduser().resolve()
    work = (a.work.expanduser().resolve() if a.work else game.parent / 'BotW_build')
    kind = B.input_kind(game)
    update = a.update.expanduser().resolve() if a.update else None
    if update and (not update.is_file() or B.input_kind(update) not in ('nsp', 'nsz')):
        raise B.BuildError('--update must be the cumulative update NSP or NSZ')
    dlcs = list(dict.fromkeys(d.expanduser().resolve() for d in a.dlc or []))
    for d in dlcs:
        if not d.is_file() or B.input_kind(d) != 'nsp':
            raise B.BuildError(f'--dlc must be an add-on content NSP: {d}')
    firmware = a.firmware.expanduser().resolve() if a.firmware else None
    shared_fonts = a.shared_fonts.expanduser().resolve() if a.shared_fonts else None
    if firmware and not firmware.is_dir():
        raise B.BuildError(f'no firmware folder: {firmware}')
    if shared_fonts:
        B.font_report(shared_fonts)
    if not (firmware or shared_fonts or B.fonts_valid(work / 'totk/extracted/firmware/shared-fonts')):
        raise B.BuildError('Breath of the Wild needs the system shared fonts: --firmware <folder with your firmware NCAs> '
                           'or --shared-fonts <folder with the six .bfttf files>')
    py = prerequisites()
    B.build_hactool = build_hactool   # step_data_botw builds hactool through this name
    logs = work / 'logs'
    t0 = time.time()
    tree = B.prepare_work(work, py, logs)
    B.check_package(game, 'botw', 'base')   # by the tickets, before a compressed game is expanded
    if update:
        B.check_package(update, 'botw', 'update')
    elif kind != 'xci':
        raise B.BuildError('a base NSP needs its update: pass the 1.9.0 (or 1.6.0) update with --update <update.nsp>')
    B.check_dlc(dlcs, B.BOTW190)
    game = expand_nsz(tree, game, keys, logs, 'base game')
    which = B.identify_game(tree, game, keys, logs)
    if which != 'botw':
        raise B.BuildError(f'this is {B.APPS[which][0]}: the Linux port builds Breath of the Wild only for now')
    if update:
        update = expand_nsz(tree, update, keys, logs, 'update')
        version = B.identify_botw_update(tree, update, keys, logs)
    else:
        version = 'botw160'
    if dlcs and version != 'botw190':
        raise B.BuildError('--dlc takes the add-on content of 1.9.0; a 1.6.0 card image brings its own')
    expected = B.VERSIONS[version]
    botw = work / 'botw'
    say(f'Breath of the Wild {expected["display_version"]}; work folder {work}')
    B.step_code_botw(tree, botw, update or game, keys, dict(os.environ), logs, version=version)
    B.step_data_botw(tree, botw, game, keys, logs, version=version, update=update, dlc=dlcs, firmware=firmware, shared_fonts=shared_fonts)
    spirv = build_spirv_cross(tree, logs)
    translator = build_translator(tree, logs)
    seed = step_seed(tree, botw, version, translator, spirv, logs)
    step_headers(tree, version, logs)
    binary = step_binary(tree, version, logs)
    step_install(tree, version, binary, translator, spirv, out, logs)
    data = Path(os.environ.get('XDG_DATA_HOME') or Path.home() / '.local/share') / 'MacNativeGames/BotW'
    import_seed(seed, data / 'shader-cache/rt-totk-native-v7-unverified')   # the launcher's TOTK_SHADER_CACHE
    say(f'done in {(time.time() - t0) / 60:.0f} min: run {out / "botw"} (the first start verifies the 15 GB game data once)')


if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except B.BuildError as e:
        print(f'\nERROR: {e}', file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
