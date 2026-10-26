/* Language of the host's own text on screen (boot screen, warm-up progress, overlays, keyboard prompt): Chinese when the
   Mac's language is Chinese, English otherwise. TOTK_UI_LANG=zh or en overrides it (read on every call, so tests can
   switch). The game's own language is chosen separately (TOTK_LANG, services.c desired_language). */
#pragma once
#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#endif
#include <stdlib.h>
#include <string.h>

static inline int totk_ui_zh(void) {
    const char *e = getenv("TOTK_UI_LANG");
    if (e && *e) return !strncmp(e, "zh", 2);
    static int system_zh = -1;
    if (system_zh < 0) {
        char language[16] = {0};
#ifdef __APPLE__
        CFLocaleRef locale = CFLocaleCopyCurrent();
        CFStringRef value = CFLocaleGetValue(locale, kCFLocaleLanguageCode);
        if (value) CFStringGetCString(value, language, sizeof language, kCFStringEncodingUTF8);
        CFRelease(locale);
#else   /* Linux: the desktop's message language (LC_ALL, LC_MESSAGES, LANG) */
        const char *l = getenv("LC_ALL"); if (!l || !*l) l = getenv("LC_MESSAGES"); if (!l || !*l) l = getenv("LANG");
        if (l) { strncpy(language, l, 2); language[2] = 0; }
#endif
        system_zh = !strcmp(language, "zh");
    }
    return system_zh;
}
#define TOTK_UI(zh, en) (totk_ui_zh() ? (zh) : (en))
