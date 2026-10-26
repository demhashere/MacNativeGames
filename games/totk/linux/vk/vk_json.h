/* A small JSON reader/writer for the Vulkan backend: shader reflection (<variant>_b.json), variant indexes
   (<shader>.variants) and translator requests. Objects keep their key order; jv_dump writes compact JSON with sorted
   keys the way NSJSONSerialization (NSJSONWritingSortedKeys) does, so variant names hash the same as on the Mac. */
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef enum { JV_NULL, JV_BOOL, JV_NUM, JV_STR, JV_ARR, JV_OBJ } jv_kind;
typedef struct jv jv;
struct jv {
    jv_kind k;
    int b;
    double n;
    char *s;            /* JV_STR */
    int len, cap;       /* JV_ARR / JV_OBJ: element count */
    jv **items;         /* JV_ARR elements, JV_OBJ values */
    char **keys;        /* JV_OBJ keys */
};

jv *jv_parse(const char *text, size_t n, char *err, size_t errlen);
jv *jv_parse_file(const char *path);   /* NULL when missing or invalid */
void jv_free(jv *v);

jv *jv_get(const jv *obj, const char *key);           /* NULL when absent or obj is not an object */
const char *jv_str(const jv *v);                       /* NULL unless a string */
double jv_num(const jv *v, double fallback);
int jv_int(const jv *v, int fallback);
int jv_truthy(const jv *v);                            /* Python truth */

jv *jv_new_obj(void);
jv *jv_new_arr(void);
jv *jv_new_str(const char *s);
jv *jv_new_num(double n);
jv *jv_new_bool(int b);
void jv_set(jv *obj, const char *key, jv *val);       /* replaces an existing key */
void jv_push(jv *arr, jv *val);
jv *jv_clone(const jv *v);
int jv_equal(const jv *a, const jv *b);

/* Compact JSON with keys sorted (byte order, as NSString compare: for ASCII keys); returns malloc'd text. */
char *jv_dump_sorted(const jv *v);
/* Compact JSON in insertion order (requests). */
char *jv_dump(const jv *v);
