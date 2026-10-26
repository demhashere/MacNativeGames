/* CommonCrypto digests on OpenSSL libcrypto (its ARMv8 SHA-256 is as fast as Apple's). */
#pragma once
#include <stdint.h>
#include <openssl/sha.h>
#include <openssl/md5.h>
typedef uint32_t CC_LONG;
#define CC_SHA256_DIGEST_LENGTH SHA256_DIGEST_LENGTH
#define CC_SHA1_DIGEST_LENGTH SHA_DIGEST_LENGTH
#define CC_MD5_DIGEST_LENGTH MD5_DIGEST_LENGTH
#define CC_SHA256_BLOCK_BYTES SHA256_CBLOCK
typedef SHA256_CTX CC_SHA256_CTX;
typedef SHA_CTX CC_SHA1_CTX;
static inline int CC_SHA256_Init(CC_SHA256_CTX *c) { return SHA256_Init(c); }
static inline int CC_SHA256_Update(CC_SHA256_CTX *c, const void *d, CC_LONG n) { return SHA256_Update(c, d, n); }
static inline int CC_SHA256_Final(unsigned char *md, CC_SHA256_CTX *c) { return SHA256_Final(md, c); }
static inline unsigned char *CC_SHA256(const void *d, CC_LONG n, unsigned char *md) { return SHA256((const unsigned char *)d, n, md); }
static inline int CC_SHA1_Init(CC_SHA1_CTX *c) { return SHA1_Init(c); }
static inline int CC_SHA1_Update(CC_SHA1_CTX *c, const void *d, CC_LONG n) { return SHA1_Update(c, d, n); }
static inline int CC_SHA1_Final(unsigned char *md, CC_SHA1_CTX *c) { return SHA1_Final(md, c); }
static inline unsigned char *CC_SHA1(const void *d, CC_LONG n, unsigned char *md) { return SHA1((const unsigned char *)d, n, md); }
