/* SHA1 and blake2b-256 OpenCL: Copyright (c) Volcano Authors 2018.
 * Licensed under the GPLv3.
 *
 * This code is not a general-purpose SHA1 implementation, though it should
 * be easy to make it into one. See the companion CPU-side code, ocl-sha1.cpp.
 *
 * This code takes a halfway-done B2SHAstate where the values in
 * hash[SHA_DIGEST_LEN] are the halfway-done digest (computed on the CPU),
 * bytesRemaining is the number of bytes (packed as an array of B2SHAbuffer)
 * left to digest, and length is the total length of the message.
 *
 * Bytes in a B2SHAbuffer past the end of the message *must* be set to 0, even
 * though the bytesRemaining and len indicate they should be ignored.
 *
 * This code the resumes the SHA1_Update() process and computes SHA1_Final(),
 * outputting the final hash.
 *
 * Do one SHA1_Update per thread. If you don't want the SHA1_Final done (why?)
 * this could be tweaked to add another field in B2SHAstate 'numBlocks' and
 * only do that many SHA1_Update passes, not adding the final padding / len.
 *
 * mod(bytesRemaining, 64) == mod(len, 64) must always be true, since a
 * B2SHAbuffer contains 64 bytes.
 */

#define UINT_64BYTES (64/sizeof(unsigned int))
typedef struct {
  unsigned int buffer[UINT_64BYTES];
} B2SHAbuffer;

#define SHA_DIGEST_LEN (5)
#define B2H_DIGEST_LEN (8)
typedef struct {
  unsigned long b2iv[B2H_DIGEST_LEN];
  unsigned int shaiv[SHA_DIGEST_LEN];
  unsigned int len;  // The overall length of the message to digest.
  unsigned int bytesRemaining;  // Bytes to be digested on the GPU.
  unsigned int buffers;  // Buffers to be digested.
} B2SHAconst;

typedef struct {
  unsigned long b2hash[B2H_DIGEST_LEN];
  unsigned int hash[SHA_DIGEST_LEN];

  // counterPos is the position in the input of the last ASCII digit of the
  // counter to increment while searching for a match. Incrementing a number
  // stored as ASCII is done by incrementing the last digit, then carrying the
  // 1 to the next digit (repeat as many times as needed).
  unsigned int counterPos;

  // counts is the number of increments done across all workers. (The workers
  // figure out on their own who gets to do any odd work items.)
  //
  // When a match is found, state[idx].matchLen is set to the length of the
  // substring that matches, and states[idx].matchCount is the remainder (how
  // many counts were left). Otherwise both are set to 0.
  unsigned long counts;
  unsigned long matchCount;
  unsigned int matchLen;
  unsigned int matchCtimeCount;
  unsigned int ctimePos;
  unsigned int ctimeCount;
} B2SHAstate;

#define rotl(a, n) rotate((a), (n)) 
#define rotr(a, n) rotate((a), 64-(n)) 

unsigned int swap(unsigned int val) {
    return (rotate(((val) & 0x00FF00FF), 24U) |
            rotate(((val) & 0xFF00FF00), 8U));
}

#define mod(x, y) ((x) - ((x)/(y) * (y)))

#define F2(x, y, z) ((x) ^ (y) ^ (z))
#define F1(x, y, z) (bitselect(z, y, x))
#define F0(x, y, z) (bitselect(x, y, (x ^ z)))

// Notice that in big-endian, this counts from 0 - 7
#define SHA1M_A 0x67452301u
// Notice that in big-endian, this counts from 8 - f
#define SHA1M_B 0xefcdab89u
// Notice that in big-endian, this counts down, f - 8
#define SHA1M_C 0x98badcfeu
// Notice that in big-endian, this counts down, 7 - 0
#define SHA1M_D 0x10325476u
// Somewhat arbitrary:
#define SHA1M_E 0xc3d2e1f0u

#define SHA1C00 0x5a827999u
#define SHA1C01 0x6ed9eba1u
#define SHA1C02 0x8f1bbcdcu
#define SHA1C03 0xca62c1d6u

#define SHA1step(f, a, b, c, d, e, k, x) \
{ \
  e += k;            \
  e += x;            \
  e += f(b, c, d);   \
  e += rotl(a,  5u); \
  b  = rotl(b, 30u); \
}

#define shuffleAndSHA1(w, f, a, b, c, d, e, k, x) \
{ \
  x = rotl(w ^ x, 1u); \
  SHA1step(f, a, b, c, d, e, k, x); \
}

static void sha1_update(uint4 *WV, unsigned int *digest) {
  unsigned int A = digest[0];
  unsigned int B = digest[1];
  unsigned int C = digest[2];
  unsigned int D = digest[3];
  unsigned int E = digest[4];

  #define w0_t WV[0].s0
  #define w1_t WV[0].s1
  #define w2_t WV[0].s2
  #define w3_t WV[0].s3
  #define w4_t WV[1].s0
  #define w5_t WV[1].s1
  #define w6_t WV[1].s2
  #define w7_t WV[1].s3
  #define w8_t WV[2].s0
  #define w9_t WV[2].s1
  #define wa_t WV[2].s2
  #define wb_t WV[2].s3
  #define wc_t WV[3].s0
  #define wd_t WV[3].s1
  #define we_t WV[3].s2
  #define wf_t WV[3].s3

  // Round 1 - loop unrolled.
  unsigned int shaK = SHA1C00;
  SHA1step(F1, A, B, C, D, E, shaK, w0_t);
  SHA1step(F1, E, A, B, C, D, shaK, w1_t);
  SHA1step(F1, D, E, A, B, C, shaK, w2_t);
  SHA1step(F1, C, D, E, A, B, shaK, w3_t);
  SHA1step(F1, B, C, D, E, A, shaK, w4_t);
  SHA1step(F1, A, B, C, D, E, shaK, w5_t);
  SHA1step(F1, E, A, B, C, D, shaK, w6_t);
  SHA1step(F1, D, E, A, B, C, shaK, w7_t);
  SHA1step(F1, C, D, E, A, B, shaK, w8_t);
  SHA1step(F1, B, C, D, E, A, shaK, w9_t);
  SHA1step(F1, A, B, C, D, E, shaK, wa_t);
  SHA1step(F1, E, A, B, C, D, shaK, wb_t);
  SHA1step(F1, D, E, A, B, C, shaK, wc_t);
  SHA1step(F1, C, D, E, A, B, shaK, wd_t);
  SHA1step(F1, B, C, D, E, A, shaK, we_t);
  SHA1step(F1, A, B, C, D, E, shaK, wf_t);
  shuffleAndSHA1(wd_t ^ w8_t ^ w2_t, F1, E, A, B, C, D, shaK, w0_t);
  shuffleAndSHA1(we_t ^ w9_t ^ w3_t, F1, D, E, A, B, C, shaK, w1_t);
  shuffleAndSHA1(wf_t ^ wa_t ^ w4_t, F1, C, D, E, A, B, shaK, w2_t);
  shuffleAndSHA1(w0_t ^ wb_t ^ w5_t, F1, B, C, D, E, A, shaK, w3_t);

  // Round 2 - loop unrolled.
  shaK = SHA1C01;
  shuffleAndSHA1(w1_t ^ wc_t ^ w6_t, F2, A, B, C, D, E, shaK, w4_t);
  shuffleAndSHA1(w2_t ^ wd_t ^ w7_t, F2, E, A, B, C, D, shaK, w5_t);
  shuffleAndSHA1(w3_t ^ we_t ^ w8_t, F2, D, E, A, B, C, shaK, w6_t);
  shuffleAndSHA1(w4_t ^ wf_t ^ w9_t, F2, C, D, E, A, B, shaK, w7_t);
  shuffleAndSHA1(w5_t ^ w0_t ^ wa_t, F2, B, C, D, E, A, shaK, w8_t);
  shuffleAndSHA1(w6_t ^ w1_t ^ wb_t, F2, A, B, C, D, E, shaK, w9_t);
  shuffleAndSHA1(w7_t ^ w2_t ^ wc_t, F2, E, A, B, C, D, shaK, wa_t);
  shuffleAndSHA1(w8_t ^ w3_t ^ wd_t, F2, D, E, A, B, C, shaK, wb_t);
  shuffleAndSHA1(w9_t ^ w4_t ^ we_t, F2, C, D, E, A, B, shaK, wc_t);
  shuffleAndSHA1(wa_t ^ w5_t ^ wf_t, F2, B, C, D, E, A, shaK, wd_t);
  shuffleAndSHA1(wb_t ^ w6_t ^ w0_t, F2, A, B, C, D, E, shaK, we_t);
  shuffleAndSHA1(wc_t ^ w7_t ^ w1_t, F2, E, A, B, C, D, shaK, wf_t);
  shuffleAndSHA1(wd_t ^ w8_t ^ w2_t, F2, D, E, A, B, C, shaK, w0_t);
  shuffleAndSHA1(we_t ^ w9_t ^ w3_t, F2, C, D, E, A, B, shaK, w1_t);
  shuffleAndSHA1(wf_t ^ wa_t ^ w4_t, F2, B, C, D, E, A, shaK, w2_t);
  shuffleAndSHA1(w0_t ^ wb_t ^ w5_t, F2, A, B, C, D, E, shaK, w3_t);
  shuffleAndSHA1(w1_t ^ wc_t ^ w6_t, F2, E, A, B, C, D, shaK, w4_t);
  shuffleAndSHA1(w2_t ^ wd_t ^ w7_t, F2, D, E, A, B, C, shaK, w5_t);
  shuffleAndSHA1(w3_t ^ we_t ^ w8_t, F2, C, D, E, A, B, shaK, w6_t);
  shuffleAndSHA1(w4_t ^ wf_t ^ w9_t, F2, B, C, D, E, A, shaK, w7_t);

  // Round 3 - loop unrolled.
  shaK = SHA1C02;
  shuffleAndSHA1(w5_t ^ w0_t ^ wa_t, F0, A, B, C, D, E, shaK, w8_t);
  shuffleAndSHA1(w6_t ^ w1_t ^ wb_t, F0, E, A, B, C, D, shaK, w9_t);
  shuffleAndSHA1(w7_t ^ w2_t ^ wc_t, F0, D, E, A, B, C, shaK, wa_t);
  shuffleAndSHA1(w8_t ^ w3_t ^ wd_t, F0, C, D, E, A, B, shaK, wb_t);
  shuffleAndSHA1(w9_t ^ w4_t ^ we_t, F0, B, C, D, E, A, shaK, wc_t);
  shuffleAndSHA1(wa_t ^ w5_t ^ wf_t, F0, A, B, C, D, E, shaK, wd_t);
  shuffleAndSHA1(wb_t ^ w6_t ^ w0_t, F0, E, A, B, C, D, shaK, we_t);
  shuffleAndSHA1(wc_t ^ w7_t ^ w1_t, F0, D, E, A, B, C, shaK, wf_t);
  shuffleAndSHA1(wd_t ^ w8_t ^ w2_t, F0, C, D, E, A, B, shaK, w0_t);
  shuffleAndSHA1(we_t ^ w9_t ^ w3_t, F0, B, C, D, E, A, shaK, w1_t);
  shuffleAndSHA1(wf_t ^ wa_t ^ w4_t, F0, A, B, C, D, E, shaK, w2_t);
  shuffleAndSHA1(w0_t ^ wb_t ^ w5_t, F0, E, A, B, C, D, shaK, w3_t);
  shuffleAndSHA1(w1_t ^ wc_t ^ w6_t, F0, D, E, A, B, C, shaK, w4_t);
  shuffleAndSHA1(w2_t ^ wd_t ^ w7_t, F0, C, D, E, A, B, shaK, w5_t);
  shuffleAndSHA1(w3_t ^ we_t ^ w8_t, F0, B, C, D, E, A, shaK, w6_t);
  shuffleAndSHA1(w4_t ^ wf_t ^ w9_t, F0, A, B, C, D, E, shaK, w7_t);
  shuffleAndSHA1(w5_t ^ w0_t ^ wa_t, F0, E, A, B, C, D, shaK, w8_t);
  shuffleAndSHA1(w6_t ^ w1_t ^ wb_t, F0, D, E, A, B, C, shaK, w9_t);
  shuffleAndSHA1(w7_t ^ w2_t ^ wc_t, F0, C, D, E, A, B, shaK, wa_t);
  shuffleAndSHA1(w8_t ^ w3_t ^ wd_t, F0, B, C, D, E, A, shaK, wb_t);

  // Round 4 - loop unrolled.
  shaK = SHA1C03;
  shuffleAndSHA1(w9_t ^ w4_t ^ we_t, F2, A, B, C, D, E, shaK, wc_t);
  shuffleAndSHA1(wa_t ^ w5_t ^ wf_t, F2, E, A, B, C, D, shaK, wd_t);
  shuffleAndSHA1(wb_t ^ w6_t ^ w0_t, F2, D, E, A, B, C, shaK, we_t);
  shuffleAndSHA1(wc_t ^ w7_t ^ w1_t, F2, C, D, E, A, B, shaK, wf_t);
  shuffleAndSHA1(wd_t ^ w8_t ^ w2_t, F2, B, C, D, E, A, shaK, w0_t);
  shuffleAndSHA1(we_t ^ w9_t ^ w3_t, F2, A, B, C, D, E, shaK, w1_t);
  shuffleAndSHA1(wf_t ^ wa_t ^ w4_t, F2, E, A, B, C, D, shaK, w2_t);
  shuffleAndSHA1(w0_t ^ wb_t ^ w5_t, F2, D, E, A, B, C, shaK, w3_t);
  shuffleAndSHA1(w1_t ^ wc_t ^ w6_t, F2, C, D, E, A, B, shaK, w4_t);
  shuffleAndSHA1(w2_t ^ wd_t ^ w7_t, F2, B, C, D, E, A, shaK, w5_t);
  shuffleAndSHA1(w3_t ^ we_t ^ w8_t, F2, A, B, C, D, E, shaK, w6_t);
  shuffleAndSHA1(w4_t ^ wf_t ^ w9_t, F2, E, A, B, C, D, shaK, w7_t);
  shuffleAndSHA1(w5_t ^ w0_t ^ wa_t, F2, D, E, A, B, C, shaK, w8_t);
  shuffleAndSHA1(w6_t ^ w1_t ^ wb_t, F2, C, D, E, A, B, shaK, w9_t);
  shuffleAndSHA1(w7_t ^ w2_t ^ wc_t, F2, B, C, D, E, A, shaK, wa_t);
  shuffleAndSHA1(w8_t ^ w3_t ^ wd_t, F2, A, B, C, D, E, shaK, wb_t);
  shuffleAndSHA1(w9_t ^ w4_t ^ we_t, F2, E, A, B, C, D, shaK, wc_t);
  shuffleAndSHA1(wa_t ^ w5_t ^ wf_t, F2, D, E, A, B, C, shaK, wd_t);
  shuffleAndSHA1(wb_t ^ w6_t ^ w0_t, F2, C, D, E, A, B, shaK, we_t);
  shuffleAndSHA1(wc_t ^ w7_t ^ w1_t, F2, B, C, D, E, A, shaK, wf_t);

  digest[0] += A;
  digest[1] += B;
  digest[2] += C;
  digest[3] += D;
  digest[4] += E;
  #undef w0_t
  #undef w1_t
  #undef w2_t
  #undef w3_t
  #undef w4_t
  #undef w5_t
  #undef w6_t
  #undef w7_t
  #undef w8_t
  #undef w9_t
  #undef wa_t
  #undef wb_t
  #undef wc_t
  #undef wd_t
  #undef we_t
  #undef wf_t
}


#define tail mod(fixed->len, 64)
static inline void write_padding(uint4* WV, __constant B2SHAconst* fixed) {
  unsigned int padding = swap(0x80 << (mod(fixed->len, 4) * 8));
  uint4 v = WV[tail/16];
  switch (mod(fixed->len, 16)/4) {
    case 0: v.s0 |= padding; break;
    case 1: v.s1 |= padding; break;
    case 2: v.s2 |= padding; break;
    case 3: v.s3 |= padding; break;
  }
  WV[tail/16] = v;
}

static inline void write_len(uint4* WV, __constant B2SHAconst* fixed) {
  WV[3].s2 = fixed->len >> (32-3);
  WV[3].s3 = fixed->len << 3;
}

static void sha1(__constant B2SHAconst* fixed,
                 __global B2SHAstate* state,
                 __global const B2SHAbuffer* src) {
  unsigned int hash[SHA_DIGEST_LEN] = {
    fixed->shaiv[0],  // Hash IV must be set on CPU.
    fixed->shaiv[1],
    fixed->shaiv[2],
    fixed->shaiv[3],
    fixed->shaiv[4],
  };

  unsigned int bytesRemaining = fixed->bytesRemaining;
  int i;
  for (i = 0; bytesRemaining; i++, src++) {
    uint4 WV[UINT_64BYTES/4];
    // Copy 64 bytes from src->buffer[], swapping to big-endian.
    // NOTE: src->buffer[] bytes past "bytesRemaining" *must* be provided as 0.
    for (int j = 0; j < UINT_64BYTES; ) {
      WV[j/4].s0 = swap(src->buffer[j]);
      j++;
      WV[j/4].s1 = swap(src->buffer[j]);
      j++;
      WV[j/4].s2 = swap(src->buffer[j]);
      j++;
      WV[j/4].s3 = swap(src->buffer[j]);
      j++;
    }

    // If this will be the last loop and some of {padding,len} should be added.
    if (bytesRemaining < 64) {
      bytesRemaining = 0;
      if (tail != 0) {
        write_padding(WV, fixed);
        if (tail < 56) {
          write_len(WV, fixed);
        }
      }
    } else {
      bytesRemaining -= 64;
    }

    sha1_update(WV, hash);
  }

  // If an additional block is needed just to write len
  if (tail >= 56) {
    uint4 WV[UINT_64BYTES/4] = {0, 0, 0, 0};
    if (tail == 0) {
      write_padding(WV, fixed);
    }
    write_len(WV, fixed);

    sha1_update(WV, hash);
  }

  for (i = 0; i < SHA_DIGEST_LEN; i++) {
    state->hash[i] = swap(hash[i]);
  }
}



typedef unsigned char uint8_t;
typedef unsigned int uint32_t;
typedef unsigned long uint64_t;

#define B2_128BYTES (128)
#define B2_OUTSIZE B2H_DIGEST_LEN

#define BLAKE2_EXABYTE_NOT_EXPECTED (1)
// Optimization: blake2 can overflow a uint64_t only if messages larger than
// 17 exabytes are seen. You must then set BLAKE2_EXABYTE_NOT_EXPECTED (2)
typedef struct
{
  uint64_t h[B2_OUTSIZE];
  uint64_t t[BLAKE2_EXABYTE_NOT_EXPECTED];
  uint64_t f[1];
  uint32_t buf[B2_128BYTES / sizeof(uint32_t)];
  uint32_t buflen;
} blake2b_state;

static const uint32_t __constant blake2b_sigma32[12][4] = {
  { 0x00010203, 0x04050607, 0x08090a0b, 0x0c0d0e0f } ,
  { 0x0e0a0408, 0x090f0d06, 0x010c0002, 0x0b070503 } ,
  { 0x0b080c00, 0x05020f0d, 0x0a0e0306, 0x07010904 } ,
  { 0x07090301, 0x0d0c0b0e, 0x0206050a, 0x04000f08 } ,
  { 0x09000507, 0x02040a0f, 0x0e010b0c, 0x0608030d } ,
  { 0x020c060a, 0x000b0803, 0x040d0705, 0x0f0e0109 } ,
  { 0x0c05010f, 0x0e0d040a, 0x00070603, 0x0902080b } ,
  { 0x0d0b070e, 0x0c010309, 0x05000f04, 0x0806020a } ,
  { 0x060f0e09, 0x0b030008, 0x0c020d07, 0x01040a05 } ,
  { 0x0a020804, 0x07060105, 0x0f0b090e, 0x030c0d00 } ,
  { 0x00010203, 0x04050607, 0x08090a0b, 0x0c0d0e0f } ,
  { 0x0e0a0408, 0x090f0d06, 0x010c0002, 0x0b070503 } ,
};

// Next optimization: create another G32 that splits vvb into sep vars.
// This will reduce a lot of the copying around in ROUND
// vv[18/2] can be completely eliminated

#define G32(r,i1,vva,vvb,vvc,vvd) \
  do { \
    vva += vvb;                            \
    unsigned int s = blake2b_sigma32[r][i1/2]; \
    vva.s0 += m[s >> 24]; \
    vva.s1 += m[(s >> 8) & 0xf]; \
    vvd = rotr(vvd ^ vva, 32lu);           \
    vvc += vvd;                            \
    vvb = rotr(vvb ^ vvc, 24lu);           \
    vva += vvb;                            \
    vva.s0 += m[(s >> 16) & 0xf]; \
    vva.s1 += m[s & 0xf]; \
    vvd = rotr(vvd ^ vva, 16lu);           \
    vvc += vvd;                            \
    vvb = rotr(vvb ^ vvc, 63lu);           \
  } while (0)

#define G32splitB(r,i1,vva,vb1,vb2,vvc,vvd) \
  do { \
    unsigned int s = blake2b_sigma32[r][i1/2]; \
    vva += (ulong2) (vb1 + m[s >> 24], vb2 + m[(s >> 8) & 0xf]); \
    vvd = rotr(vvd ^ vva, 32lu);           \
    vvc += vvd;                            \
    vb1 = rotr(vb1 ^ vvc.s0, 24lu);        \
    vb2 = rotr(vb2 ^ vvc.s1, 24lu);        \
    vva += (ulong2) (vb1 + m[(s >> 16) & 0xf], vb2 + m[s & 0xf]); \
    vvd = rotr(vvd ^ vva, 16lu);           \
    vvc += vvd;                            \
    vb1 = rotr(vb1 ^ vvc.s0, 63lu);        \
    vb2 = rotr(vb2 ^ vvc.s1, 63lu);        \
  } while (0)

#define G2v(r,i1,a,b,c,d) \
  G32(r,i1,vv[a/2],vv[b/2],vv[c/2],vv[d/2])

#define G2vsplit(r,i1,a,vb1,vb2,c,d) \
  G32splitB(r,i1,vv[a/2],vb1,vb2,vv[c/2],vv[d/2])

#define ROUND(r)           \
  do {                     \
    G2v(r,0, 0, 4, 8,12); \
    G2v(r,2, 2, 6,10,14); \
    vv[16/2] = (ulong2)(vv[15/2].s1, vv[12/2].s0); \
    G2vsplit(r,4, 0,vv[5/2].s1,vv[6/2].s0,10,16); \
    vv[12/2] = (ulong2)(vv[13/2].s1, vv[14/2].s0); \
    G2vsplit(r,6, 2,vv[7/2].s1,vv[4/2].s0, 8,12); \
    vv[14/2] = (ulong2)(vv[13/2].s1, vv[16/2].s0); \
    vv[12/2] = (ulong2)(vv[17/2].s1, vv[12/2].s0); \
  } while(0)

static void blake2b_compress(
    __constant B2SHAconst* fixed,
    blake2b_state *S,
    const uint32_t block[B2_128BYTES/sizeof(unsigned int)]) {
  uint64_t m[16];
  uint32_t i;

  for( i = 0; i < 16; ++i ) {
    m[i] = block[i*2] | ((uint64_t)block[i*2 + 1] << 32);
  }

  ulong2 vv[9] = {
    { S->h[0], S->h[1] },
    { S->h[2], S->h[3] },
    { S->h[4], S->h[5] },
    { S->h[6], S->h[7] },
    { fixed->b2iv[0], fixed->b2iv[1] },
    { fixed->b2iv[2], fixed->b2iv[3] },
    { fixed->b2iv[4] ^ S->t[0],
#if BLAKE2_EXABYTE_NOT_EXPECTED > 1
//      fixed->b2iv[5] ^ S->t[1] },
#else
      fixed->b2iv[5] },
#endif
    { fixed->b2iv[6] ^ S->f[0],
      fixed->b2iv[7] /* ^ S->f[1] removed: no last_node */ },
    { 0, 0 },
  };

  ROUND( 0 );
  ROUND( 1 );
  ROUND( 2 );
  ROUND( 3 );
  ROUND( 4 );
  ROUND( 5 );
  ROUND( 6 );
  ROUND( 7 );
  ROUND( 8 );
  ROUND( 9 );
  ROUND( 10 );
  ROUND( 11 );

  for( i = 0; i < B2_OUTSIZE/2; ++i ) {
    ulong2 x = vv[i] ^ vv[i + 4];
    S->h[i*2  ] ^= x.s0;
    S->h[i*2+1] ^= x.s1;
  }
}

#undef G
#undef ROUND

static inline void blake2b_increment_counter(blake2b_state *S, uint64_t inc) {
  S->t[0] += inc;
  // Optimization:
#if BLAKE2_EXABYTE_NOT_EXPECTED > 1
  S->t[1] += ( S->t[0] < inc );
#endif
}

static inline void blake2b_update(__constant B2SHAconst* fixed,
                                  __global B2SHAstate* state,
                                  __global const B2SHAbuffer* src) {
  blake2b_state S;

  // inline blake2b_init:
  S.t[0] = 0;
#if BLAKE2_EXABYTE_NOT_EXPECTED > 1
  S.t[1] = 0;
#endif
  S.f[0] = 0;

  uint32_t i;
  for( i = 0; i < sizeof(S.buf)/sizeof(S.buf[0]); i++ ) S.buf[i] = 0;
  for( i = 0; i < B2_OUTSIZE; i++ ) S.h[i] = fixed->b2iv[i];
  S.buflen = 0;

  // xor with P->fanout       = 1;
  // xor with P->depth        = 1;
  S.h[0] ^= 0x01010000 | sizeof(S.h);

  // begin blake2b_update:
  uint32_t rem = fixed->bytesRemaining;
  while (rem > B2_128BYTES) {
    __global const unsigned int* in = src->buffer;
    for (i = 0; i < UINT_64BYTES; i++) {
      S.buf[i] = in[i];
    }
    src++;
    in = src->buffer;
    for (; i < UINT_64BYTES*2; i++) {
      S.buf[i] = in[i - UINT_64BYTES];
    }
    blake2b_increment_counter(&S, B2_128BYTES);
    blake2b_compress( fixed, &S, S.buf );
    src++;
    rem -= B2_128BYTES;
  }

  // Prepare for blake2b_final.
  for (i = 0; i < B2_128BYTES/4; i++) S.buf[i] = 0;
  unsigned int words = (rem + sizeof(unsigned int) - 1)/sizeof(unsigned int);
  S.buflen = rem;
  __global const unsigned int* in = src->buffer;
  for (i = 0; i < words && i < UINT_64BYTES; i++) {
    S.buf[i] = in[i];
  }
  if (i < words) {
    src++;
    words -= UINT_64BYTES;
    in = src->buffer;
    for (i = 0; i < words && i < UINT_64BYTES; i++) {
      S.buf[i + UINT_64BYTES] = in[i];
    }
  }

  // inline blake2b_final:
  blake2b_increment_counter( &S, S.buflen );
  // blake2b_set_lastblock( S ) is a single line, so it is inlined as:
  S.f[0] = (uint64_t)-1;
  // Rely on any extra bytes copied from src->buffer to be 0.
  blake2b_compress( fixed, &S, S.buf );

  for (i = 0; i < B2_OUTSIZE; i++) state->b2hash[i] = S.h[i];
}

void asciiIncrement(__global B2SHAbuffer* src, unsigned int i) {
  unsigned int srcIdx = i / 64;
  i &= (64 - 1);
  unsigned int bits = 8*mod(i, sizeof(unsigned int));
  i /= sizeof(unsigned int);
  for (;;) {
    unsigned int digit = src[srcIdx].buffer[i];
    src[srcIdx].buffer[i] = digit +
        (((((digit >> bits) & 0xff) >= 0x39) ? -9 : 1) << bits);
    if (((digit >> bits) & 0xff) < 0x39 /*ASCII '9'*/) {
      break;
    }
    if (bits == 0) {
      bits = 8*sizeof(unsigned int);
      if (i == 0) {
        if (srcIdx == 0) {
          break;
        }
        srcIdx--;
        i = 64;
      }
      i--;
    }
    bits -= 8;
  }
}

void compareB2SHA(__global B2SHAstate* state) {
  unsigned int bits;
  for (bits = 0; bits < B2H_DIGEST_LEN*sizeof(unsigned long); bits++) {
    unsigned long b2h = state->b2hash[bits/sizeof(unsigned long)];
    b2h >>= 8*mod(bits, sizeof(unsigned long));
    if ((b2h & 0xff) == (state->hash[0] & 0xff)) {
      unsigned int i;
      for (i = 1; i < SHA_DIGEST_LEN*sizeof(unsigned int); i++) {
        if (bits + i >= B2H_DIGEST_LEN*sizeof(unsigned long)) break;

        unsigned int sha = state->hash[i/sizeof(unsigned int)];
        sha >>= 8*mod(i, sizeof(unsigned int));
        sha &= 0xff;
        b2h = state->b2hash[(bits+i)/sizeof(unsigned long)];
        b2h = (b2h >> (8*mod(bits+i, sizeof(unsigned long)))) & 0xff;
        if (b2h != sha) break;
      }
      if (i > state->matchLen) {
        state->matchCount = state->counts;
        state->matchLen = i;
        state->matchCtimeCount = state->ctimeCount;
      }
    }
  }
}

void getOldSrc(__global B2SHAbuffer* src, __global B2SHAstate* state,
               unsigned int oldSrc[2]) {
  unsigned int i = state->counterPos;
  unsigned int srcIdx = i / 64;
  i &= (64 - 1);
  i /= sizeof(unsigned int);
  oldSrc[0] = src[srcIdx].buffer[i];
  if (i == 0) {
    srcIdx--;
    i = 64;
  }
  i--;
  oldSrc[1] = src[srcIdx].buffer[i];
}

void restoreOldSrc(__global B2SHAbuffer* src, __global B2SHAstate* state,
                   unsigned int oldSrc[2]) {
  unsigned int i = state->counterPos;
  unsigned int srcIdx = i / 64;
  i &= (64 - 1);
  i /= sizeof(unsigned int);
  src[srcIdx].buffer[i] = oldSrc[0];
  if (i == 0) {
    srcIdx--;
    i = 64;
  }
  i--;
  src[srcIdx].buffer[i] = oldSrc[1];
}

__kernel void main(__constant B2SHAconst* fixed,
                   __global B2SHAstate* states,
                   __global B2SHAbuffer* srcs) {
  #define idx get_global_id(0)
  #define state (&states[idx])
  #define src (&srcs[idx*fixed->buffers])

  unsigned int oldCounts = state->counts;
  unsigned int oldSrc[2];
  getOldSrc(src, state, oldSrc);
  while (state->ctimeCount) {
    while (state->counts) {
      sha1(fixed, state, src);
      blake2b_update(fixed, state, src);
      compareB2SHA(state);
      state->counts--;

      asciiIncrement(src, state->counterPos);
    }
    state->counts = oldCounts;
    restoreOldSrc(src, state, oldSrc);
    asciiIncrement(src, state->ctimePos);
    state->ctimeCount--;
  }
  // Things that might speed it up:
  // 1. OpenCL best practices:
  //    * can tune the register usage with
  // http://developer.download.nvidia.com/compute/cuda/3_2_prod/toolkit/docs/OpenCL_Extensions/cl_nv_compiler_options.txt
  //    * can tune threads per block. 64 is min, 128 or 256 is good
  //    * have the program auto-benchmark different local work sizes
  // 1. Only write to out (only save the hash) if there is a match
}
