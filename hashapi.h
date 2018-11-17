#include <errno.h>
#include <openssl/sha.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <string>
#include <vector>
#include "blake2.h"

#pragma once

int hexdump(char* buf, size_t buflen, uint8_t* bytes, size_t byteslen);

class Sha1Hash {
public:
  uint8_t result[SHA_DIGEST_LENGTH];

  Sha1Hash() : init_done(false) {}

  // update_and_flush: Not streaming - supply all the data at once.
  void update_and_flush(const char* data, size_t len) {
    SHA1(reinterpret_cast<const uint8_t*>(data), len, result);
  }

  // Streaming: supply data as it is received, call flush() to get hash.
  void update(const char* data, size_t len) {
    if (!init_done) {
      SHA1_Init(&ctx);
      init_done = true;
    }
    SHA1_Update(&ctx, reinterpret_cast<const void*>(data), len);
  }

  void flush() {
    SHA1_Final(result, &ctx);
    init_done = false;
  }

  int dump(char* buf, size_t buflen) {
    return hexdump(buf, buflen, result, sizeof(result));
  }
  SHA_CTX ctx;
protected:
  bool init_done;
};

class Blake2Hash {
public:
  uint8_t result[BLAKE2B_OUTBYTES];

  Blake2Hash() : init_done(false) {}

  // update_and_flush: Not streaming - supply all the data at once.
  void update_and_flush(const char* data, size_t len) {
    blake2b(result, sizeof(result), reinterpret_cast<const uint8_t*>(data),
            // key, keylen are NULL, 0. See blake2b_init() which is what b2sum
            // and other blake2 impl's (such as golang) mimic: NULL key.
            len, NULL, 0);
  }

  // Streaming: supply data as it is received, call flush() to get hash.
  void update(const char* data, size_t len) {
    if (!init_done) {
      init_done = true;
      blake2b_init(&ctx, sizeof(result));
    }
    blake2b_update(&ctx, reinterpret_cast<const void*>(data), len);
  }

  void flush() {
    blake2b_final(&ctx, result, sizeof(result));
    init_done = false;
  }

  int dump(char* buf, size_t buflen) {
    return hexdump(buf, buflen, result, sizeof(result));
  }

  // instr finds where the bytes want[] occur in result.
  // returns: -1 (no match found) or the index of the match.
  // or else finds the largest matching sequence and also sets matchlen.
  int instr(uint8_t* want, size_t wantlen, size_t* matchlen) {
    int best = -1;
    *matchlen = 0;
    if (!wantlen) {
      return best;
    }
    for (size_t n = 0; n < sizeof(result); ) {
      size_t i = 0;
      void* p = memchr(&result[n], want[i], sizeof(result) - n);
      if (!p) {
        return best;
      }
      // memchr matched one byte.
      n = (reinterpret_cast<uint8_t*>(p) - result) + 1;
      i++;
      // Check how many bytes long this match is.
      while (n < sizeof(result) && i < wantlen && want[i] == result[n]) {
        i++;
        n++;
      }
      if (i > *matchlen) {
        best = n - i;
        *matchlen = i;
      }
    }
    return best;
  }

private:
  bool init_done;
  blake2b_state ctx;
};

class CommitMessage {
public:

  // set: take a copy of message (does not retain a pointer to message).
  int set(char* message, size_t len) {
    header.clear();
    parent.clear();
    type = MessageUNKNOWN;

    // Parse message.
    char* p = message;
    if (!strncmp(message, "commit ", 7)) {
      type = MessageCOMMIT;
      p += 7;
    } else if (!strncmp(message, "tree ", 5)) {
      type = MessageTREE;
      p += 5;
    } else {
      fprintf(stderr, "CommitMessage: invalid type \"%.*s\"\n", 4, message);
      return 1;
    }

    int n;
    size_t declared_len = 0;
    if (sscanf(p, "%zu%n", &declared_len, &n) != 1) {
      fprintf(stderr, "CommitMessage: invalid len \"%s\"\n", message);
      return 1;
    }
    if (p[n] != 0) {
      fprintf(stderr, "CommitMessage: missing null byte\n");
      return 1;
    }
    p += n + 1;
    if (len - (p - message) != declared_len) {
      fprintf(stderr, "CommitMessage: bad declared_len: got %zu, want %zu\n",
              declared_len, len - (p - message));
      return 1;
    }

    if (!strncmp(p, "tree ", 5)) {
      p += strcspn(p, "\r\n");
      p += strspn(p, "\r\n");
    } else {
      fprintf(stderr, "CommitMessage: missing tree:\n%s", p);
      return 1;
    }
    header.insert(header.end(), message, p);
    if (!strncmp(p, "parent ", 7)) {
      char* first = p;
      p += strcspn(p, "\r\n");
      p += strspn(p, "\r\n");
      parent.assign(first, p);
    }
    if (!strncmp(p, "author ", 7)) {
      char* first = p;
      p += strcspn(p, "\r\n");
      p += strspn(p, "\r\n");
      author.assign(first, p);
    }
    // Note that committer has the extra \n between committer and log.
    if (!strncmp(p, "committer ", 10)) {
      char* first = p;
      p += strcspn(p, "\r\n");
      p += strspn(p, "\r\n");
      committer.assign(first, p);
    } else {
      fprintf(stderr, "CommitMessage: missing committer:\n%s", p);
      return 1;
    }
    log.assign(p);

    // Parse timestamps
    if (!author.empty()) {
      if (CommitMessage::parseTimestamp(&author, &author_time, &author_tz,
                                        &author_btime)) {
        return 1;
      }
    }
    if (CommitMessage::parseTimestamp(&committer, &committer_time,
                                      &committer_tz, &committer_btime)) {
      return 1;
    }
    return 0;
  }

  int hash(Sha1Hash& sha, Blake2Hash& b2h) {
    sha.update(header.data(), header.size());
    sha.update(parent.c_str(), parent.size());
    sha.update(author.c_str(), author.size());
    sha.update(author_time.c_str(), author_time.size());
    sha.update(author_tz.c_str(), author_tz.size());
    sha.update(committer.c_str(), committer.size());
    sha.update(committer_time.c_str(), committer_time.size());
    sha.update(committer_tz.c_str(), committer_tz.size());
    sha.update(log.c_str(), log.size());
    sha.flush();
    b2h.update(header.data(), header.size());
    b2h.update(parent.c_str(), parent.size());
    b2h.update(author.c_str(), author.size());
    b2h.update(author_time.c_str(), author_time.size());
    b2h.update(author_tz.c_str(), author_tz.size());
    b2h.update(committer.c_str(), committer.size());
    b2h.update(committer_time.c_str(), committer_time.size());
    b2h.update(committer_tz.c_str(), committer_tz.size());
    b2h.update(log.c_str(), log.size());
    b2h.flush();
    return 0;
  }

  enum MessageType {
    MessageUNKNOWN = 0,
    MessageCOMMIT,
    MessageTREE,
  };

  MessageType type;
  std::vector<char> header;
  std::string parent;
  std::string author;
  std::string author_time;
  std::string author_tz;
  std::string committer;
  std::string committer_time;
  std::string committer_tz;
  std::string log;
  long long author_btime;
  long long committer_btime;

  static int parseTimestamp(std::string* packed, std::string* thetime,
                            std::string* thetz, long long* binarytime) {
    const char* s = packed->c_str();
    int n = 0;
    int limit = strlen(s);
    // The timestamp is after <email-id> and some whitespace.
    const char* token[] = {"<", ">", " "};
    for (size_t i = 0; i < sizeof(token)/sizeof(token[0]); i++) {
      n += strcspn(s + n, token[i]);
      n += strspn(s + n, token[i]);
      if (n >= limit) {
        break;
      }
    }
    if (n >= limit) {
      fprintf(stderr, "invalid timestamp - overran limit\n");
      return 1;
    }
    s += n;
    limit -= n;
    if (sscanf(s, "%lld%n", binarytime, &n) != 1) {
      fprintf(stderr, "invalid timestamp: \"%.*s\"\n", limit, s);
      return 1;
    }
    thetime->assign(s, s + n);
    thetz->assign(s + n);
    packed->erase(packed->begin() + (s - packed->c_str()), packed->end());
    return 0;
  }
};

class CommitReader {
public:
  CommitReader(const char* whoami) : whoami(whoami) {}

  int read_from(FILE* fin, CommitMessage* out) {
    size_t max = 1024;
    char* message = reinterpret_cast<char*>(malloc(max));
    if (!message) {
      fprintf(stderr, "%s: out of memory at %zuK\n", whoami, max / 1024);
      return 1;
    }

    char extra_room_sizer[16];
    size_t remain = max;
    // EXTRA_ROOM_PAD: padding to make this a commit.
    // strlen("commit ") + strlen("\0") + '\0' at the very end
    #define EXTRA_ROOM_PAD (7+1+1)
    int extra_room = EXTRA_ROOM_PAD + 4 /*strlen(sprintf("%s", max))*/;
    char* p = message;
    while (!feof(fin)) {
      size_t r = fread(p, 1, remain - extra_room, fin);
      if (r < remain && !feof(fin) && errno) {
        // This utility must be passed on stdin the
        // contents of the commit (completely unedited), like so:
        // git cat-file commit HEAD | $whoami
        fprintf(stderr, "Usage: %s $GIT_TOPLEVEL\n"
          "Failed to read stdin after %zu bytes: %d %s\n",
          whoami, max - remain, errno, strerror(errno));
        return 1;
      }
      p += r;
      remain -= r;
      if (remain <= (size_t) extra_room) {
        remain += max;
        message = reinterpret_cast<char*>(realloc(message, max *= 2));
        if (!message) {
          fprintf(stderr, "%s: out of memory at %zuK\n", whoami,
                  max / (2*1024));
          return 1;
        }
        p = message + (max - remain);
        extra_room = snprintf(extra_room_sizer, sizeof(extra_room_sizer), "%zu",
                              max);
        if (extra_room < 4) {
          fprintf(stderr, "%s: failed to size extra_room\n", whoami);
          return 1;
        }
        extra_room += EXTRA_ROOM_PAD;
      }
    }

    // Success. Clean up message.
    if (remain <= (size_t) extra_room) {
      fprintf(stderr, "%s: BUG: remain too small (want %d)\n", whoami,
              extra_room);
      return 1;
    }
    size_t len = max - remain;
    extra_room = snprintf(extra_room_sizer, sizeof(extra_room_sizer), "%zu",
                          len);
    if (extra_room < 1) {
      fprintf(stderr, "%s: failed to get final extra_room\n", whoami);
      return 1;
    }
    // Make room at the beginning for "commit %zu\0"
    max = max - remain + extra_room + EXTRA_ROOM_PAD;
    memmove(message + extra_room + 7 + 1, message, len);
    // Write "commit %zu\0"
    strncpy(message, "commit ", 7);
    strncpy(message + 7, extra_room_sizer, extra_room);
    message[extra_room + 7] = 0;
    message[max - 1] = 0;  // Add '\0' null terminator at very end of message.

    return out->set(message, max - 1);
  }

private:
  const char* const whoami;
};

int doGitCommit(size_t thId, Sha1Hash& sha, Blake2Hash& b2h,
                CommitMessage& noodle);
