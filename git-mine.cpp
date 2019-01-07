#include "hashapi.h"

#include <stdlib.h>
#include <unistd.h>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <set>
#include <thread>

/**
 * Hash difficulty stats:
 * - Summary: 4 bytes ~        300 MH
 *            5 bytes ~     76,800 MH
 *            6 bytes ~ 19,660,800 MH
 * volcano cb9a1bcf9b92820d42b0a2e2a1c4f4d852e1dc87
 *  3 byte match 4539e6     0.0353 * 50.119K =   1.7962 MHash
 *  4 byte match ? (bug)    7.44   * 50.119K = 372.89 MHash
 *     atime:1491326952
 *     ctime:1541450688
 *  5 byte match 698147a34f 277 * 50.119K =  13883    MHash
 *     atime:1528115780
 *     ctime:1541450959
 *
 * volcano 17573e7ba0b3eb820793a2adf53da2ffe01b000d
 *  3 byte match a045eb     0.034 * 51M =   1.734 MHash
 *  4 byte match f8c4d335   0.534 * 51M =  27.234 MHash
 *     atime:1537683579
 *     ctime:1541871978
 *  5 byte match 15e5be397a 766 * 51M =  39066    MHash
 *     atime:
 *     ctime:
 *
 * volcano 56a8122de10e7bc424358b550033655afbd12fb8
 *  3 byte match (happened too fast)
 *  4 byte match c5e49e99   0.348 * 51M =  27.234 MHash
 *     atime:1529577996
 *     ctime:1542130496
 *  5 byte match 1492f859a6  11 * 51M =    561    MHash
 *     atime:1507383631
 *     ctime:1542130614
 *
 * subninja 68d1800069d0d0f098d151560a5c62049113da1f
 *  3 byte match 11901b     2.40 * 1.80M =     4.320 MHash
 *  4 byte match 1dcf3e20   9.77 * 1.80M =    17.586 MHash
 *     atime:1539471984
 *     ctime:1541188269
 *  5 byte match 61a46cc192 21860 * 1.8M = 40784     MHash
 *     atime:1540342039
 *     ctime:1541210550
 *
 * git-mine 56ec509b2401d2bdea4627c44f51d0ce026ebcd6
 *  5 byte match 73f02fd029                61727     MHash
 *
 * git-mine 94c19337b5027a9a74b0db4bc7bcd84a72b2afe4
 *  5 byte match f443ef67dd                15917     MHash
 *     atime=1544904115  ctime=1545005705
 *
 * subninja 75fc7c9e3e1d2ddee99729adb3fda737960db04e
 *  6 byte match 33056ea186f6         15,714,598     MHash
 *    atime=1550405995  ctime=1551400964
 */

class MineBoss {
public:
  CommitMessage orig;

  void start() {
    size_t nCPU = 0;
    FILE* cpuinfo = fopen("/proc/cpuinfo", "r");
    if (!cpuinfo) {
      fprintf(stderr, "Failed to open /proc/cpuinfo: %d %s\n", errno, strerror(errno));
      return;
    }
    char buf[256];
    while (!feof(cpuinfo)) {
      if (!fgets(buf, sizeof(buf), cpuinfo)) {
        if (feof(cpuinfo)) break;
        fprintf(stderr, "Read /proc/cpuinfo failed: %d %s\n", errno,
                strerror(errno));
        return;
      }
      buf[strcspn(buf, "\r\n")] = 0;
      if (!strncmp(buf, "processor", strlen("processor"))) {
        int n = strcspn(buf, ":");
        if (n < (int)strlen(buf)) {
          size_t v;
          int end;
          if (sscanf(buf + n + 1, "%zu%n", &v, &end) == 1 &&
              (int)strlen(buf + n + 1) == end) {
            if (v + 1 > nCPU) {
              nCPU = v + 1;
            }
          }
        }
      }
    }
    fclose(cpuinfo);
    if (pool.size()) {
      fprintf(stderr, "pool not empty - already started?\n");
      return;
    }
    if (atime_hint < orig.atime()) {
      if (atime_hint) {
        fprintf(stderr, "invalid atime_hint %lld (must be at least %lld)\n",
                atime_hint, orig.atime());
      }
      atime_hint = orig.atime();
    }
    if (ctime_hint < orig.ctime()) {
      if (ctime_hint) {
        fprintf(stderr, "invalid ctime_hint %lld (must be at least %lld)\n",
                ctime_hint, orig.ctime());
      }
      ctime_hint = orig.ctime();
    }

    // Lock bossMutex while adding threads to pool.
    std::unique_lock<std::mutex> lock(bossMutex);
    for (size_t i = 0; i < nCPU; i++) {
      pool.emplace(pool.begin(), new ThreadLocal(this, i, nCPU));
    }
    start_t = Clock::now();
  }

  void dumpMatchAt(size_t wantBest) {
    CommitMessage noodle(orig);
    Sha1Hash sha;
    Blake2Hash b2h;
    for (size_t i = 0; i < pool.size(); i++) {
      if (pool.at(i)->best >= wantBest) {
        fprintf(stderr, "Thread %zu says:\n", i);
        noodle.set_atime(pool.at(i)->best_atime);
        noodle.set_ctime(pool.at(i)->best_ctime);
        noodle.hash(sha, b2h);
        char buf[1024];
        if (sha.dump(buf, sizeof(buf))) {
          fprintf(stderr, "sha.dump failed\n");
          return;
        }
        fprintf(stderr, "sha1:   %s\n", buf);
        if (b2h.dump(buf, sizeof(buf))) {
          fprintf(stderr, "b2h.dump failed\n");
          return;
        }
        fprintf(stderr, "blake2: %s\n", buf);
        fprintf(stderr, "author time=%lld\n", pool.at(i)->best_atime);
        fprintf(stderr, "committer  =%lld\n", pool.at(i)->best_ctime);
        return;
      }
    }
    fprintf(stderr, "No best of %zu found.\n", wantBest);
  }

  void commitMatch() {
    for (size_t i = 0; i < pool.size(); i++) {
      if (pool.at(i)->matchFound) {
        commitMatchWith(*pool.at(i));
        return;
      }
    }
    fprintf(stderr, "A thread set searchDone but didn't set matchFound.\n");
  }

  enum {
    terminateAt = 5,
    COUNT_DIVISOR = 16*1024,
  };

  long long atime_hint;
  long long ctime_hint;

  // printProgressAt1Hz returns 1 if all threads quit or if they should.
  int printProgressAt1Hz() {
    long long total_work = (ctime_hint - atime_hint) / COUNT_DIVISOR;
    auto t0 = Clock::now();
    // lock is needed for cond.wait_until.
    std::unique_lock<std::mutex> lock(bossMutex);
    for (auto t1 = t0 + std::chrono::seconds(1);;) {
      long long total = 0;
      cond.wait_until(lock, t1);
      if (searchDone) {
        return 1;
      }

      // Check threads to see who is still running.
      int r = 1;
      size_t best = 0;
      for (size_t i = 0; i < pool.size(); i++) {
        if (pool.at(i)->best > best) {
          best = pool.at(i)->best;
        }
        if (pool.at(i)->go) {
          r = 0;  // At least 1 thread is still running.
          total += pool.at(i)->count;
        } else if (pool.at(i)->bossSaidGo) {
          pool.at(i)->bossSaidGo = false;
          //fprintf(stderr, "Thread %zu quit.\n", i);
          // In stop, do join(). Not here.
        }
      }
      if (r) return r;
      t0 = Clock::now();
      if (t0 < t1) continue;

      // Report progress if a full second passed.
      std::chrono::duration<float> elapsed_sec = t0 - start_t;
      fprintf(stderr, "%4.1fs progress: %7.2f%%   best:%zu  100%%=%.2f MHash\n",
              elapsed_sec.count(),
              100.0f*float(total)/float(total_work), best,
              float(total_work) * COUNT_DIVISOR / 1e6);
      if (best > last_best) {
        last_best = best;
        dumpMatchAt(best);
      }
      break;
    }
    return 0;
  }

  void stop() {
    { // Signal all threads to quit.
      std::unique_lock<std::mutex> lock(bossMutex);
      stopRequested = true;
      cond.notify_all();
    }
    // Wait for threads to quit.
    for (size_t patience = 5; ; patience--) {
      if (!patience) {
        fprintf(stderr,
                "Out of patience! Use ctrl+C to kill me.\n"
                "Threads seem to be deadlocked.\n");
        break;
      }
      if (patience != 5) {
        fprintf(stderr, "stop: wait %zus:", patience);
      }
      if (printProgressAt1Hz()) {
        break;
      }
    }
    // Print something if thread cleanup hangs process.
    if (0) fprintf(stderr, "Cleaning up threads.\n");
    // Join (reap) threads in ThreadLocal::~ThreadLocal().
    pool.clear();
  }

  bool getSearchDone() {
    std::unique_lock<std::mutex> lock(bossMutex);
    return searchDone;
  }

private:
  typedef std::chrono::steady_clock Clock;
  Clock::time_point start_t;
  size_t last_best{0};

  struct ThreadLocal {
    ThreadLocal(MineBoss* parent_, size_t id, size_t idMax)
      : parent(parent_)
      , th(&ThreadLocal::worker, this)
      , noodle(parent_->orig)
      , id(id)
      , idMax(idMax) {}

    ~ThreadLocal() {
      th.join();
    }
    MineBoss* parent;
    std::thread th;
    long long commit_delta{0};
    CommitMessage noodle;

    bool go{true};
    bool bossSaidGo{true};
    size_t id;
    size_t idMax;
    size_t matchFound{0};
    volatile size_t best{0};
    long long best_atime{0};
    long long best_ctime{0};
    Sha1Hash sha;
    Blake2Hash b2h;

    long long count{0};

    void worker() {
      doWork();
      std::unique_lock<std::mutex> lock(parent->bossMutex);
      go = false;
      parent->cond.notify_all();
    }

    // search returns 1 if a match was found or there is some other reason
    // to abort the search.
    int search(long long atime) {
      float total_work = float(noodle.ctime() - atime);
      // Use floats to trade off accuracy for avoiding overflow.
      long long my_work_start =
          (long long) ((float(id) * total_work) / float(idMax)) + atime;
      long long my_work_end =
          (long long) ((float(id + 1) * total_work) / float(idMax)) + atime;
      int my_count = 1;
      for (long long t = my_work_start; t < my_work_end; t++, my_count++) {
        if ((my_count & (COUNT_DIVISOR - 1)) == 0) {
          my_count = 0;
          std::unique_lock<std::mutex> lock(parent->bossMutex);
          count++;
          if (parent->stopRequested) {
            return 1;
          }
        }
        noodle.set_atime(t);
        noodle.hash(sha, b2h);
        size_t matchlen = 0;
        int match = b2h.instr(sha.result, sizeof(sha.result), &matchlen);
        if (match != -1) {
          if (matchlen > best) {
            best = matchlen;
            best_atime = t;
            best_ctime = noodle.ctime();
          }
          if (matchlen >= terminateAt) {
            // Signal that a match was found.
            std::unique_lock<std::mutex> lock(parent->bossMutex);
            matchFound = 1;
            parent->searchDone = true;
            parent->cond.notify_all();
            return 1;
          }
        }
      }
      return 0;
    }

    void doWork() {
      noodle.set_ctime(parent->ctime_hint);
      for (;;) {
        if (search(parent->atime_hint)) {
          return;
        }

        // Increment ctime and try again.
        commit_delta++;
        noodle.set_ctime(noodle.ctime() + 1);
      }
    }
  };

  void commitMatchWith(ThreadLocal& th) {
    doGitCommit(th.id, th.sha, th.b2h, th.noodle);
  }

  // bossMutex and cond guard the rest of the members of this class.
  std::mutex bossMutex;
  std::condition_variable cond;
  bool stopRequested{false};
  bool searchDone{false};

  std::vector<std::shared_ptr<ThreadLocal>> pool;
};

int main(int argc, char ** argv) {
  if (argc != 3 && argc != 1) {
    // This utility must be called from a post-commit hook
    // with $GIT_TOPLEVEL as the only argument.
    fprintf(stderr, "Usage: %s [ atime_hint ctime_hint ]\n",
            argv[0]);
    return 1;
  }
  long long atime_hint = 0;
  long long ctime_hint = 0;
  if (argc == 3) {
    int n;
    if (sscanf(argv[1], "%lld%n", &atime_hint, &n) != 1 ||
        (int)strlen(argv[1]) != n) {
      fprintf(stderr, "Invalid atime_hint: \"%s\"\n", argv[2]);
      return 1;
    }
    if (sscanf(argv[2], "%lld%n", &ctime_hint, &n) != 1 ||
        (int)strlen(argv[2]) != n) {
      fprintf(stderr, "Invalid ctime_hint: \"%s\"\n", argv[2]);
      return 1;
    }
  }

  MineBoss boss;
  boss.atime_hint = atime_hint;
  boss.ctime_hint = ctime_hint;
  {
    FILE* f = stdin;
    CommitReader reader(argv[0]);
    if (reader.read_from(f, &boss.orig)) {
      return 1;
    }
    Sha1Hash sha;
    Blake2Hash b2h;
    if (boss.orig.hash(sha, b2h)) {
      return 1;
    }
    char shabuf[1024];
    if (sha.dump(shabuf, sizeof(shabuf))) {
      return 1;
    }
    fprintf(stderr, "Signing commit: %s\n", shabuf);
  }

  boss.start();
  for (size_t i = 90;;) {
    if (boss.printProgressAt1Hz()) break;
    i--;
    i++;
    if (!i) {
      fprintf(stderr, "\ntimed out\n");
      break;
    }
  }
  if (boss.getSearchDone()) {
    boss.commitMatch();
  }
  boss.stop();
  return 0;
}
