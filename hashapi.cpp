#include "hashapi.h"

#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/epoll.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>

int hexdump(char* buf, size_t buflen, uint8_t* bytes, size_t byteslen) {
  char* p = buf;
  size_t len = buflen;
  for (size_t i = 0; len > 0 && i < byteslen; i++) {
    size_t r = snprintf(p, len, "%02x", bytes[i]);
    if (r < len) {
      p += r;
      len -= r;
    } else {
      fprintf(stderr, "overflow on byte %zu\n", i);
      return 1;
    }
  }
  if (len) {
    *p = 0;
  }
  return 0;
}

static void handle_SIGPIPE(int) {
  fprintf(stderr, "received SIGPIPE\n");
}

int doGitCommit(size_t thId, Sha1Hash& sha, Blake2Hash& b2h,
                CommitMessage& noodle) {
  fprintf(stderr, "Th%zu author time = %lld\n", thId, noodle.atime());
  fprintf(stderr, "Th%zu committer   = %lld\n", thId, noodle.ctime());
  fprintf(stderr, "Th%zu sha1: ", thId);
  char buf[1024];
  if (sha.dump(buf, sizeof(buf))) {
    fprintf(stderr, "sha.dump failed\n");
    return 1;
  }
  std::string wantOutput = buf;
  size_t matchlen = 0;
  int match = b2h.instr(sha.result, sizeof(sha.result), &matchlen);
  if (match == -1) {
    fprintf(stderr, "unable to match sha1-b2h.\n");
    return 1;
  }
  unsigned part = matchlen*2 + 2;
  if (part > strlen(buf)) part = strlen(buf);
  fprintf(stderr, "%.*s%s\n", part, buf, (part != strlen(buf)) ? "..." : "");
  fprintf(stderr, "Th%zu blake2: ", thId);
  if (b2h.dump(buf, sizeof(buf))) {
    fprintf(stderr, "b2h.dump failed\n");
    return 1;
  }
  if (match != 0) {
    fprintf(stderr, "%.*s ", match*2, buf);
  }
  fprintf(stderr, "%.*s", int(matchlen*2), &buf[match*2]);
  if ((match + matchlen)*2 < strlen(buf)) {
    fprintf(stderr, " %s", &buf[(match + matchlen)*2]);
  }
  fprintf(stderr, "\n");

  // Create environment for subprocess.
  std::string author = noodle.author;
  size_t pos = author.find("<");
  if (pos == std::string::npos) {
    fprintf(stderr, "Failed to parse author <: %s\n", author.c_str());
    return 1;
  }
  std::string author_email = author.substr(pos + 1);
  if (author_email.find(">") == std::string::npos) {
    fprintf(stderr, "Failed to parse author >: %s\n", author.c_str());
    return 1;
  }
  author = author.substr(0, pos);
  author_email = author_email.substr(0, author_email.find(">"));
  if (author.substr(0, 7) != "author ") {
    fprintf(stderr, "Failed to parse author: %s\n", author.c_str());
    return 1;
  }
  author = author.substr(7);
  std::string author_date = noodle.author_time + noodle.author_tz;
  pos = author_date.find_last_not_of("\r\n ");
  if (pos == std::string::npos) {
    fprintf(stderr, "Failed to trim author_date: %s\n", author_date.c_str());
    return 1;
  }
  author_date = author_date.substr(0, pos + 1);

  std::string com = noodle.committer;
  pos = com.find("<");
  if (pos == std::string::npos) {
    fprintf(stderr, "Failed to parse committer <: %s\n", com.c_str());
    return 1;
  }
  std::string com_email = com.substr(pos + 1);
  if (com.find(">") == std::string::npos) {
    fprintf(stderr, "Failed to parse committer >: %s\n", com.c_str());
    return 1;
  }
  com = com.substr(0, pos);
  com_email = com_email.substr(0, com_email.find(">"));
  if (com.substr(0, 10) != "committer ") {
    fprintf(stderr, "Failed to parse committer: %s\n", com.c_str());
    return 1;
  }
  com = com.substr(10);
  std::string com_date = noodle.committer_time + noodle.committer_tz;
  pos = com_date.find_last_not_of("\r\n ");
  if (pos == std::string::npos) {
    fprintf(stderr, "Failed to trim com_date: %s\n", com_date.c_str());
    return 1;
  }
  com_date = com_date.substr(0, pos + 1);

  std::string parent = noodle.parent;
  if (parent.substr(0, 7) != "parent ") {
    fprintf(stderr, "Failed to parse parent: %s\n", parent.c_str());
    return 1;
  }
  parent = parent.substr(7);
  pos = parent.find_last_not_of("\r\n ");
  if (pos == std::string::npos) {
    fprintf(stderr, "Failed to trim parent: %s\n", parent.c_str());
    return 1;
  }
  parent = parent.substr(0, pos + 1);

  std::string tree;
  for (pos = 0; pos < noodle.header.size(); pos++) {
    if (noodle.header.at(pos) == 0) {
      pos++;
      if (pos >= noodle.header.size()) {
        fprintf(stderr, "Failed to find tree in header\n");
        return 1;
      }
      tree.assign(noodle.header.data() + pos,
                  noodle.header.size() - pos);
      if (tree.substr(0, 5) != "tree ") {
        fprintf(stderr, "Failed to parse tree: %s\n", tree.c_str());
        return 1;
      }
      tree = tree.substr(5);
      pos = tree.find_last_not_of("\r\n ");
      if (pos == std::string::npos) {
        fprintf(stderr, "Failed to trim tree: %s\n", tree.c_str());
        return 1;
      }
      tree = tree.substr(0, pos + 1);
      break;
    }
  }
  if (pos >= noodle.header.size()) {
    fprintf(stderr, "Failed to parse header\n");
    return 1;
  }

  std::vector<char *> git_env;
  author = "GIT_AUTHOR_NAME=" + author;
  git_env.push_back((char*)author.c_str());
  author_email = "GIT_AUTHOR_EMAIL=" + author_email;
  git_env.push_back((char*)author_email.c_str());
  author_date = "GIT_AUTHOR_DATE=" + author_date;
  git_env.push_back((char*)author_date.c_str());

  com = "GIT_COMMITTER_NAME=" + com;
  git_env.push_back((char*)com.c_str());
  com_email = "GIT_COMMITTER_EMAIL=" + com_email;
  git_env.push_back((char*)com_email.c_str());
  com_date = "GIT_COMMITTER_DATE=" + com_date;
  git_env.push_back((char*)com_date.c_str());

  if (0) {
    fprintf(stderr, "tree \"%s\"\n", tree.c_str());
    fprintf(stderr, "parent \"%s\"\n", parent.c_str());
    fprintf(stderr, "\"%s\" \"%s\" \"%s\"\n",
            author.c_str(), author_email.c_str(), author_date.c_str());
    fprintf(stderr, "\"%s\" \"%s\" \"%s\"\n",
            com.c_str(), com_email.c_str(), com_date.c_str());
  }

  char ** p = environ;
  while (*p) {
    git_env.push_back(*p);
    p++;
  }
  git_env.push_back(NULL);  // Terminate the environment linked list.

  int pipe1[2];
  if (pipe(pipe1) < 0) {
    fprintf(stderr, "pipe(pipe1): %d %s\n", errno, strerror(errno));
    return 1;
  }
  int pipe2[2];
  if (pipe(pipe2) < 0) {
    fprintf(stderr, "pipe(pipe2): %d %s\n", errno, strerror(errno));
    return 1;
  }

  signal(SIGPIPE, handle_SIGPIPE);
  char* git_argv[] = {
    (char*)"git", (char*)"commit-tree", (char*)tree.c_str(),
    (char*)"-p", (char*)parent.c_str(),
    NULL,  // Terminate git_argv with a NULL.
  };
  int pid = fork();
  if (pid < 0) {
    fprintf(stderr, "fork(git commit-tree) failed: %d %s\n", errno,
            strerror(errno));
    return 1;
  }
  if (!pid) {
    // In child process: exec(git commit-tree).
    close(pipe1[0]);  // Close "read end"
    dup2(pipe1[1], 1);  // Redirect stdout to pipe1[1] ("write end")
    dup2(pipe1[1], 2);  // Redirect stderr to pipe1[1] ("write end")
    close(pipe1[1]);  // Close "write end" since it is now stdout/stderr.
    close(pipe2[1]);  // Close "write end"
    dup2(pipe2[0], 0);  // Redirect pipe2[0] ("read end") to stdin
    close(pipe2[0]);  // Close "read end"
    execvpe("git", git_argv, git_env.data());
    fprintf(stderr, "execvpe(git commit-tree) failed: %d %s\n", errno,
            strerror(errno));
    fflush(stderr);
    _exit(1);
  }
  close(pipe1[1]);  // Close "write end" that will not be used in parent
  close(pipe2[0]);  // Close "read end" that will not be used in parent

  // Wait for child (and read in what it says).
  std::string output;
  int epollfd = epoll_create1(0 /*flags*/);
  if (epollfd < 0) {
    fprintf(stderr, "epoll_create1 failed: %d %s\n", errno, strerror(errno));
    return 1;
  }

  std::vector<epoll_event> events;
  events.emplace_back();
  events.back().events = EPOLLIN;
  events.back().data.fd = pipe1[0];
  if (epoll_ctl(epollfd, EPOLL_CTL_ADD, pipe1[0], &events.back())) {
    fprintf(stderr, "EPOLL_CTL_ADD pipe1 failed: %d %s\n", errno,
            strerror(errno));
    return 1;
  }
  events.emplace_back();
  events.back().events = EPOLLOUT;
  events.back().data.fd = pipe2[1];
  if (epoll_ctl(epollfd, EPOLL_CTL_ADD, pipe2[1], &events.back())) {
    fprintf(stderr, "EPOLL_CTL_ADD pipe2 failed: %d %s\n", errno,
            strerror(errno));
    return 1;
  }

  int go = 1;
  size_t wrote_count = 0;
  while (go) {
    int r = epoll_wait(epollfd, events.data(), events.size(),
                        1000 /*milliseconds*/);
    if (r < 0) {
      fprintf(stderr, "epoll_wait failed: %d %s\n", errno, strerror(errno));
      return 1;
    } else if ((size_t)r > events.size()) {
      fprintf(stderr, "epoll_wait referred to %d, only %zu exist\n", r,
              events.size());
      return 1;
    } else if (!r) {
      fprintf(stderr, "Waiting for pid %d: git commit-tree...\n", pid);
      continue;
    }
    for (size_t i = 0; i < (size_t)r; i++) {
      if (events.at(i).data.fd == pipe1[0]) {
        if (events.at(i).events & EPOLLHUP) {
          go = 0;
          continue;
        }
        char buf[256];
        ssize_t nr = read(pipe1[0], buf, sizeof(buf));
        if (nr < 0) {
          fprintf(stderr, "read(pipe1[0]) failed: %d %s\n", errno,
                  strerror(errno));
          return 1;
        }
        if (nr == 0) {
          // 0 bytes indicates pipe1[0] was closed.
          close(pipe1[0]);
          go = 0;
          continue;
        }
        std::string s(buf, nr);
        output += s;
      } else if (events.at(i).data.fd == pipe2[1]) {
        // Stream commit message to child process.
        ssize_t wrlen = write(pipe2[1], noodle.log.c_str() + wrote_count,
                              noodle.log.size() - wrote_count);
        if (wrlen != ssize_t(noodle.log.size() - wrote_count)) {
          fprintf(stderr, "Write failed: %d %s (wrote %lld, want %lld)\n",
                  errno, strerror(errno), (long long) wrlen,
                  (long long) noodle.log.size() - wrote_count);
          close(pipe2[1]);
          return 1;
        }
        wrote_count += wrlen;
        if (wrote_count >= noodle.log.size()) {
          // All of noodle.log was sent. Close pipe2[1] so child knows.
          if (epoll_ctl(epollfd, EPOLL_CTL_DEL, pipe2[1], &events.at(i))) {
            fprintf(stderr, "EPOLL_CTL_DEL pipe2 failed: %d %s\n", errno,
                    strerror(errno));
            return 1;
          }
          close(pipe2[1]);
        }
      }
    }
  }
  int code;
  int waitres = waitpid(pid, &code, 0);
  if (waitres != pid) {
    fprintf(stderr, "waitpid failed: got %d, want pid %d. %d %s\n",
            waitres, pid, errno, strerror(errno));
    return 1;
  }
  // wantOutput has the hexdump of sha1 from the top of the function.
  wantOutput += "\n";
  if (code || output != wantOutput) {
    fprintf(stderr, "git commit-tree exited with code %d:\n", code);
    fprintf(stderr, "%s", output.c_str());
    return 1;
  }
  fprintf(stderr, "repo updated.\n# hint: %s %s",
          "git checkout master; git reset --hard", wantOutput.c_str());
  return 0;
}
