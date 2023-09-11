# git-mine

Yes, even git repositories are a cryptocurrency now.

Another implementation, in rust (but also using OpenCL): https://github.com/not-an-aardvark/lucky-commit

# How to verify a commit hash

Assuming the commit is HEAD:

1. Get the first 10 digits of the (plain, ordinary) git hash:
   `WANT=$(git rev-parse HEAD | cut -c -10)`
1. `echo $WANT`
1. Make sure `b2sum` is installed. On debian it is in "coreutils".
1. Re-compute the git hash using `sha1sum`. Also do it with `b2sum`:
   ```
   MSG=$(git cat-file commit $WANT)
   for alg in sha1sum b2sum; do
     echo -n "$alg "
     ( echo -ne "commit $(( ${#MSG} + 1 ))\x00"; echo "${MSG}" ) | $alg | \
     grep $WANT
   done
   ```

`grep` should verify for you that b2sum spit out the $WANT digits just like
sha1sum. If there is no b2sum line, it means grep could not find $WANT digits
in the b2sum output:

### Example of a valid commit:
...<br/>
`done`<br/>
`sha1sum` **aabbccddee** 92dbc803c03f2f45cba7ddc5813106cf  -<br/>
`b2sum` 4f9997f02a6091ac98d7a47d1a72468dfcda926212e2c855b0600c7b8aa82737d8e082acfea6a95ecad42836881916129c508675f58509 **aabbccddee** 171a74a9  -

### Example of an invalid commit:
...<br/>
`done`<br/>
`sha1sum` **aabbccddee** 92dbc803c03f2f45cba7ddc5813106cf  -

(The above is missing a b2sum line with a highlighted section, if grep
can use of color.)

# How to sign your commit
First commit using whatever your normal tools are.

Then from a directory inside the git repo you want to sign (a submodule is
ok), run:

`git cat-file commit HEAD | git-mine`

Maybe a later version will change this to take the commit hash (or HEAD,
etc.) as the first parameter. `git-mine` can run
`git cat-file commit $HASH` as a subprocess and capture the output. To
Keep It Simple, right now `git-mine` just expects the raw commit to be
piped in.

## How to sign your commit using OpenCL

```
cd git-mine
make git-mine-ocl
cd /path/to/your/repo
git cat-file commit HEAD | ~/git-mine/git-mine-ocl
```
