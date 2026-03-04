/* iconv stub for Android NDK — provides the symbols SDL3 expects when
   building with iconv support but Android does not ship libiconv. */
#include <errno.h>
#include <stddef.h>


typedef void *iconv_t;

iconv_t iconv_open(const char *tocode, const char *fromcode) {
  (void)tocode;
  (void)fromcode;
  errno = EINVAL;
  return (iconv_t)-1;
}

int iconv_close(iconv_t cd) {
  (void)cd;
  return 0;
}

size_t iconv(iconv_t cd, char **inbuf, size_t *inbytesleft, char **outbuf,
             size_t *outbytesleft) {
  (void)cd;
  (void)inbuf;
  (void)inbytesleft;
  (void)outbuf;
  (void)outbytesleft;
  errno = EILSEQ;
  return (size_t)-1;
}
