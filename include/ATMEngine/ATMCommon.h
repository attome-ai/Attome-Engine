#define ATM_CONFIG_LOG 1
#define ATM_CONFIG_TEST 1
#include "SDL3/SDL.h"


#ifdef ATM_CONFIG_LOG
#define ATMLOG(...) SDL_Log(__VA_ARGS__)
#else
#define ATMLOG(...)
#endif // !ATM_NO_LOG
