#include <SDL3/SDL.h>
#include <SDL3/SDL_gpu.h>

#define ATM_CONFIG_LOG_ON
#define ATM_CONFIG_DEBUG_CHECK_ON


#ifdef ATM_CONFIG_LOG_ON
#define ATMLOG(x, ...) SDL_Log(x, ##__VA_ARGS__)
#else
#define ATMLOG(x, ...)
#endif // ATM_CONFIG_LOG_ON


#ifdef ATM_CONFIG_DEBUG_CHECK_ON

#define ATMLOGC(x,y, ...) if(x) {SDL_Log(y, ##__VA_ARGS__); abort();}
#define ATMLOGS(x,y, ...) if(x) {SDL_Log(y, ##__VA_ARGS__);}
#define ATMLOGE(x,y, ...) if(x) {SDL_Log(y, ##__VA_ARGS__);}

#else
#define ATMLOGC(x, ...)
#define ATMLOGS(x, ...)
#define ATMLOGE(x, ...) x

#endif // ATM_CONFIG_DEBUG_CHECK_ON
