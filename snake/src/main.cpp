#include "ATMEngine.h"
#include <SDL3/SDL.h>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <string>

// ── constants
// ──────────────────────────────────────────────────────────────────
static constexpr int WIN_W = 1000;
static constexpr int WIN_H = 800;
static constexpr int GRID_COLS = 25;
static constexpr int GRID_ROWS = 20;
static constexpr int CELL = 32;
static constexpr int ORIGIN_X = (WIN_W - GRID_COLS * CELL) / 2;
static constexpr int ORIGIN_Y = (WIN_H - GRID_ROWS * CELL) / 2 + 20;
static constexpr float STEP_SEC = 0.15f;

// ── colors
// ─────────────────────────────────────────────────────────────────────
static const SDL_Color C_BG = {10, 10, 20, 255};
static const SDL_Color C_GRID = {25, 25, 45, 255};
static const SDL_Color C_HEAD = {232, 72, 153, 255};
static const SDL_Color C_BODY = {168, 40, 110, 255};
static const SDL_Color C_FOOD = {20, 184, 166, 255};
static const SDL_Color C_BORDER = {60, 60, 90, 255};
static const SDL_Color C_TEXT = {240, 240, 255, 255};
static const SDL_Color C_OVERLAY = {255, 80, 80, 255};

// ── types
// ──────────────────────────────────────────────────────────────────────
struct Vec2 {
  int x, y;
};
enum class Dir { UP, DOWN, LEFT, RIGHT };

struct GameState {
  std::deque<Vec2> snake;
  Dir dir = Dir::RIGHT;
  Dir nextDir = Dir::RIGHT;
  Vec2 food = {10, 10};
  int score = 0;
  int highScore = 0;
  bool gameOver = false;
  bool started = false;
  float stepTimer = 0.0f;
  float glowTime = 0.0f;
};

// ── pixel font (5×7 each digit 0-9, stored as 5 column bytes)
// ───────────────── Each byte = 7-bit column, MSB = top pixel
static const Uint8 DIGITS[10][5] = {
    {0x7C, 0x8A, 0x92, 0xA2, 0x7C}, // 0
    {0x00, 0x42, 0xFE, 0x02, 0x00}, // 1
    {0x46, 0x8A, 0x92, 0x92, 0x62}, // 2
    {0x44, 0x82, 0x92, 0x92, 0x6C}, // 3
    {0x18, 0x28, 0x48, 0xFE, 0x08}, // 4
    {0xE4, 0xA2, 0xA2, 0xA2, 0x9C}, // 5
    {0x3C, 0x52, 0x92, 0x92, 0x0C}, // 6
    {0x80, 0x8E, 0x90, 0xA0, 0xC0}, // 7
    {0x6C, 0x92, 0x92, 0x92, 0x6C}, // 8
    {0x60, 0x92, 0x92, 0x94, 0x78}, // 9
};
static constexpr int PX = 3; // pixel scale

static void drawDigit(SDL_Renderer *r, int d, int ox, int oy, SDL_Color c) {
  SDL_SetRenderDrawColor(r, c.r, c.g, c.b, c.a);
  for (int col = 0; col < 5; ++col) {
    Uint8 bits = DIGITS[d][col];
    for (int row = 0; row < 7; ++row) {
      if (bits & (0x80 >> row)) {
        SDL_FRect px = {(float)(ox + col * PX), (float)(oy + row * PX),
                        (float)PX, (float)PX};
        SDL_RenderFillRect(r, &px);
      }
    }
  }
}

static void drawNumber(SDL_Renderer *r, int n, int x, int y, SDL_Color c) {
  if (n < 0)
    n = 0;
  std::string s = std::to_string(n);
  for (char ch : s) {
    drawDigit(r, ch - '0', x, y, c);
    x += (5 + 1) * PX; // 5 cols + 1 gap
  }
}

// Width of a number string in pixels
static int numberWidth(int n) {
  int digits = (n == 0) ? 1 : 0;
  int tmp = n;
  while (tmp > 0) {
    ++digits;
    tmp /= 10;
  }
  return digits * (5 + 1) * PX;
}

// ── helpers
// ────────────────────────────────────────────────────────────────────
static void setColor(SDL_Renderer *r, SDL_Color c) {
  SDL_SetRenderDrawColor(r, c.r, c.g, c.b, c.a);
}

static void fillCell(SDL_Renderer *r, int gx, int gy, SDL_Color c,
                     int inset = 1) {
  SDL_FRect rect = {(float)(ORIGIN_X + gx * CELL + inset),
                    (float)(ORIGIN_Y + gy * CELL + inset),
                    (float)(CELL - inset * 2), (float)(CELL - inset * 2)};
  setColor(r, c);
  SDL_RenderFillRect(r, &rect);
}

static void drawRoundedCell(SDL_Renderer *r, int gx, int gy, SDL_Color c) {
  fillCell(r, gx, gy,
           {(Uint8)(c.r / 2), (Uint8)(c.g / 2), (Uint8)(c.b / 2), 255}, 0);
  fillCell(r, gx, gy, c, 2);
}

static void spawnFood(GameState &gs) {
  do {
    gs.food.x = rand() % GRID_COLS;
    gs.food.y = rand() % GRID_ROWS;
  } while ([&] {
    for (auto &s : gs.snake)
      if (s.x == gs.food.x && s.y == gs.food.y)
        return true;
    return false;
  }());
}

static bool collidesSelf(const std::deque<Vec2> &snake, Vec2 head) {
  for (size_t i = 1; i < snake.size(); ++i)
    if (snake[i].x == head.x && snake[i].y == head.y)
      return true;
  return false;
}

static void resetGame(GameState &gs) {
  gs.snake.clear();
  gs.snake.push_back({12, 10});
  gs.snake.push_back({11, 10});
  gs.snake.push_back({10, 10});
  gs.dir = Dir::RIGHT;
  gs.nextDir = Dir::RIGHT;
  gs.score = 0;
  gs.gameOver = false;
  gs.started = false;
  gs.stepTimer = 0.0f;
  spawnFood(gs);
}

// ── draw grid
// ──────────────────────────────────────────────────────────────────
static void drawGrid(SDL_Renderer *r) {
  setColor(r, C_GRID);
  for (int x = 0; x <= GRID_COLS; ++x) {
    float px = (float)(ORIGIN_X + x * CELL);
    SDL_RenderLine(r, px, (float)ORIGIN_Y, px,
                   (float)(ORIGIN_Y + GRID_ROWS * CELL));
  }
  for (int y = 0; y <= GRID_ROWS; ++y) {
    float py = (float)(ORIGIN_Y + y * CELL);
    SDL_RenderLine(r, (float)ORIGIN_X, py, (float)(ORIGIN_X + GRID_COLS * CELL),
                   py);
  }
  SDL_FRect border = {(float)(ORIGIN_X - 2), (float)(ORIGIN_Y - 2),
                      (float)(GRID_COLS * CELL + 4),
                      (float)(GRID_ROWS * CELL + 4)};
  setColor(r, C_BORDER);
  SDL_RenderRect(r, &border);
}

// ── draw a thick horizontal divider line of dots/dashes
// ───────────────────────
static void drawLabel(SDL_Renderer *r, const char *text, int x, int y,
                      SDL_Color c) {
  // Draw each character as a tiny block — fall back to pixel digits for numbers
  // For letters we just draw a labeled rectangle (simple indicator)
  SDL_SetRenderDrawColor(r, c.r, c.g, c.b, 80);
  SDL_FRect bg = {(float)(x - 2), (float)(y - 2), 80, (float)(7 * PX + 4)};
  SDL_RenderFillRect(r, &bg);
  // Draw the text character by character using digit renderer for digits
  int cx = x;
  for (const char *p = text; *p; ++p) {
    if (*p >= '0' && *p <= '9') {
      drawDigit(r, *p - '0', cx, y, c);
      cx += (5 + 1) * PX;
    } else {
      cx += (2) * PX; // space for non-digit chars
    }
  }
}

// ── entry
// ──────────────────────────────────────────────────────────────────────
int main(int argc, char *argv[]) {
  srand((unsigned)time(nullptr));

  Engine *engine = engine_create(WIN_W, WIN_H, 10000, 10000, 64);
  if (!engine) {
    SDL_Log("Engine init failed");
    return 1;
  }
  SDL_SetWindowTitle(engine->window, "Snake - Attome Engine");

  GameState gs;
  resetGame(gs);

  bool running = true;
  float dt = 0.016f;

  while (running) {
    Uint64 t0 = SDL_GetTicks();

    SDL_Event ev;
    while (SDL_PollEvent(&ev)) {
      if (ev.type == SDL_EVENT_QUIT)
        running = false;
      if (ev.type == SDL_EVENT_KEY_DOWN) {
        auto k = ev.key.key;
        if (k == SDLK_ESCAPE)
          running = false;
        if (!gs.started && (k == SDLK_SPACE || k == SDLK_RETURN))
          gs.started = true;
        if (gs.gameOver && (k == SDLK_SPACE || k == SDLK_RETURN)) {
          if (gs.score > gs.highScore)
            gs.highScore = gs.score;
          resetGame(gs);
          gs.started = true;
        }
        if (gs.started && !gs.gameOver) {
          if ((k == SDLK_UP || k == SDLK_W) && gs.dir != Dir::DOWN)
            gs.nextDir = Dir::UP;
          if ((k == SDLK_DOWN || k == SDLK_S) && gs.dir != Dir::UP)
            gs.nextDir = Dir::DOWN;
          if ((k == SDLK_LEFT || k == SDLK_A) && gs.dir != Dir::RIGHT)
            gs.nextDir = Dir::LEFT;
          if ((k == SDLK_RIGHT || k == SDLK_D) && gs.dir != Dir::LEFT)
            gs.nextDir = Dir::RIGHT;
        }
      }
    }

    // update
    gs.glowTime += dt;
    float speed = 1.0f + gs.score * 0.01f;
    if (gs.started && !gs.gameOver) {
      gs.stepTimer += dt * speed;
      if (gs.stepTimer >= STEP_SEC) {
        gs.stepTimer -= STEP_SEC;
        gs.dir = gs.nextDir;
        Vec2 head = gs.snake.front();
        if (gs.dir == Dir::UP)
          head.y--;
        if (gs.dir == Dir::DOWN)
          head.y++;
        if (gs.dir == Dir::LEFT)
          head.x--;
        if (gs.dir == Dir::RIGHT)
          head.x++;
        if (head.x < 0 || head.x >= GRID_COLS || head.y < 0 ||
            head.y >= GRID_ROWS || collidesSelf(gs.snake, head)) {
          gs.gameOver = true;
        } else {
          gs.snake.push_front(head);
          if (head.x == gs.food.x && head.y == gs.food.y) {
            gs.score += 10;
            spawnFood(gs);
          } else {
            gs.snake.pop_back();
          }
        }
      }
    }

    engine_update(engine);

    // render
    SDL_Renderer *r = engine->renderer;
    setColor(r, C_BG);
    SDL_RenderClear(r);

    drawGrid(r);

    // food
    {
      float pulse = 0.7f + 0.3f * SDL_sinf(gs.glowTime * 6.0f);
      SDL_Color fc = {(Uint8)(C_FOOD.r * pulse), (Uint8)(C_FOOD.g * pulse),
                      (Uint8)(C_FOOD.b * pulse), 255};
      fillCell(r, gs.food.x, gs.food.y, {fc.r, fc.g, fc.b, 60}, -3);
      drawRoundedCell(r, gs.food.x, gs.food.y, fc);
      fillCell(r, gs.food.x, gs.food.y, {255, 255, 255, 180}, CELL / 2 - 4);
    }

    // snake
    for (size_t i = 0; i < gs.snake.size(); ++i) {
      SDL_Color col = (i == 0) ? C_HEAD : C_BODY;
      if (i > 0) {
        float t = (float)i / gs.snake.size();
        col.r = (Uint8)(col.r * (1.0f - t * 0.5f));
        col.g = (Uint8)(col.g * (1.0f - t * 0.5f));
        col.b = (Uint8)(col.b * (1.0f - t * 0.5f));
      }
      drawRoundedCell(r, gs.snake[i].x, gs.snake[i].y, col);
    }

    // HUD — pixel score
    int hudY = ORIGIN_Y - 26;
    drawNumber(r, gs.score, ORIGIN_X + 4, hudY, C_TEXT);
    int bw = numberWidth(gs.highScore);
    drawNumber(r, gs.highScore, ORIGIN_X + GRID_COLS * CELL - bw - 4, hudY,
               C_FOOD);

    // start screen — draw big pixel squares as banner
    if (!gs.started && !gs.gameOver) {
      // Pulsing "press space" indicator (a blinking rect)
      float blink = SDL_sinf(gs.glowTime * 4.0f);
      if (blink > 0) {
        SDL_SetRenderDrawBlendMode(r, SDL_BLENDMODE_BLEND);
        setColor(r, {232, 72, 153, (Uint8)(blink * 180)});
        SDL_FRect hint = {(float)(WIN_W / 2 - 80), (float)(WIN_H / 2 - 8), 160,
                          16};
        SDL_RenderFillRect(r, &hint);
        SDL_SetRenderDrawBlendMode(r, SDL_BLENDMODE_NONE);
      }
    }

    if (gs.gameOver) {
      SDL_SetRenderDrawBlendMode(r, SDL_BLENDMODE_BLEND);
      setColor(r, {0, 0, 0, 160});
      SDL_FRect full = {0, 0, (float)WIN_W, (float)WIN_H};
      SDL_RenderFillRect(r, &full);
      SDL_SetRenderDrawBlendMode(r, SDL_BLENDMODE_NONE);

      // GAME OVER — large pixel score display
      int sw = numberWidth(gs.score);
      drawNumber(r, gs.score, WIN_W / 2 - sw / 2, WIN_H / 2 - 10, C_OVERLAY);

      // blinking restart hint
      float blink = SDL_sinf(gs.glowTime * 5.0f);
      if (blink > 0) {
        SDL_SetRenderDrawBlendMode(r, SDL_BLENDMODE_BLEND);
        setColor(r, {255, 255, 255, (Uint8)(blink * 120)});
        SDL_FRect hint = {(float)(WIN_W / 2 - 70), (float)(WIN_H / 2 + 30), 140,
                          10};
        SDL_RenderFillRect(r, &hint);
        SDL_SetRenderDrawBlendMode(r, SDL_BLENDMODE_NONE);
      }
    }

    engine_present(engine);

    Uint64 t1 = SDL_GetTicks();
    Uint64 el = t1 - t0;
    if (el < 7) {
      SDL_Delay((Uint32)(7 - el));
      t1 = SDL_GetTicks();
    }

    dt = (float)(t1 - t0) / 1000.0f;
    if (dt < 0.001f)
      dt = 0.001f;
  }

  engine_destroy(engine);
  return 0;
}
