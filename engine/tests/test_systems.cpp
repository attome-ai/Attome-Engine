// JSON, config/tunables, input, audio, assets and text.

#include "atm_test.h"

#include "ATMAssets.h"
#include "ATMAudio.h"
#include "ATMConfig.h"
#include "ATMEngine.h"
#include "ATMInput.h"
#include "ATMJson.h"
#if defined(ATM_HAS_TEXT)
#include "ATMText.h"
#endif

#include <string>
#include <vector>

// Tunables declared the same way games declare them.
namespace test_tunables {
ATM_TUNABLE_SECTION("player");
ATM_TUNABLE(float, kSpeed, 540.0f);
ATM_TUNABLE(int, kLives, 3);
ATM_TUNABLE(bool, kGodMode, false);
ATM_TUNABLE(std::string, kName, "hero");
namespace weapons {
ATM_TUNABLE_SECTION("player.weapons");
ATM_TUNABLE(uint8_t, kSlots, 2);
} // namespace weapons
} // namespace test_tunables

namespace {

std::string temp_path(const char *name) {
  const char *base = SDL_GetPrefPath("attome", "tests");
  std::string path = base ? std::string(base) + name : std::string(name);
  SDL_free(const_cast<char *>(base));
  return path;
}

SDL_Event key_event(SDL_Scancode scancode, bool down) {
  SDL_Event e{};
  e.type = down ? SDL_EVENT_KEY_DOWN : SDL_EVENT_KEY_UP;
  e.key.scancode = scancode;
  e.key.down = down;
  return e;
}

} // namespace

// SECTION: json

ATM_TEST(json_parse_values_and_comments) {
  atm::Json j;
  std::string err;
  const bool ok = atm::Json::parse(R"({
    // comment
    "a": 1, "b": -2.5e1, "c": true, "d": null,
    "s": "x\"yé\n", /* block */ "arr": [1, 2, 3,],
    "obj": {"nested": {"deep": "yes"}},
  })",
                                   j, &err);
  ATM_REQUIRE(ok);
  ATM_CHECK_EQ(j.find("a")->asNumber(), 1.0);
  ATM_CHECK_EQ(j.find("b")->asNumber(), -25.0);
  ATM_CHECK(j.find("c")->asBool());
  ATM_CHECK(j.find("d")->isNull());
  ATM_CHECK_EQ(j.find("s")->asString(), std::string("x\"y\xC3\xA9\n"));
  ATM_CHECK_EQ(j.find("arr")->size(), 3u);
  ATM_CHECK_EQ(j.findPath("obj.nested.deep")->asString(), std::string("yes"));
  ATM_CHECK(j.findPath("obj.missing") == nullptr);
}

ATM_TEST(json_errors_report_position) {
  atm::Json j;
  std::string err;
  ATM_CHECK(!atm::Json::parse("{\n  \"a\": ,\n}", j, &err));
  ATM_CHECK(err.rfind("2:", 0) == 0);
  ATM_CHECK(!atm::Json::parse("[1, 2", j, &err));
  ATM_CHECK(!atm::Json::parse("{\"a\": 1} x", j, &err));
  ATM_CHECK(!atm::Json::parse("\"unterminated", j, &err));
}

ATM_TEST(json_round_trip) {
  atm::Json root = atm::Json::object();
  root["int"] = atm::Json(42);
  root["float"] = atm::Json(0.25);
  root["str"] = atm::Json("a\tb");
  root["color"] = atm::Json(atm::Json::Array{atm::Json(1), atm::Json(2)});
  root["nested"]["x"] = atm::Json(true);

  atm::Json back;
  ATM_REQUIRE(atm::Json::parse(root.dump(), back, nullptr));
  ATM_CHECK_EQ(back.find("int")->asNumber(), 42.0);
  ATM_CHECK_EQ(back.find("float")->asNumber(), 0.25);
  ATM_CHECK_EQ(back.find("str")->asString(), std::string("a\tb"));
  ATM_CHECK_EQ(back.find("color")->size(), 2u);
  ATM_CHECK(back.findPath("nested.x")->asBool());
  // Integers print without a trailing ".0"; float values print short.
  ATM_CHECK(root.dump().find("42.0") == std::string::npos);
  ATM_CHECK_EQ(atm::Json(0.7f).dump(0), std::string("0.7\n"));
  ATM_CHECK_EQ(atm::Json(0.1).dump(0), std::string("0.1\n"));
  atm::Json pi;
  ATM_REQUIRE(atm::Json::parse(atm::Json(3.14159265358979).dump(0), pi, nullptr));
  ATM_CHECK_EQ(pi.asNumber(), 3.14159265358979);
}

// SECTION: engine_config

ATM_TEST(engine_config_json) {
  atm::Json j;
  ATM_REQUIRE(atm::Json::parse(R"({
    "window": {"title": "Test", "width": 800, "vsync": true},
    "grid": {"cell_size": 128},
    "time": {"fixed_timestep_hz": 60},
    "audio": {"max_voices": 8}
  })",
                               j, nullptr));
  EngineConfig cfg;
  ATM_CHECK(engine_config_from_json(j, cfg));
  ATM_CHECK_EQ(cfg.window_title, std::string("Test"));
  ATM_CHECK_EQ(cfg.window_width, 800);
  ATM_CHECK_EQ(cfg.window_height, 720); // untouched default
  ATM_CHECK(cfg.vsync);
  ATM_CHECK_EQ(cfg.grid_cell_size, 128);
  ATM_CHECK_EQ(cfg.fixed_timestep_hz, 60.0f);
  ATM_CHECK_EQ(cfg.audio_max_voices, 8);

  // Round trip through JSON gives the same config.
  EngineConfig again;
  ATM_CHECK(engine_config_from_json(engine_config_to_json(cfg), again));
  ATM_CHECK_EQ(again.window_title, cfg.window_title);
  ATM_CHECK_EQ(again.grid_cell_size, cfg.grid_cell_size);
}

ATM_TEST(engine_config_rejects_wrong_types_and_clamps) {
  atm::Json j;
  ATM_REQUIRE(atm::Json::parse(
      R"({"window": {"width": "wide"}, "grid": {"cell_size": -5}})", j, nullptr));
  EngineConfig cfg;
  ATM_CHECK(!engine_config_from_json(j, cfg));
  ATM_CHECK_EQ(cfg.window_width, 1280); // kept
  ATM_CHECK_EQ(cfg.grid_cell_size, 1);  // clamped to a valid value
}

// SECTION: tunables

ATM_TEST(tunables_apply_and_dump) {
  using namespace test_tunables;
  atm::Json j;
  ATM_REQUIRE(atm::Json::parse(R"({"player": {"kSpeed": 600, "kLives": 5,
      "kName": "ace", "weapons": {"kSlots": 4}, "kTypo": 1}})",
                               j, nullptr));
  ATM_CHECK(atm::Tunables::instance().apply(j));
  ATM_CHECK_EQ(kSpeed, 600.0f);
  ATM_CHECK_EQ(kLives, 5);
  ATM_CHECK_EQ(kGodMode, false); // missing -> default kept
  ATM_CHECK_EQ(kName, std::string("ace"));
  ATM_CHECK_EQ(weapons::kSlots, 4);

  const atm::Json dumped = atm::Tunables::instance().toJson();
  ATM_CHECK_EQ(dumped.findPath("player.kLives")->asNumber(), 5.0);
  ATM_CHECK_EQ(dumped.findPath("player.weapons.kSlots")->asNumber(), 4.0);
}

ATM_TEST(tunables_reject_bad_values) {
  using namespace test_tunables;
  kLives = 3;
  weapons::kSlots = 2;
  atm::Json j;
  ATM_REQUIRE(atm::Json::parse(
      R"({"player": {"kLives": 2.5, "weapons": {"kSlots": 300}}})", j, nullptr));
  ATM_CHECK(!atm::Tunables::instance().apply(j));
  ATM_CHECK_EQ(kLives, 3);          // not an integer
  ATM_CHECK_EQ(weapons::kSlots, 2); // out of uint8_t range
}

ATM_TEST(tunables_load_and_hot_reload) {
  using namespace test_tunables;
  const std::string path = temp_path("tunables_test.json");
  ATM_REQUIRE(atm::write_text_file(path, R"({"player": {"kLives": 7}})"));

  int reloads = 0;
  atm::Tunables::instance().addReloadListener([&reloads] { ++reloads; });

  ATM_REQUIRE(atm::Tunables::instance().loadFile(path));
  ATM_CHECK_EQ(kLives, 7);

  // Unchanged file: no reload.
  ATM_CHECK(!atm::Tunables::instance().reloadIfChanged(0));

  // Make sure the timestamp moves even on coarse file systems.
  SDL_Delay(1100);
  ATM_REQUIRE(atm::write_text_file(path, R"({"player": {"kLives": 9}})"));
  ATM_CHECK(atm::Tunables::instance().reloadIfChanged(0));
  ATM_CHECK_EQ(kLives, 9);
  ATM_CHECK(reloads >= 2);

  // A broken file keeps the previous values.
  SDL_Delay(1100);
  ATM_REQUIRE(atm::write_text_file(path, R"({"player": {"kLives": )"));
  ATM_CHECK(!atm::Tunables::instance().reloadIfChanged(0));
  ATM_CHECK_EQ(kLives, 9);
  SDL_RemovePath(path.c_str());
}

// SECTION: input

ATM_TEST(input_actions_pressed_held_released) {
  atm::InputMap input;
  ATM_REQUIRE(input.bind("jump", "Space"));
  ATM_REQUIRE(input.bind("jump", "W"));
  ATM_CHECK(!input.bind("jump", "NotAKey"));
  const atm::ActionId jump = input.actionId("jump");

  input.beginFrame();
  input.handleEvent(key_event(SDL_SCANCODE_SPACE, true));
  ATM_CHECK(input.pressed(jump));
  ATM_CHECK(input.held(jump));

  input.beginFrame();
  input.handleEvent(key_event(SDL_SCANCODE_SPACE, true)); // key repeat
  ATM_CHECK(!input.pressed(jump));
  input.handleEvent(key_event(SDL_SCANCODE_W, true));
  input.handleEvent(key_event(SDL_SCANCODE_SPACE, false));
  ATM_CHECK(input.held(jump)); // W still down
  ATM_CHECK(!input.released(jump));

  input.beginFrame();
  input.handleEvent(key_event(SDL_SCANCODE_W, false));
  ATM_CHECK(input.released(jump));
  ATM_CHECK(!input.held(jump));
}

ATM_TEST(input_bindings_from_json_and_axes) {
  atm::Json j;
  ATM_REQUIRE(atm::Json::parse(R"({
    "bindings": {"left": ["A", "Left"], "right": "D", "fire": ["mouse:left"]},
    "axes": {"move_x": {"negative": "left", "positive": "right"}}
  })",
                               j, nullptr));
  atm::InputMap input;
  ATM_CHECK(input.loadBindings(j));

  input.beginFrame();
  input.handleEvent(key_event(SDL_SCANCODE_LEFT, true));
  ATM_CHECK_EQ(input.axis("move_x"), -1.0f);
  input.handleEvent(key_event(SDL_SCANCODE_D, true));
  ATM_CHECK_EQ(input.axis("move_x"), 0.0f);

  SDL_Event click{};
  click.type = SDL_EVENT_MOUSE_BUTTON_DOWN;
  click.button.button = SDL_BUTTON_LEFT;
  click.button.x = 12;
  click.button.y = 34;
  input.handleEvent(click);
  ATM_CHECK(input.pressed("fire"));
  ATM_CHECK_EQ(input.mouseX(), 12.0f);

  // Focus loss releases everything that was held.
  SDL_Event blur{};
  blur.type = SDL_EVENT_WINDOW_FOCUS_LOST;
  input.handleEvent(blur);
  ATM_CHECK(!input.held("left"));
  ATM_CHECK(!input.held("fire"));

  const atm::Json saved = input.bindingsToJson();
  ATM_CHECK_EQ(saved.findPath("bindings.left")->size(), 2u);
  ATM_CHECK(saved.findPath("axes.move_x") != nullptr);
}

// SECTION: audio

ATM_TEST(audio_plays_one_shots_and_loops) {
  atm::Audio audio;
  ATM_REQUIRE(audio.init(4, 0.5f));
  ATM_CHECK_EQ(audio.voiceCount(), 4);
  ATM_CHECK_EQ(audio.masterVolume(), 0.5f);

  // 0.1s of 16-bit mono silence.
  const SDL_AudioSpec spec{SDL_AUDIO_S16, 1, 22050};
  std::vector<uint8_t> pcm(2205 * 2, 0);
  const atm::SoundId sound =
      audio.loadSoundFromMemory(spec, pcm.data(), static_cast<uint32_t>(pcm.size()));
  ATM_REQUIRE(sound != atm::kInvalidSound);

  const atm::VoiceId one_shot = audio.play(sound);
  ATM_CHECK(one_shot.valid());
  ATM_CHECK(audio.isPlaying(one_shot));
  audio.stop(one_shot);
  ATM_CHECK(!audio.isPlaying(one_shot));

  const atm::VoiceId loop = audio.play(sound, 0.8f, true);
  ATM_CHECK(audio.isPlaying(loop));

  // Voices are stolen once all are busy; the stale handle becomes invalid.
  std::vector<atm::VoiceId> shots;
  for (int i = 0; i < 6; ++i)
    shots.push_back(audio.play(sound));
  ATM_CHECK(!audio.isPlaying(shots[0]));
  ATM_CHECK(audio.isPlaying(loop)); // loops are stolen last

  audio.unloadSound(sound);
  ATM_CHECK(!audio.isPlaying(loop));
  ATM_CHECK(!audio.play(sound).valid());
}

// SECTION: assets

ATM_TEST(assets_are_reference_counted) {
  EngineConfig cfg;
  cfg.window_width = 64;
  cfg.window_height = 64;
  cfg.grid_node_reserve = 16;
  Engine *engine = engine_create_with_config(cfg);
  ATM_REQUIRE(engine);
  {
    atm::Assets assets(engine);
    const std::string png = std::string(ATM_TEST_DATA_DIR) + "/ship1.png";
    const int a = assets.acquireTexture(png);
    ATM_REQUIRE(a >= 0);
    const int b = assets.acquireTexture(png);
    ATM_CHECK_EQ(a, b);
    ATM_CHECK_EQ(assets.refCount(png), 2);

    assets.releaseTexture(png);
    ATM_CHECK(engine->atlas.getTexture(a) != nullptr);
    assets.releaseTexture(png);
    ATM_CHECK_EQ(assets.refCount(png), 0);
    ATM_CHECK(engine->atlas.getTexture(a) == nullptr);

    ATM_CHECK_EQ(assets.acquireTexture("does/not/exist.png"), -1);
  }
  engine_destroy(engine);
}

#if defined(ATM_HAS_TEXT)
ATM_TEST(text_glyph_atlas_and_measure) {
  EngineConfig cfg;
  cfg.window_width = 64;
  cfg.window_height = 64;
  cfg.grid_node_reserve = 16;
  Engine *engine = engine_create_with_config(cfg);
  ATM_REQUIRE(engine);
  {
    atm::Font font;
    const std::string ttf = std::string(ATM_TEST_FONT);
    ATM_REQUIRE(font.load(engine->renderer, ttf, 16.0f));
    ATM_CHECK(font.lineHeight() > 0.0f);

    const SDL_FPoint one = font.measure("Hello");
    const SDL_FPoint two = font.measure("Hello\nWorld!");
    ATM_CHECK(one.x > 0.0f);
    ATM_CHECK_NEAR(two.y, 2.0f * font.lineHeight(), 0.01);
    ATM_CHECK_NEAR(font.measure("Hello", 2.0f).x, one.x * 2.0f, 0.01);

    font.draw(engine->renderer, "Score: 120 \xC3\xA9", 0, 0); // non-ASCII glyph
    atm::Font moved(std::move(font));
    ATM_CHECK(moved.isLoaded());
    ATM_CHECK(!font.isLoaded());
  }
  engine_destroy(engine);
}
#endif
