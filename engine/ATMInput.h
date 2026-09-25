#ifndef ATM_INPUT_H
#define ATM_INPUT_H

// Action-based input. Games ask "is `jump` pressed?" instead of checking
// keys, and players/designers rebind keys in JSON:
//
//   "input": {
//     "bindings": {
//       "move_left":  ["A", "Left", "pad:dpleft"],
//       "fire":       ["Space", "mouse:left", "pad:a"]
//     },
//     "axes": {
//       "move_x": { "negative": "move_left", "positive": "move_right",
//                   "pad_axis": "leftx", "deadzone": 0.2 }
//     }
//   }
//
// Key names are SDL scancode names ("A", "Space", "Left", "Left Shift", ...).
// Mouse: "mouse:left|middle|right|x1|x2". Gamepad buttons: "pad:<SDL name>"
// (a, b, x, y, back, start, leftshoulder, dpup, ...).
//
// Per frame: beginFrame(), handleEvent() for every SDL event, then query.
// Lookups use integer ids (ActionId); resolve names once at init with
// actionId("fire") and keep the id.

#include "ATMJson.h"

#include <SDL3/SDL.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace atm {

using ActionId = int32_t;
inline constexpr ActionId kInvalidAction = -1;

class InputMap {
public:
  InputMap();
  ~InputMap();

  InputMap(const InputMap &) = delete;
  InputMap &operator=(const InputMap &) = delete;

  // Returns the id for `name`, creating an unbound action if needed.
  ActionId actionId(std::string_view name);
  // kInvalidAction when the action was never declared/bound.
  ActionId findAction(std::string_view name) const;

  // Binding strings use the formats described at the top of this file.
  bool bind(std::string_view action, std::string_view binding);
  void clearBindings(std::string_view action);
  void clearAllBindings();

  // Axis in [-1, 1] from two actions and optionally a gamepad stick axis.
  void defineAxis(std::string_view axis, std::string_view negative_action,
                  std::string_view positive_action,
                  std::string_view pad_axis = {}, float deadzone = 0.2f);

  // Reads {"bindings": {...}, "axes": {...}}. Pass the "input" object of a
  // config file. Replaces the bindings of every action it mentions.
  bool loadBindings(const Json &input_section, std::string *error = nullptr);
  Json bindingsToJson() const;

  // Frame lifecycle.
  void beginFrame();
  void handleEvent(const SDL_Event &event);

  bool held(ActionId action) const;
  bool pressed(ActionId action) const;  // went down this frame
  bool released(ActionId action) const; // went up this frame
  float axis(std::string_view axis_name) const;

  // Convenience name-based queries (hash lookup; prefer ids in hot code).
  bool held(std::string_view action) const { return held(findAction(action)); }
  bool pressed(std::string_view action) const {
    return pressed(findAction(action));
  }
  bool released(std::string_view action) const {
    return released(findAction(action));
  }

  float mouseX() const { return mouse_x_; }
  float mouseY() const { return mouse_y_; }
  float mouseWheel() const { return wheel_y_; }

private:
  enum class Source : uint8_t { Key, Mouse, PadButton };

  struct Binding {
    Source source;
    int32_t code; // SDL_Scancode / mouse button / SDL_GamepadButton
  };

  struct Action {
    std::string name;
    std::vector<Binding> bindings;
    // How many bound inputs are currently down (an action stays held while
    // any of its inputs is held).
    int32_t down_count = 0;
    bool pressed = false;
    bool released = false;
  };

  struct Axis {
    ActionId negative = kInvalidAction;
    ActionId positive = kInvalidAction;
    int32_t pad_axis = -1; // SDL_GamepadAxis
    float deadzone = 0.2f;
  };

  static bool parseBinding(std::string_view text, Binding &out);
  static std::string bindingToString(const Binding &binding);
  void onInput(Source source, int32_t code, bool down);
  void openGamepad(SDL_JoystickID id);

  std::vector<Action> actions_;
  std::unordered_map<std::string, ActionId> action_ids_;
  std::unordered_map<std::string, Axis> axes_;

  // Reverse lookup from an input to the actions bound to it.
  std::unordered_map<uint64_t, std::vector<ActionId>> by_input_;
  // Inputs currently down, so a key repeat or duplicate event never counts
  // twice.
  std::unordered_map<uint64_t, bool> input_down_;

  SDL_Gamepad *gamepad_ = nullptr;
  float mouse_x_ = 0.0f;
  float mouse_y_ = 0.0f;
  float wheel_y_ = 0.0f;
};

} // namespace atm

#endif // ATM_INPUT_H
