#include "ATMInput.h"

#include <algorithm>
#include <cmath>

namespace atm {

namespace {

uint64_t input_key(uint8_t source, int32_t code) {
  return (static_cast<uint64_t>(source) << 32) | static_cast<uint32_t>(code);
}

struct NamedCode {
  const char *name;
  int32_t code;
};

constexpr NamedCode kMouseButtons[] = {
    {"left", SDL_BUTTON_LEFT}, {"middle", SDL_BUTTON_MIDDLE},
    {"right", SDL_BUTTON_RIGHT}, {"x1", SDL_BUTTON_X1},
    {"x2", SDL_BUTTON_X2},
};

bool iequals(std::string_view a, std::string_view b) {
  if (a.size() != b.size())
    return false;
  for (size_t i = 0; i < a.size(); ++i) {
    const char ca = static_cast<char>(SDL_tolower(a[i]));
    const char cb = static_cast<char>(SDL_tolower(b[i]));
    if (ca != cb)
      return false;
  }
  return true;
}

} // namespace

InputMap::InputMap() = default;

InputMap::~InputMap() {
  if (gamepad_) {
    SDL_CloseGamepad(gamepad_);
  }
}

ActionId InputMap::actionId(std::string_view name) {
  const std::string key(name);
  const auto it = action_ids_.find(key);
  if (it != action_ids_.end())
    return it->second;
  const ActionId id = static_cast<ActionId>(actions_.size());
  actions_.push_back(Action{key, {}, 0, false, false});
  action_ids_.emplace(key, id);
  return id;
}

ActionId InputMap::findAction(std::string_view name) const {
  const auto it = action_ids_.find(std::string(name));
  return it == action_ids_.end() ? kInvalidAction : it->second;
}

bool InputMap::parseBinding(std::string_view text, Binding &out) {
  if (text.rfind("mouse:", 0) == 0) {
    const std::string_view button = text.substr(6);
    for (const NamedCode &named : kMouseButtons) {
      if (iequals(button, named.name)) {
        out = {Source::Mouse, named.code};
        return true;
      }
    }
    return false;
  }
  if (text.rfind("pad:", 0) == 0) {
    const std::string button(text.substr(4));
    const SDL_GamepadButton code = SDL_GetGamepadButtonFromString(button.c_str());
    if (code == SDL_GAMEPAD_BUTTON_INVALID)
      return false;
    out = {Source::PadButton, static_cast<int32_t>(code)};
    return true;
  }
  const std::string key(text);
  const SDL_Scancode code = SDL_GetScancodeFromName(key.c_str());
  if (code == SDL_SCANCODE_UNKNOWN)
    return false;
  out = {Source::Key, static_cast<int32_t>(code)};
  return true;
}

std::string InputMap::bindingToString(const Binding &binding) {
  switch (binding.source) {
  case Source::Key:
    return SDL_GetScancodeName(static_cast<SDL_Scancode>(binding.code));
  case Source::Mouse:
    for (const NamedCode &named : kMouseButtons)
      if (named.code == binding.code)
        return std::string("mouse:") + named.name;
    return "mouse:?";
  case Source::PadButton: {
    const char *name =
        SDL_GetGamepadStringForButton(static_cast<SDL_GamepadButton>(binding.code));
    return std::string("pad:") + (name ? name : "?");
  }
  }
  return {};
}

bool InputMap::bind(std::string_view action, std::string_view binding_text) {
  Binding binding{};
  if (!parseBinding(binding_text, binding)) {
    SDL_Log("[input] unknown binding '%.*s' for action '%.*s'",
            static_cast<int>(binding_text.size()), binding_text.data(),
            static_cast<int>(action.size()), action.data());
    return false;
  }
  const ActionId id = actionId(action);
  Action &a = actions_[id];
  for (const Binding &existing : a.bindings)
    if (existing.source == binding.source && existing.code == binding.code)
      return true;
  a.bindings.push_back(binding);
  by_input_[input_key(static_cast<uint8_t>(binding.source), binding.code)]
      .push_back(id);
  return true;
}

void InputMap::clearBindings(std::string_view action) {
  const ActionId id = findAction(action);
  if (id == kInvalidAction)
    return;
  for (const Binding &b : actions_[id].bindings) {
    auto &ids = by_input_[input_key(static_cast<uint8_t>(b.source), b.code)];
    ids.erase(std::remove(ids.begin(), ids.end(), id), ids.end());
  }
  actions_[id].bindings.clear();
  actions_[id].down_count = 0;
}

void InputMap::clearAllBindings() {
  for (Action &a : actions_) {
    a.bindings.clear();
    a.down_count = 0;
  }
  by_input_.clear();
}

void InputMap::defineAxis(std::string_view axis,
                          std::string_view negative_action,
                          std::string_view positive_action,
                          std::string_view pad_axis, float deadzone) {
  Axis def;
  def.negative = actionId(negative_action);
  def.positive = actionId(positive_action);
  def.deadzone = std::clamp(deadzone, 0.0f, 0.99f);
  if (!pad_axis.empty()) {
    const std::string name(pad_axis);
    const SDL_GamepadAxis a = SDL_GetGamepadAxisFromString(name.c_str());
    def.pad_axis = a == SDL_GAMEPAD_AXIS_INVALID ? -1 : static_cast<int32_t>(a);
  }
  axes_[std::string(axis)] = def;
}

bool InputMap::loadBindings(const Json &input_section, std::string *error) {
  if (!input_section.isObject()) {
    if (error)
      *error = "input section must be an object";
    return false;
  }

  bool ok = true;
  if (const Json *bindings = input_section.find("bindings")) {
    for (const auto &[action, list] : bindings->asObject()) {
      clearBindings(action);
      actionId(action);
      const auto add = [&](const Json &item) {
        if (!item.isString() || !bind(action, item.asString()))
          ok = false;
      };
      if (list.isArray()) {
        for (const Json &item : list.asArray())
          add(item);
      } else {
        add(list);
      }
    }
  }

  if (const Json *axes = input_section.find("axes")) {
    for (const auto &[name, def] : axes->asObject()) {
      const Json *neg = def.find("negative");
      const Json *pos = def.find("positive");
      const Json *pad = def.find("pad_axis");
      const Json *dz = def.find("deadzone");
      if (!neg || !pos || !neg->isString() || !pos->isString()) {
        SDL_Log("[input] axis '%s' needs string 'negative' and 'positive'",
                name.c_str());
        ok = false;
        continue;
      }
      defineAxis(name, neg->asString(), pos->asString(),
                 pad && pad->isString() ? std::string_view(pad->asString())
                                        : std::string_view(),
                 dz ? static_cast<float>(dz->asNumber(0.2)) : 0.2f);
    }
  }

  if (!ok && error)
    *error = "some input bindings were invalid (see log)";
  return ok;
}

Json InputMap::bindingsToJson() const {
  Json root = Json::object();
  Json &bindings = root["bindings"];
  bindings = Json::object();
  for (const Action &a : actions_) {
    Json list = Json::array();
    for (const Binding &b : a.bindings)
      list.push(Json(bindingToString(b)));
    bindings[a.name] = std::move(list);
  }
  if (!axes_.empty()) {
    Json &axes = root["axes"];
    axes = Json::object();
    for (const auto &[name, def] : axes_) {
      Json entry = Json::object();
      entry["negative"] = Json(actions_[def.negative].name);
      entry["positive"] = Json(actions_[def.positive].name);
      if (def.pad_axis >= 0) {
        const char *axis_name = SDL_GetGamepadStringForAxis(
            static_cast<SDL_GamepadAxis>(def.pad_axis));
        entry["pad_axis"] = Json(axis_name ? axis_name : "");
      }
      entry["deadzone"] = Json(def.deadzone);
      axes[name] = std::move(entry);
    }
  }
  return root;
}

void InputMap::beginFrame() {
  for (Action &a : actions_) {
    a.pressed = false;
    a.released = false;
  }
  wheel_y_ = 0.0f;
}

void InputMap::onInput(Source source, int32_t code, bool down) {
  const uint64_t key = input_key(static_cast<uint8_t>(source), code);
  bool &was_down = input_down_[key];
  if (was_down == down)
    return; // key repeat or duplicate event
  was_down = down;

  const auto it = by_input_.find(key);
  if (it == by_input_.end())
    return;
  for (const ActionId id : it->second) {
    Action &a = actions_[id];
    if (down) {
      if (a.down_count++ == 0)
        a.pressed = true;
    } else if (a.down_count > 0) {
      if (--a.down_count == 0)
        a.released = true;
    }
  }
}

void InputMap::openGamepad(SDL_JoystickID id) {
  if (gamepad_)
    return;
  gamepad_ = SDL_OpenGamepad(id);
}

void InputMap::handleEvent(const SDL_Event &event) {
  switch (event.type) {
  case SDL_EVENT_KEY_DOWN:
  case SDL_EVENT_KEY_UP:
    onInput(Source::Key, static_cast<int32_t>(event.key.scancode),
            event.type == SDL_EVENT_KEY_DOWN);
    break;
  case SDL_EVENT_MOUSE_BUTTON_DOWN:
  case SDL_EVENT_MOUSE_BUTTON_UP:
    mouse_x_ = event.button.x;
    mouse_y_ = event.button.y;
    onInput(Source::Mouse, event.button.button,
            event.type == SDL_EVENT_MOUSE_BUTTON_DOWN);
    break;
  case SDL_EVENT_MOUSE_MOTION:
    mouse_x_ = event.motion.x;
    mouse_y_ = event.motion.y;
    break;
  case SDL_EVENT_MOUSE_WHEEL:
    wheel_y_ += event.wheel.y;
    break;
  case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
  case SDL_EVENT_GAMEPAD_BUTTON_UP:
    onInput(Source::PadButton, event.gbutton.button,
            event.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN);
    break;
  case SDL_EVENT_GAMEPAD_ADDED:
    // Only delivered when the game initialised SDL_INIT_GAMEPAD.
    openGamepad(event.gdevice.which);
    break;
  case SDL_EVENT_GAMEPAD_REMOVED:
    if (gamepad_ && SDL_GetGamepadID(gamepad_) == event.gdevice.which) {
      SDL_CloseGamepad(gamepad_);
      gamepad_ = nullptr;
      // Release any pad buttons that were held when it disconnected.
      for (int b = 0; b < SDL_GAMEPAD_BUTTON_COUNT; ++b)
        onInput(Source::PadButton, b, false);
    }
    break;
  case SDL_EVENT_WINDOW_FOCUS_LOST:
    // Keys released while unfocused never send KEY_UP; drop them now.
    for (auto &[key, down] : input_down_) {
      if (down)
        onInput(static_cast<Source>(key >> 32),
                static_cast<int32_t>(key & 0xFFFFFFFFu), false);
    }
    break;
  default:
    break;
  }
}

bool InputMap::held(ActionId action) const {
  return action >= 0 && action < static_cast<ActionId>(actions_.size()) &&
         actions_[action].down_count > 0;
}

bool InputMap::pressed(ActionId action) const {
  return action >= 0 && action < static_cast<ActionId>(actions_.size()) &&
         actions_[action].pressed;
}

bool InputMap::released(ActionId action) const {
  return action >= 0 && action < static_cast<ActionId>(actions_.size()) &&
         actions_[action].released;
}

float InputMap::axis(std::string_view axis_name) const {
  const auto it = axes_.find(std::string(axis_name));
  if (it == axes_.end())
    return 0.0f;
  const Axis &def = it->second;
  float value = (held(def.positive) ? 1.0f : 0.0f) -
                (held(def.negative) ? 1.0f : 0.0f);
  if (value == 0.0f && gamepad_ && def.pad_axis >= 0) {
    const float raw =
        SDL_GetGamepadAxis(gamepad_, static_cast<SDL_GamepadAxis>(def.pad_axis)) /
        32767.0f;
    if (std::fabs(raw) > def.deadzone) {
      // Rescale so the output starts at 0 right outside the deadzone.
      value = (std::fabs(raw) - def.deadzone) / (1.0f - def.deadzone);
      value = std::copysign(std::min(value, 1.0f), raw);
    }
  }
  return value;
}

} // namespace atm
