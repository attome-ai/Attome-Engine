#ifndef ATM_JSON_H
#define ATM_JSON_H

// Minimal, dependency-free JSON DOM used for engine/game config files.
// Not meant for hot paths: parse once at load time, read values into plain
// fields, then throw the DOM away.

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace atm {

class Json {
public:
  enum class Type : uint8_t { Null, Bool, Number, String, Array, Object };

  using Array = std::vector<Json>;
  using Object = std::map<std::string, Json, std::less<>>;

  Json() = default;
  Json(std::nullptr_t) {}
  Json(bool v) : type_(Type::Bool), bool_(v) {}
  Json(int v) : type_(Type::Number), number_(v) {}
  Json(unsigned v) : type_(Type::Number), number_(v) {}
  Json(int64_t v) : type_(Type::Number), number_(static_cast<double>(v)) {}
  Json(uint64_t v) : type_(Type::Number), number_(static_cast<double>(v)) {}
  Json(float v) : type_(Type::Number), number_(v) {}
  Json(double v) : type_(Type::Number), number_(v) {}
  Json(const char *v) : type_(Type::String), string_(v) {}
  Json(std::string v) : type_(Type::String), string_(std::move(v)) {}
  Json(Array v) : type_(Type::Array), array_(std::move(v)) {}
  Json(Object v) : type_(Type::Object), object_(std::move(v)) {}

  static Json object() { return Json(Object{}); }
  static Json array() { return Json(Array{}); }

  Type type() const { return type_; }
  bool isNull() const { return type_ == Type::Null; }
  bool isBool() const { return type_ == Type::Bool; }
  bool isNumber() const { return type_ == Type::Number; }
  bool isString() const { return type_ == Type::String; }
  bool isArray() const { return type_ == Type::Array; }
  bool isObject() const { return type_ == Type::Object; }

  bool asBool(bool fallback = false) const {
    return type_ == Type::Bool ? bool_ : fallback;
  }
  double asNumber(double fallback = 0.0) const {
    return type_ == Type::Number ? number_ : fallback;
  }
  const std::string &asString() const { return string_; }
  const Array &asArray() const { return array_; }
  const Object &asObject() const { return object_; }
  Array &asArray() { return array_; }
  Object &asObject() { return object_; }

  // Object access. find() returns nullptr when missing or not an object.
  const Json *find(std::string_view key) const;
  // Dotted path lookup: "window.width".
  const Json *findPath(std::string_view dotted_path) const;
  // Inserts (converting this value to an object if needed).
  Json &operator[](const std::string &key);

  void push(Json v);
  size_t size() const;

  // Returns false and fills *error ("line:col: message") on failure.
  static bool parse(std::string_view text, Json &out, std::string *error);
  static bool parseFile(const std::string &path, Json &out, std::string *error);

  std::string dump(int indent = 2) const;

private:
  void dumpTo(std::string &out, int indent, int depth) const;

  Type type_ = Type::Null;
  bool bool_ = false;
  double number_ = 0.0;
  std::string string_;
  Array array_;
  Object object_;
};

// Reads a whole file (works with SDL's virtual FS on web/mobile too).
bool read_text_file(const std::string &path, std::string &out);
bool write_text_file(const std::string &path, std::string_view text);

} // namespace atm

#endif // ATM_JSON_H
