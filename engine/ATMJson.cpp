#include "ATMJson.h"

#include <SDL3/SDL.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace atm {

// SECTION: dom

const Json *Json::find(std::string_view key) const {
  if (type_ != Type::Object) {
    return nullptr;
  }
  const auto it = object_.find(key);
  return it == object_.end() ? nullptr : &it->second;
}

const Json *Json::findPath(std::string_view dotted_path) const {
  const Json *node = this;
  while (node && !dotted_path.empty()) {
    const size_t dot = dotted_path.find('.');
    const std::string_view part = dotted_path.substr(0, dot);
    node = node->find(part);
    if (dot == std::string_view::npos) {
      break;
    }
    dotted_path.remove_prefix(dot + 1);
  }
  return node;
}

Json &Json::operator[](const std::string &key) {
  if (type_ != Type::Object) {
    *this = Json(Object{});
  }
  return object_[key];
}

void Json::push(Json v) {
  if (type_ != Type::Array) {
    *this = Json(Array{});
  }
  array_.push_back(std::move(v));
}

size_t Json::size() const {
  if (type_ == Type::Array)
    return array_.size();
  if (type_ == Type::Object)
    return object_.size();
  return 0;
}

// SECTION: parser

namespace {

class Parser {
public:
  explicit Parser(std::string_view text) : text_(text) {}

  bool run(Json &out, std::string *error) {
    skipWhitespace();
    if (!parseValue(out, 0)) {
      return fail(error);
    }
    skipWhitespace();
    if (pos_ != text_.size()) {
      message_ = "unexpected trailing characters";
      return fail(error);
    }
    return true;
  }

private:
  static constexpr int kMaxDepth = 128;

  bool fail(std::string *error) const {
    if (error) {
      int line = 1;
      int col = 1;
      for (size_t i = 0; i < pos_ && i < text_.size(); ++i) {
        if (text_[i] == '\n') {
          ++line;
          col = 1;
        } else {
          ++col;
        }
      }
      *error = std::to_string(line) + ":" + std::to_string(col) + ": " +
               message_;
    }
    return false;
  }

  bool error(const char *message) {
    message_ = message;
    return false;
  }

  // Allows // line comments and /* block */ comments so config files can be
  // annotated. Standard JSON input is unaffected.
  void skipWhitespace() {
    while (pos_ < text_.size()) {
      const char c = text_[pos_];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
        ++pos_;
      } else if (c == '/' && pos_ + 1 < text_.size() && text_[pos_ + 1] == '/') {
        while (pos_ < text_.size() && text_[pos_] != '\n')
          ++pos_;
      } else if (c == '/' && pos_ + 1 < text_.size() &&
                 text_[pos_ + 1] == '*') {
        pos_ += 2;
        while (pos_ + 1 < text_.size() &&
               !(text_[pos_] == '*' && text_[pos_ + 1] == '/'))
          ++pos_;
        pos_ = pos_ + 2 <= text_.size() ? pos_ + 2 : text_.size();
      } else {
        break;
      }
    }
  }

  bool consumeLiteral(std::string_view literal) {
    if (text_.substr(pos_, literal.size()) != literal) {
      return false;
    }
    pos_ += literal.size();
    return true;
  }

  bool parseValue(Json &out, int depth) {
    if (depth > kMaxDepth) {
      return error("nesting too deep");
    }
    if (pos_ >= text_.size()) {
      return error("unexpected end of input");
    }

    switch (text_[pos_]) {
    case '{':
      return parseObject(out, depth);
    case '[':
      return parseArray(out, depth);
    case '"': {
      std::string s;
      if (!parseString(s))
        return false;
      out = Json(std::move(s));
      return true;
    }
    case 't':
      if (consumeLiteral("true")) {
        out = Json(true);
        return true;
      }
      return error("invalid literal");
    case 'f':
      if (consumeLiteral("false")) {
        out = Json(false);
        return true;
      }
      return error("invalid literal");
    case 'n':
      if (consumeLiteral("null")) {
        out = Json();
        return true;
      }
      return error("invalid literal");
    default:
      return parseNumber(out);
    }
  }

  bool parseNumber(Json &out) {
    const size_t start = pos_;
    if (pos_ < text_.size() && (text_[pos_] == '-' || text_[pos_] == '+'))
      ++pos_;
    bool digits = false;
    while (pos_ < text_.size()) {
      const char c = text_[pos_];
      if ((c >= '0' && c <= '9')) {
        digits = true;
        ++pos_;
      } else if (c == '.' || c == 'e' || c == 'E' || c == '-' || c == '+') {
        ++pos_;
      } else {
        break;
      }
    }
    if (!digits) {
      pos_ = start;
      return error("expected a value");
    }

    const std::string token(text_.substr(start, pos_ - start));
    char *end = nullptr;
    const double value = std::strtod(token.c_str(), &end);
    if (!end || *end != '\0' || !std::isfinite(value)) {
      pos_ = start;
      return error("invalid number");
    }
    out = Json(value);
    return true;
  }

  static void appendUtf8(std::string &s, uint32_t cp) {
    if (cp < 0x80) {
      s += static_cast<char>(cp);
    } else if (cp < 0x800) {
      s += static_cast<char>(0xC0 | (cp >> 6));
      s += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
      s += static_cast<char>(0xE0 | (cp >> 12));
      s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      s += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
      s += static_cast<char>(0xF0 | (cp >> 18));
      s += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
      s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      s += static_cast<char>(0x80 | (cp & 0x3F));
    }
  }

  bool parseHex4(uint32_t &out) {
    if (pos_ + 4 > text_.size())
      return error("truncated \\u escape");
    out = 0;
    for (int i = 0; i < 4; ++i) {
      const char c = text_[pos_++];
      out <<= 4;
      if (c >= '0' && c <= '9')
        out |= static_cast<uint32_t>(c - '0');
      else if (c >= 'a' && c <= 'f')
        out |= static_cast<uint32_t>(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F')
        out |= static_cast<uint32_t>(c - 'A' + 10);
      else
        return error("invalid \\u escape");
    }
    return true;
  }

  bool parseString(std::string &out) {
    ++pos_; // opening quote
    while (pos_ < text_.size()) {
      const char c = text_[pos_++];
      if (c == '"') {
        return true;
      }
      if (c != '\\') {
        out += c;
        continue;
      }
      if (pos_ >= text_.size())
        break;
      const char esc = text_[pos_++];
      switch (esc) {
      case '"':
      case '\\':
      case '/':
        out += esc;
        break;
      case 'b':
        out += '\b';
        break;
      case 'f':
        out += '\f';
        break;
      case 'n':
        out += '\n';
        break;
      case 'r':
        out += '\r';
        break;
      case 't':
        out += '\t';
        break;
      case 'u': {
        uint32_t cp = 0;
        if (!parseHex4(cp))
          return false;
        if (cp >= 0xD800 && cp <= 0xDBFF && consumeLiteral("\\u")) {
          uint32_t low = 0;
          if (!parseHex4(low))
            return false;
          cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
        }
        appendUtf8(out, cp);
        break;
      }
      default:
        return error("invalid escape sequence");
      }
    }
    return error("unterminated string");
  }

  bool parseArray(Json &out, int depth) {
    ++pos_;
    out = Json::array();
    skipWhitespace();
    if (pos_ < text_.size() && text_[pos_] == ']') {
      ++pos_;
      return true;
    }
    while (true) {
      Json item;
      skipWhitespace();
      if (!parseValue(item, depth + 1))
        return false;
      out.push(std::move(item));
      skipWhitespace();
      if (pos_ >= text_.size())
        return error("unterminated array");
      if (text_[pos_] == ',') {
        ++pos_;
        skipWhitespace();
        if (pos_ < text_.size() && text_[pos_] == ']') { // trailing comma
          ++pos_;
          return true;
        }
        continue;
      }
      if (text_[pos_] == ']') {
        ++pos_;
        return true;
      }
      return error("expected ',' or ']'");
    }
  }

  bool parseObject(Json &out, int depth) {
    ++pos_;
    out = Json::object();
    skipWhitespace();
    if (pos_ < text_.size() && text_[pos_] == '}') {
      ++pos_;
      return true;
    }
    while (true) {
      skipWhitespace();
      if (pos_ >= text_.size() || text_[pos_] != '"')
        return error("expected string key");
      std::string key;
      if (!parseString(key))
        return false;
      skipWhitespace();
      if (pos_ >= text_.size() || text_[pos_] != ':')
        return error("expected ':'");
      ++pos_;
      skipWhitespace();
      Json value;
      if (!parseValue(value, depth + 1))
        return false;
      out.asObject()[std::move(key)] = std::move(value);
      skipWhitespace();
      if (pos_ >= text_.size())
        return error("unterminated object");
      if (text_[pos_] == ',') {
        ++pos_;
        skipWhitespace();
        if (pos_ < text_.size() && text_[pos_] == '}') { // trailing comma
          ++pos_;
          return true;
        }
        continue;
      }
      if (text_[pos_] == '}') {
        ++pos_;
        return true;
      }
      return error("expected ',' or '}'");
    }
  }

  std::string_view text_;
  size_t pos_ = 0;
  const char *message_ = "";
};

void dumpString(std::string &out, const std::string &s) {
  out += '"';
  for (const char c : s) {
    switch (c) {
    case '"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      if (static_cast<unsigned char>(c) < 0x20) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "\\u%04x", c);
        out += buf;
      } else {
        out += c;
      }
    }
  }
  out += '"';
}

void dumpNumber(std::string &out, double v) {
  char buf[32];
  if (v == std::floor(v) && std::fabs(v) < 1e15) {
    std::snprintf(buf, sizeof(buf), "%.1f", v);
    // Integers print without ".0" so int fields round-trip cleanly.
    std::string s(buf);
    if (s.size() > 2 && s.compare(s.size() - 2, 2, ".0") == 0)
      s.resize(s.size() - 2);
    out += s;
    return;
  }
  // Shortest text that parses back to the same value. Values that came from a
  // float (most config values) round-trip at float precision, so 0.7f prints
  // as "0.7" rather than "0.699999988".
  const bool is_float =
      std::fabs(v) <= 3.4e38 && static_cast<double>(static_cast<float>(v)) == v;
  for (int precision = 1; precision <= 17; ++precision) {
    std::snprintf(buf, sizeof(buf), "%.*g", precision, v);
    const double back = std::strtod(buf, nullptr);
    if (is_float ? static_cast<float>(back) == static_cast<float>(v) : back == v)
      break;
  }
  out += buf;
}

} // namespace

bool Json::parse(std::string_view text, Json &out, std::string *error) {
  // Skip UTF-8 BOM (files saved by some Windows editors).
  if (text.size() >= 3 && static_cast<unsigned char>(text[0]) == 0xEF &&
      static_cast<unsigned char>(text[1]) == 0xBB &&
      static_cast<unsigned char>(text[2]) == 0xBF) {
    text.remove_prefix(3);
  }
  Parser parser(text);
  return parser.run(out, error);
}

bool Json::parseFile(const std::string &path, Json &out, std::string *error) {
  std::string text;
  if (!read_text_file(path, text)) {
    if (error)
      *error = "cannot read file '" + path + "'";
    return false;
  }
  std::string parse_error;
  if (!parse(text, out, &parse_error)) {
    if (error)
      *error = path + ":" + parse_error;
    return false;
  }
  return true;
}

// SECTION: writer

std::string Json::dump(int indent) const {
  std::string out;
  dumpTo(out, indent, 0);
  out += '\n';
  return out;
}

void Json::dumpTo(std::string &out, int indent, int depth) const {
  const auto newline = [&](int d) {
    if (indent <= 0)
      return;
    out += '\n';
    out.append(static_cast<size_t>(indent * d), ' ');
  };

  switch (type_) {
  case Type::Null:
    out += "null";
    break;
  case Type::Bool:
    out += bool_ ? "true" : "false";
    break;
  case Type::Number:
    dumpNumber(out, number_);
    break;
  case Type::String:
    dumpString(out, string_);
    break;
  case Type::Array: {
    // Short arrays of scalars (colors, vectors) stay on one line.
    bool inline_array = array_.size() <= 8;
    for (const Json &v : array_)
      inline_array = inline_array && !v.isArray() && !v.isObject();
    out += '[';
    for (size_t i = 0; i < array_.size(); ++i) {
      if (i > 0)
        out += inline_array ? ", " : ",";
      if (!inline_array)
        newline(depth + 1);
      array_[i].dumpTo(out, indent, depth + 1);
    }
    if (!inline_array && !array_.empty())
      newline(depth);
    out += ']';
    break;
  }
  case Type::Object: {
    out += '{';
    size_t i = 0;
    for (const auto &[key, value] : object_) {
      if (i++ > 0)
        out += ',';
      newline(depth + 1);
      dumpString(out, key);
      out += indent > 0 ? ": " : ":";
      value.dumpTo(out, indent, depth + 1);
    }
    if (!object_.empty())
      newline(depth);
    out += '}';
    break;
  }
  }
}

// SECTION: file_io

bool read_text_file(const std::string &path, std::string &out) {
  size_t size = 0;
  void *data = SDL_LoadFile(path.c_str(), &size);
  if (!data) {
    return false;
  }
  out.assign(static_cast<const char *>(data), size);
  SDL_free(data);
  return true;
}

bool write_text_file(const std::string &path, std::string_view text) {
  SDL_IOStream *io = SDL_IOFromFile(path.c_str(), "wb");
  if (!io) {
    return false;
  }
  const size_t written = SDL_WriteIO(io, text.data(), text.size());
  const bool closed = SDL_CloseIO(io);
  return written == text.size() && closed;
}

} // namespace atm
