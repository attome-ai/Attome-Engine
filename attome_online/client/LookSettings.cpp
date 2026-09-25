#include "LookSettings.h"

#include "../../engine/ATMJson.h"

namespace ao::client {

LookSettings::LookSettings() {
  env.skyColor = glm::vec3(0.30f, 0.58f, 1.0f);
  env.fogColor = glm::vec3(0.58f, 0.76f, 0.96f); // clear blue haze, not milky white
  env.ambient = 0.46f; // sky fill: shaded sides stay readable from any angle
}

std::vector<LookSettings::Field> LookSettings::fields() {
  atm::render::Environment &e = env;
  return {
      {"exposure", "Exposure", "Image", &e.exposure, 0.3f, 1.8f, "Overall brightness before tonemapping."},
      {"contrast", "Contrast", "Image", &e.contrast, 0.0f, 1.0f, "S-curve after tonemapping. High values strain the eyes."},
      {"vibrance", "Vibrance", "Image", &e.vibrance, 0.0f, 0.8f, "Boosts muted colours (saturated ones less)."},
      {"bloom", "Bloom", "Image", &e.bloom, 0.0f, 2.0f, "Glow around bright / emissive things."},
      {"vignette", "Vignette", "Image", &e.vignette, 0.0f, 0.6f, "Darkening toward the screen corners."},

      {"sunStrength", "Sun strength", "Light", &e.sunStrength, 0.0f, 2.5f, "Direct sunlight."},
      {"ambient", "Sky light (ambient)", "Light", &e.ambient, 0.0f, 1.2f, "Fill light in the shade. Higher = softer look, less contrast."},
      {"sunAzimuth", "Sun direction", "Light", &sunAzimuth, 0.0f, 360.0f, "Compass direction of the sun (0 = north)."},
      {"sunElevation", "Sun height", "Light", &sunElevation, 5.0f, 89.0f, "Low = long shadows, golden light. High = noon."},

      {"hazeStrength", "Haze amount", "Atmosphere", &e.hazeStrength, 0.0f, 1.0f, "How much far terrain fades into the sky colour."},
      {"hazeDensity", "Haze density", "Atmosphere", &e.hazeDensity, 0.0f, 0.02f, "How quickly haze builds up with distance."},

      {"shadowSoftness", "Shadow softness", "Shadows", &e.shadowSoftness, 0.2f, 3.0f, "Width of the soft shadow edge."},

      {"aoDarkness", "Corner darkening", "Surfaces", &e.aoDarkness, 0.0f, 1.0f, "Voxel ambient occlusion in corners and creases."},
      {"tileBevel", "Block edges", "Surfaces", &e.tileBevel, 0.0f, 3.0f, "Soft bevel on block edges (tiles)."},
      {"tileGrain", "Surface grain", "Surfaces", &e.tileGrain, 0.0f, 3.0f, "Fine sub-voxel detail on block faces."},
      {"blockVariation", "Block shade variation", "Surfaces", &e.blockVariation, 0.0f, 3.0f, "Each block a slightly different shade."},
      {"colorPatches", "Colour patches", "Surfaces", &e.colorPatches, 0.0f, 3.0f, "Broad lighter / darker areas across the land."},

      {"waterReflection", "Water reflection", "Water & plants", &e.waterReflection, 0.0f, 1.0f, "Sky reflection on the water surface."},
      {"foliageGlow", "Leaf sun glow", "Water & plants", &e.foliageGlow, 0.0f, 1.5f, "Sun shining through leaves toward you."},
      {"decorDensity", "Grass & flowers", "Water & plants", &decorDensity, 0.0f, 3.0f, "Density of tufts, flowers and pebbles."},

      {"rimLight", "Character rim light", "Characters", &e.rimLight, 0.0f, 1.5f, "Bright edge that makes characters pop."},
      {"fov", "Field of view", "Camera", &fov, 50.0f, 100.0f, "Vertical field of view in degrees."},
  };
}

namespace {
void colorToJson(atm::Json &j, const char *key, const glm::vec3 &c) {
  atm::Json a = atm::Json::array();
  a.push(c.r);
  a.push(c.g);
  a.push(c.b);
  j[key] = a;
}
void colorFromJson(const atm::Json &j, const char *key, glm::vec3 &c) {
  const atm::Json *a = j.find(key);
  if (!a || !a->isArray() || a->size() < 3)
    return;
  c = glm::vec3(float(a->asArray()[0].asNumber(c.r)), float(a->asArray()[1].asNumber(c.g)),
                float(a->asArray()[2].asNumber(c.b)));
}
} // namespace

bool LookSettings::load(const std::string &path, std::string *error) {
  atm::Json root;
  if (!atm::Json::parseFile(path, root, error))
    return false;
  for (const Field &f : fields())
    if (const atm::Json *v = root.find(f.key))
      *f.value = float(v->asNumber(*f.value));
  if (const atm::Json *v = root.find("contactShadows"))
    env.contactShadows = v->asBool(env.contactShadows);
  colorFromJson(root, "skyColor", env.skyColor);
  colorFromJson(root, "fogColor", env.fogColor);
  return true;
}

bool LookSettings::save(const std::string &path) const {
  LookSettings &self = const_cast<LookSettings &>(*this); // fields() hands out pointers
  atm::Json root = atm::Json::object();
  for (const Field &f : self.fields())
    root[f.key] = *f.value;
  root["contactShadows"] = env.contactShadows;
  colorToJson(root, "skyColor", env.skyColor);
  colorToJson(root, "fogColor", env.fogColor);
  return atm::write_text_file(path, root.dump(2));
}

} // namespace ao::client
