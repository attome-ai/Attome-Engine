#include "AshlandsGame.h"
#include <iostream>

SDL_Surface *create_colored_surface(int width, int height, Uint8 r, Uint8 g,
                                    Uint8 b) {
  SDL_Surface *surface =
      SDL_CreateSurface(width, height, SDL_PIXELFORMAT_RGBA8888);
  if (!surface)
    return nullptr;
  SDL_FillSurfaceRect(
      surface, NULL,
      SDL_MapRGBA(SDL_GetPixelFormatDetails(surface->format), 0, r, g, b, 255));
  return surface;
}

// ----------------------------------------------------
// SETTLEMENT CONTAINER
// ----------------------------------------------------

SettlementContainer::SettlementContainer(int typeId, uint8_t defaultLayer,
                                         int initialCapacity)
    : RenderableEntityContainer(typeId, defaultLayer, initialCapacity),
      harvest_types(initialCapacity), harvest_levels(initialCapacity),
      structure_levels(initialCapacity), timber(initialCapacity),
      stone(initialCapacity), ore(initialCapacity), ember(initialCapacity),
      factions(initialCapacity), owners(initialCapacity),
      populations(initialCapacity), loyalties(initialCapacity),
      is_strongholds(initialCapacity), build_queues(initialCapacity),
      garrisoned_troops(initialCapacity), recruit_queues(initialCapacity) {}

EntityHandle SettlementContainer::createSettlement(float x, float y,
                                                   Faction faction,
                                                   uint32_t owner,
                                                   bool is_stronghold,
                                                   ResourceLayout layout) {
  EntityHandle id = RenderableEntityContainer::createEntity();
  if (id == INVALID_ID)
    return INVALID_ID;

  uint32_t slot = getSlot(id);
  x_positions[slot] = x;
  y_positions[slot] = y;
  factions[slot] = faction;
  owners[slot] = owner;
  is_strongholds[slot] = is_stronghold;

  timber[slot] = 1000.0;
  stone[slot] = 1000.0;
  ore[slot] = 1000.0;
  ember[slot] = 1000.0;

  std::array<uint8_t, 18> h_types = {0};
  std::array<uint8_t, 18> harvest = {0};
  int t_i = layout.timber;
  int s_i = t_i + layout.stone;
  int o_i = s_i + layout.ore;
  int e_i = o_i + layout.ember;

  for (int i = 0; i < t_i; ++i) {
    harvest[i] = 1;
    h_types[i] = 0;
  }
  for (int i = t_i; i < s_i; ++i) {
    harvest[i] = 1;
    h_types[i] = 1;
  }
  for (int i = s_i; i < o_i; ++i) {
    harvest[i] = 1;
    h_types[i] = 2;
  }
  for (int i = o_i; i < e_i; ++i) {
    harvest[i] = 1;
    h_types[i] = 3;
  }

  harvest_types[slot] = h_types;
  harvest_levels[slot] = harvest;
  structure_levels[slot] = {0};

  structure_levels[slot][(int)StructureType::CentralForge] = 1;
  populations[slot] = 19;
  loyalties[slot] = 100;

  flags[slot] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
  z_indices[slot] = 2; // Above tiles, below UI
  widths[slot] = HEX_SIZE * 1.5f;
  heights[slot] = HEX_SIZE * 1.5f;
  build_queues[slot].clear();

  return id;
}

bool SettlementContainer::checkPrerequisites(uint32_t slot,
                                             StructureType type) {
  if (slot >= count)
    return false;
  auto defs = StructureDB::getDefs();
  if ((int)type >= defs.size())
    return false;
  auto &def = defs[(int)type];

  uint8_t current_level = structure_levels[slot][(int)type];
  for (const auto &job : build_queues[slot]) {
    if (job.type == type)
      current_level = job.target_level;
  }
  if (current_level >= def.max_level)
    return false;

  for (const auto &req : def.requirements) {
    if (structure_levels[slot][(int)req.type] < req.level)
      return false;
  }

  double multiplier = std::pow(def.cost_multiplier, current_level);
  double t_cost = def.base_timber * multiplier;
  double s_cost = def.base_stone * multiplier;
  double o_cost = def.base_ore * multiplier;
  double e_cost = def.base_ember * multiplier;

  if (timber[slot] < t_cost || stone[slot] < s_cost || ore[slot] < o_cost ||
      ember[slot] < e_cost) {
    return false;
  }

  return true;
}

bool SettlementContainer::queueStructure(EntityHandle id, StructureType type) {
  uint32_t slot = getSlot(id);
  if (slot == INVALID_ID)
    return false;

  if (!checkPrerequisites(slot, type))
    return false;

  auto defs = StructureDB::getDefs();
  auto &def = defs[(int)type];

  uint8_t current_level = structure_levels[slot][(int)type];
  for (const auto &job : build_queues[slot]) {
    if (job.type == type)
      current_level = job.target_level;
  }

  double multiplier = std::pow(def.cost_multiplier, current_level);
  timber[slot] -= def.base_timber * multiplier;
  stone[slot] -= def.base_stone * multiplier;
  ore[slot] -= def.base_ore * multiplier;
  ember[slot] -= def.base_ember * multiplier;

  double time_mult = std::pow(def.time_multiplier, current_level);
  float build_time = static_cast<float>(def.base_time * time_mult);

  uint8_t forge_level =
      structure_levels[slot][(int)StructureType::CentralForge];
  if (forge_level > 0 && type != StructureType::CentralForge) {
    build_time *= std::pow(0.96f, forge_level);
  }

  build_queues[slot].push_back(
      {type, static_cast<uint8_t>(current_level + 1), build_time});
  return true;
}

bool SettlementContainer::queueRecruit(EntityHandle id, UnitType type,
                                       int amount) {
  uint32_t slot = getSlot(id);
  if (slot == INVALID_ID || amount <= 0)
    return false;

  auto defs = MilitaryDB::getUnitDef(factions[slot], type);
  double t_cost = defs.cost_timber * amount;
  double s_cost = defs.cost_stone * amount;
  double o_cost = defs.cost_ore * amount;
  double e_cost = defs.cost_ember * amount;

  if (timber[slot] < t_cost || stone[slot] < s_cost || ore[slot] < o_cost ||
      ember[slot] < e_cost) {
    return false;
  }

  timber[slot] -= t_cost;
  stone[slot] -= s_cost;
  ore[slot] -= o_cost;
  ember[slot] -= e_cost;

  int barracks_level = structure_levels[slot][(int)StructureType::Barracks];
  float time_multiplier = std::pow(0.9f, barracks_level);
  float final_time = defs.build_time * time_multiplier;

  if (!recruit_queues[slot].empty() &&
      recruit_queues[slot].back().type == type) {
    recruit_queues[slot].back().amount += amount;
  } else {
    recruit_queues[slot].push_back({type, amount, final_time});
  }

  return true;
}

void UnitContainer::resolveBattle(uint32_t attacker_unit_slot,
                                  class SettlementContainer *settlements) {
  if (attacker_unit_slot >= count || !settlements)
    return;

  uint32_t defender_slot =
      settlements->getSlot(target_settlement[attacker_unit_slot]);
  if (defender_slot == INVALID_ID)
    return;

  Faction att_faction = factions[attacker_unit_slot];
  Faction def_faction = settlements->factions[defender_slot];

  double atk_points = 0;
  double atk_infantry_ratio = 0;
  double atk_cavalry_ratio = 0;

  for (int i = 0; i < (int)UnitType::NUM_UNITS; ++i) {
    int count = troops[attacker_unit_slot].counts[i];
    if (count > 0) {
      auto def = MilitaryDB::getUnitDef(att_faction, (UnitType)i);
      double atk = def.attack * count;
      atk_points += atk;
      if (i >= (int)UnitType::CavalryTier1 && i <= (int)UnitType::CavalryTier3)
        atk_cavalry_ratio += atk;
      else
        atk_infantry_ratio += atk;
    }
  }

  if (atk_points == 0)
    return;
  atk_infantry_ratio /= atk_points;
  atk_cavalry_ratio /= atk_points;

  double def_points = 10;
  for (int i = 0; i < (int)UnitType::NUM_UNITS; ++i) {
    int count = settlements->garrisoned_troops[defender_slot].counts[i];
    if (count > 0) {
      auto def = MilitaryDB::getUnitDef(def_faction, (UnitType)i);
      double effective_def = (def.defense_infantry * atk_infantry_ratio) +
                             (def.defense_cavalry * atk_cavalry_ratio);
      def_points += effective_def * count;
    }
  }

  int wall_level =
      settlements->structure_levels[defender_slot][(int)StructureType::Rampart];
  if (wall_level > 0)
    def_points *= std::pow(1.02, wall_level);

  double K = std::pow(
      std::min(atk_points, def_points) / std::max(atk_points, def_points), 1.5);
  double atk_losses = 0;
  double def_losses = 0;

  if (atk_points > def_points) {
    atk_losses = K;
    def_losses = 1.0;

    double capacity = 0;
    for (int i = 0; i < (int)UnitType::NUM_UNITS; ++i) {
      int &count = troops[attacker_unit_slot].counts[i];
      count = std::floor(count * (1.0 - atk_losses));
      if (count > 0)
        capacity +=
            MilitaryDB::getUnitDef(att_faction, (UnitType)i).carry_capacity *
            count;
    }

    double total_res =
        settlements->timber[defender_slot] + settlements->stone[defender_slot] +
        settlements->ore[defender_slot] + settlements->ember[defender_slot];
    if (total_res > 0) {
      double plunder_ratio = std::min(1.0, capacity / total_res);
      settlements->timber[defender_slot] -=
          settlements->timber[defender_slot] * plunder_ratio;
      settlements->stone[defender_slot] -=
          settlements->stone[defender_slot] * plunder_ratio;
      settlements->ore[defender_slot] -=
          settlements->ore[defender_slot] * plunder_ratio;
      settlements->ember[defender_slot] -=
          settlements->ember[defender_slot] * plunder_ratio;
    }

    move_types[attacker_unit_slot] = MovementType::Return;
    EntityHandle temp = origin_settlement[attacker_unit_slot];
    origin_settlement[attacker_unit_slot] =
        target_settlement[attacker_unit_slot];
    target_settlement[attacker_unit_slot] = temp;
    travel_progress[attacker_unit_slot] = 0.0f;
  } else {
    atk_losses = 1.0;
    def_losses = K;
    flags[attacker_unit_slot] &= ~static_cast<uint8_t>(EntityFlag::VISIBLE);
  }

  for (int i = 0; i < (int)UnitType::NUM_UNITS; ++i) {
    int &count = settlements->garrisoned_troops[defender_slot].counts[i];
    count = std::floor(count * (1.0 - def_losses));
  }
}

void SettlementContainer::swapSlots(uint32_t a, uint32_t b) {
  if (a == b)
    return;
  std::swap(harvest_types[a], harvest_types[b]);
  std::swap(harvest_levels[a], harvest_levels[b]);
  std::swap(structure_levels[a], structure_levels[b]);
  std::swap(timber[a], timber[b]);
  std::swap(stone[a], stone[b]);
  std::swap(ore[a], ore[b]);
  std::swap(ember[a], ember[b]);
  std::swap(factions[a], factions[b]);
  std::swap(owners[a], owners[b]);
  std::swap(populations[a], populations[b]);
  std::swap(loyalties[a], loyalties[b]);
  std::swap(is_strongholds[a], is_strongholds[b]);
  std::swap(build_queues[a], build_queues[b]);
  std::swap(garrisoned_troops[a], garrisoned_troops[b]);
  std::swap(recruit_queues[a], recruit_queues[b]);
  RenderableEntityContainer::swapSlots(a, b);
}

void SettlementContainer::resizeArrays(int newCapacity) {
  if (newCapacity <= capacity)
    return;
  harvest_types.resize(newCapacity, count);
  harvest_levels.resize(newCapacity, count);
  structure_levels.resize(newCapacity, count);
  timber.resize(newCapacity, count);
  stone.resize(newCapacity, count);
  ore.resize(newCapacity, count);
  ember.resize(newCapacity, count);
  factions.resize(newCapacity, count);
  owners.resize(newCapacity, count);
  populations.resize(newCapacity, count);
  loyalties.resize(newCapacity, count);
  is_strongholds.resize(newCapacity, count);
  build_queues.resize(newCapacity, count);
  garrisoned_troops.resize(newCapacity, count);
  recruit_queues.resize(newCapacity, count);
  RenderableEntityContainer::resizeArrays(newCapacity);
}

// ----------------------------------------------------
// UNIT CONTAINER
// ----------------------------------------------------

UnitContainer::UnitContainer(int typeId, uint8_t defaultLayer,
                             int initialCapacity)
    : RenderableEntityContainer(typeId, defaultLayer, initialCapacity),
      factions(initialCapacity), owners(initialCapacity),
      origin_settlement(initialCapacity), target_settlement(initialCapacity),
      move_types(initialCapacity), troops(initialCapacity),
      travel_progress(initialCapacity), total_travel_time(initialCapacity) {}

EntityHandle UnitContainer::dispatchArmy(EntityHandle origin,
                                         EntityHandle target,
                                         MovementType mType, TroopCount counts,
                                         Faction faction, uint32_t owner,
                                         float travel_time) {
  EntityHandle id = RenderableEntityContainer::createEntity();
  if (id == INVALID_ID)
    return INVALID_ID;
  uint32_t slot = getSlot(id);
  origin_settlement[slot] = origin;
  target_settlement[slot] = target;
  move_types[slot] = mType;
  troops[slot] = counts;
  factions[slot] = faction;
  owners[slot] = owner;
  total_travel_time[slot] = travel_time;
  travel_progress[slot] = 0.0f;
  flags[slot] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
  return id;
}

void UnitContainer::swapSlots(uint32_t a, uint32_t b) {
  if (a == b)
    return;
  std::swap(factions[a], factions[b]);
  std::swap(owners[a], owners[b]);
  std::swap(origin_settlement[a], origin_settlement[b]);
  std::swap(target_settlement[a], target_settlement[b]);
  std::swap(move_types[a], move_types[b]);
  std::swap(troops[a], troops[b]);
  std::swap(travel_progress[a], travel_progress[b]);
  std::swap(total_travel_time[a], total_travel_time[b]);
  RenderableEntityContainer::swapSlots(a, b);
}

void UnitContainer::resizeArrays(int newCapacity) {
  if (newCapacity <= capacity)
    return;
  factions.resize(newCapacity, count);
  owners.resize(newCapacity, count);
  origin_settlement.resize(newCapacity, count);
  target_settlement.resize(newCapacity, count);
  move_types.resize(newCapacity, count);
  troops.resize(newCapacity, count);
  travel_progress.resize(newCapacity, count);
  total_travel_time.resize(newCapacity, count);
  RenderableEntityContainer::resizeArrays(newCapacity);
}

void UnitContainer::update(float dt) {
  if (count == 0)
    return;
  const float time_multiplier = 2.5f;
  for (int i = 0; i < count; ++i) {
    if (total_travel_time[i] > 0 &&
        (flags[i] & static_cast<uint8_t>(EntityFlag::VISIBLE))) {
      travel_progress[i] += (dt * time_multiplier) / total_travel_time[i];
    }
  }
}

void SettlementContainer::update(float dt) {
  if (count == 0)
    return;
  float time_multiplier = 2.5f; // Fast-paced 2.5x base speed

  for (int i = 0; i < count; ++i) {
    double timber_prod = 0;
    double stone_prod = 0;
    double ore_prod = 0;
    double ember_prod = 0;

    for (int f = 0; f < 18; f++) {
      int level = harvest_levels[i][f];
      if (level == 0)
        continue;
      // Base 5, * 2.0 * 1.12^level per hour
      // 2.5 multiplier
      double hourly = 5.0 * 2.0 * pow(1.12, level) * time_multiplier;
      double per_second = hourly / 3600.0;
      switch (harvest_types[i][f]) {
      case 0:
        timber_prod += per_second;
        break;
      case 1:
        stone_prod += per_second;
        break;
      case 2:
        ore_prod += per_second;
        break;
      case 3:
        ember_prod += per_second;
        break;
      }
    }

    uint8_t forge_level = structure_levels[i][(int)StructureType::CentralForge];
    if (forge_level > 0) {
      timber_prod *= (1.0 + 0.05 * forge_level);
      stone_prod *= (1.0 + 0.05 * forge_level);
      ore_prod *= (1.0 + 0.05 * forge_level);
    }

    // Burnout (Ember Consumption): 2% of population + 1.0 base per second ?
    // Travian crop consumption is 1 per pop per hour. Let's make it 1 pop = 1
    // ember / hour
    double pop_consumption = (double)populations[i] / 3600.0 * time_multiplier;

    ember_prod -= pop_consumption;

    timber[i] += timber_prod * dt;
    stone[i] += stone_prod * dt;
    ore[i] += ore_prod * dt;
    ember[i] += ember_prod * dt;

    // Capacity logic:
    double capacity_limit = 10000.0; // Base 10,000 capacity
    uint8_t vault_lvl = structure_levels[i][(int)StructureType::Vault];
    if (vault_lvl > 0)
      capacity_limit += vault_lvl * 5000.0;

    if (timber[i] > capacity_limit)
      timber[i] = capacity_limit;
    if (stone[i] > capacity_limit)
      stone[i] = capacity_limit;
    if (ore[i] > capacity_limit)
      ore[i] = capacity_limit;
    if (ember[i] > capacity_limit)
      ember[i] = capacity_limit;

    // Starvation/Burnout penalty (if Ember drops below 0)
    if (ember[i] < 0) {
      ember[i] = 0;
      // Burnout kills population slowly (1 pop per 5 minutes effectively, or
      // 300 seconds) Just simple approx for now
      if (rand() % (int)(300.0f / dt) == 0 && populations[i] > 19) {
        populations[i] -= 1;
      }
    }

    flags[i] |= static_cast<uint8_t>(EntityFlag::VISIBLE);
  }
}

void WorldMap::generateMap() {
  std::mt19937 rng(42); // Seeded for consistency
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  for (int q = -WORLD_RADIUS; q <= WORLD_RADIUS; q++) {
    int r1 = std::max(-WORLD_RADIUS, -q - WORLD_RADIUS);
    int r2 = std::min(WORLD_RADIUS, -q + WORLD_RADIUS);

    for (int r = r1; r <= r2; r++) {
      int s = -q - r;
      HexCoord coord(q, r, s);
      int dist_from_center = coord.distanceToOrigin();

      HexTile tile;
      tile.coord = coord;
      tile.type = TileType::Wasteland;
      tile.layout = generateResourceLayout(dist_from_center);

      // Generate Center
      if (dist_from_center == 0) {
        tile.type = TileType::WraithBastion;
        tile.layout = {4, 4, 4, 6};
      } else {
        float roll = dist(rng);
        if (roll < 0.04f) {
          tile.type = TileType::Ruin;
        } else if (roll < 0.14f) {
          tile.type = TileType::Rift; // 10% Rift
        }
      }

      tiles[coord] = tile;
    }
  }
  std::cout << "Generated Map with " << tiles.size() << " hexes.\n";
}

ResourceLayout WorldMap::generateResourceLayout(int distance) {
  std::mt19937 rng(distance * 1337);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  // Rare 15-Ember (Ember Heart): r 8-25
  if (distance >= 8 && distance <= 25 && dist(rng) < 0.05f) {
    return {1, 1, 1, 15};
  }
  // Uncommon 9-Ember (Ember Basin): r 25-50
  if (distance >= 25 && distance <= 50 && dist(rng) < 0.10f) {
    return {3, 3, 3, 9};
  }

  float common_roll = dist(rng);
  if (common_roll < 0.25f)
    return {4, 4, 4, 6}; // Balanced
  if (common_roll < 0.50f)
    return {3, 5, 4, 6}; // Quarry Ridge
  if (common_roll < 0.75f)
    return {3, 4, 5, 6}; // Ironvein
  return {5, 4, 3, 6};   // Timberfall
}

// ----------------------------------------------------

AshlandsGame::AshlandsGame(Engine *engine)
    : engine(engine), settlements(nullptr), units(nullptr) {}

void AshlandsGame::initialize() {
  world.generateMap();

  // Grab the already-registered containers by type id
  settlements = static_cast<SettlementContainer *>(
      engine->entityManager.containers[1].get());
  units =
      static_cast<UnitContainer *>(engine->entityManager.containers[2].get());

  // Register placeholder solid-color textures packed at different atlas
  // positions (atlas is 2048x2048 — stride 64px per texture)
  SDL_Surface *hex_surf =
      create_colored_surface(64, 64, 80, 110, 80); // muted green - wasteland
  hex_texture_id = engine_register_texture(engine, hex_surf, 0, 0, 64, 64);
  SDL_DestroySurface(hex_surf);

  SDL_Surface *rift_surf =
      create_colored_surface(64, 64, 100, 30, 180); // purple - rift
  rift_texture_id = engine_register_texture(engine, rift_surf, 64, 0, 64, 64);
  SDL_DestroySurface(rift_surf);

  SDL_Surface *ruin_surf =
      create_colored_surface(64, 64, 90, 80, 60); // brown - ruin
  ruin_texture_id = engine_register_texture(engine, ruin_surf, 128, 0, 64, 64);
  SDL_DestroySurface(ruin_surf);

  SDL_Surface *bastion_surf = create_colored_surface(
      64, 64, 220, 50, 50); // deep red - wraith bastion (center)
  bastion_texture_id =
      engine_register_texture(engine, bastion_surf, 192, 0, 64, 64);
  SDL_DestroySurface(bastion_surf);

  // Center camera on the map
  engine->camera.x = engine->world_bounds.w / 2.0f;
  engine->camera.y = engine->world_bounds.h / 2.0f;

  // Add Entities for rendering the map
  renderTiles();
}

void AshlandsGame::update(float dt) {
  if (settlements)
    settlements->update(dt);
  if (units) {
    units->update(dt);
    for (int i = 0; i < units->count; ++i) {
      if ((units->flags[i] & (uint8_t)EntityFlag::VISIBLE) &&
          units->travel_progress[i] >= 1.0f) {
        if (units->move_types[i] == MovementType::Return) {
          units->flags[i] &= ~static_cast<uint8_t>(EntityFlag::VISIBLE);
        } else {
          units->resolveBattle(i, settlements);
        }
      }
    }
  }

  // Basic camera movement
  const bool *keys = SDL_GetKeyboardState(NULL);
  float cam_speed = 500.0f * dt;
  if (keys[SDL_SCANCODE_W])
    engine->camera.y -= cam_speed;
  if (keys[SDL_SCANCODE_S])
    engine->camera.y += cam_speed;
  if (keys[SDL_SCANCODE_A])
    engine->camera.x -= cam_speed;
  if (keys[SDL_SCANCODE_D])
    engine->camera.x += cam_speed;
}

void AshlandsGame::renderTiles() {
  // Assuming type_id 0 is the basic renderable entity container
  int type_id = 0;

  for (auto &pair : world.tiles) {
    HexTile &tile = pair.second;

    float px, py;
    hex_to_pixel(tile.coord, px, py);

    EntityHandle handle = engine->entityManager.createEntity(type_id);
    tile.render_entity = handle;

    RenderableEntityContainer *renderC =
        static_cast<RenderableEntityContainer *>(
            engine->entityManager.containers[type_id].get());
    uint32_t slot = renderC->getSlot(handle);

    renderC->x_positions[slot] = engine->world_bounds.w / 2.0f + px;
    renderC->y_positions[slot] = engine->world_bounds.h / 2.0f + py;
    renderC->widths[slot] = HEX_SIZE * 2;
    renderC->heights[slot] = HEX_SIZE * 2;

    if (tile.type == TileType::Wasteland)
      renderC->texture_ids[slot] = hex_texture_id;
    else if (tile.type == TileType::Rift)
      renderC->texture_ids[slot] = rift_texture_id;
    else if (tile.type == TileType::Ruin)
      renderC->texture_ids[slot] = ruin_texture_id;
    else if (tile.type == TileType::WraithBastion)
      renderC->texture_ids[slot] = bastion_texture_id;

    renderC->z_indices[slot] = 0;
    renderC->flags[slot] |= static_cast<uint8_t>(EntityFlag::VISIBLE);

    engine->grid.add({(uint32_t)type_id, handle}, renderC->x_positions[slot],
                     renderC->y_positions[slot]);
  }
}
