// App: mining, placing, attacking, crosshair aim.

#include "app/AppInternal.h"

namespace ao::client {

// ---------------------------------------------------------------------------
// Interaction: mining, placing, attacking
// ---------------------------------------------------------------------------

// The item in hand: the selected hotbar slot when it is a weapon or tool we
// may use (same rule as ServerState::heldWeapon), else the equipped main hand.
ItemId App::equippedMainHand() const {
  const ItemStack &s = inventory_[size_t(hotbar_)];
  if (s.item && s.count > 0) {
    const ItemDef &d = itemDef(s.item);
    if (isHoldable(d.kind) && levelForXp(skillXp_[size_t(d.skill)]) >= d.levelReq)
      return s.item;
  }
  return equipped_.size() > size_t(atm::model::EquipSlot::MainHand)
             ? ItemId(equipped_[size_t(atm::model::EquipSlot::MainHand)])
             : ItemId(0);
}

void App::updateInteraction(float dt) {
  attackCooldown_ = std::max(0.0f, attackCooldown_ - dt);
  placeCooldown_ = std::max(0.0f, placeCooldown_ - dt);
  hasTarget_ = false;

  // Tell the server which hotbar slot we hold (weapon / tool in hand), and
  // show it in our own hand immediately.
  if (welcomed_ && hotbar_ != sentHotbar_) {
    proto::Equip e;
    e.inventorySlot = uint8_t(hotbar_);
    e.equipSlot = kEquipSelectHotbar;
    e.unequip = false;
    net_.send(e, atm::net2::Channel::ReliableOrdered);
    sentHotbar_ = hotbar_;
  }
  {
    const size_t mh = size_t(atm::model::EquipSlot::MainHand);
    const ItemDef &hd = itemDef(equippedMainHand());
    if (mh < selfAppearance_.pieces.size())
      selfAppearance_.pieces[mh] = hd.piece ? models_.findPiece(hd.piece) : atm::model::kNoPiece;
  }
  if (!world_ || !welcomed_ || !mouseCaptured_ || chatOpen_) {
    mineProgress_ = 0.0f;
    return;
  }

  // Aim from the camera through the crosshair; accept hits within reach of
  // the player's eye.
  const glm::dvec3 eye = eyePosition(float(accumulator_ / kSimDt));
  BlockPos hit{};
  FaceDir face = FaceDir::PosY;
  if (raycastBlock(*world_, blocks_, camPos_, aimDirection(), camDistance_ + kReach + 1.0f, hit, face)) {
    const glm::dvec3 center{hit.x + 0.5, hit.y + 0.5, hit.z + 0.5};
    const glm::dvec3 d = center - eye;
    if (std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z) <= kReach + 0.5) {
      hasTarget_ = true;
      targetBlock_ = hit;
      targetFace_ = face;
    }
  }

  const ItemId mainItem = equippedMainHand();
  const ItemDef &mainDef = itemDef(mainItem);
  const bool holdingWeapon = mainDef.kind == ItemKind::Weapon;
  const bool primary = input_.held("primary");

  // Mining: primary on a block with a tool or bare hands (not a weapon).
  // Unbreakable (hardness < 0, e.g. bedrock) or above our Mining level: the
  // server would refuse it, so don't start mining at all.
  bool breakable = false;
  if (hasTarget_) {
    const auto &tdef = blocks_.get(world_->blockAt(targetBlock_));
    breakable = tdef.hardness >= 0.0f &&
                levelForXp(skillXp_[size_t(Skill::Mining)]) >= uint32_t(tdef.miningLevel);
  }
  if (primary && breakable && !holdingWeapon) {
    if (!(miningBlock_ == targetBlock_)) {
      miningBlock_ = targetBlock_;
      mineProgress_ = 0.0f;
    }
    const auto &def = blocks_.get(world_->blockAt(targetBlock_));
    const bool pick = mainDef.weapon == WeaponType::Pickaxe;
    const float breakTime = std::max(0.08f, def.hardness * (pick ? 0.3f : 1.0f));
    const float before = mineProgress_;
    mineProgress_ += dt / breakTime;
    // Swing animation + sound about every 0.35 s while mining.
    if (int(before * breakTime / 0.35f) != int(mineProgress_ * breakTime / 0.35f) || before == 0.0f) {
      playActionAnim(selfAnim_, act::Mine, mainDef.weapon);
      if (sfx_)
        sfx_->playBlockHit(world_->blockAt(targetBlock_));
      // Chips fly off the struck face.
      static const glm::dvec3 kN[6] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
      const glm::dvec3 fn = kN[std::min(5, int(targetFace_))];
      Particles::Burst chip;
      chip.count = 5;
      chip.block = world_->blockAt(targetBlock_);
      chip.speed = 2.5f;
      chip.up = 2.5f;
      chip.size = 0.09f;
      chip.life = 0.5f;
      chip.spread = 0.3f;
      particles_.burst(glm::dvec3(targetBlock_.x + 0.5, targetBlock_.y + 0.5, targetBlock_.z + 0.5) + fn * 0.55,
                       chip);
    }
    if (mineProgress_ >= 1.0f) {
      proto::BlockAction a;
      a.action = 0;
      a.x = targetBlock_.x;
      a.y = targetBlock_.y;
      a.z = targetBlock_.z;
      a.face = uint8_t(targetFace_);
      a.hotbarSlot = uint8_t(hotbar_);
      net_.send(a, atm::net2::Channel::ReliableOrdered);
      mineProgress_ = 0.0f;
    }
  } else {
    mineProgress_ = 0.0f;
  }

  // Attacking: primary with a weapon (or no block targeted).
  if (primary && (holdingWeapon || !hasTarget_) && attackCooldown_ <= 0.0f) {
    proto::Attack a;
    // The server tick we are displaying monsters at (lag compensation).
    a.tick = uint32_t(std::max(0.0f, std::round(serverTickEstimate_ - kInterpDelayTicks)));
    // Aim from our eye (where the server spawns the shot) at whatever the
    // crosshair is on: the camera sits over the shoulder, so the raw camera
    // direction would land about a block to the side of the crosshair.
    const glm::dvec3 shotEye = prediction_.current().pos + glm::dvec3(0.0, moveTuning().eyeHeight, 0.0);
    const glm::vec3 shotDir = crosshairAim(shotEye);
    a.yaw = std::atan2(-shotDir.x, -shotDir.z);
    a.pitch = std::asin(std::clamp(shotDir.y, -1.0f, 1.0f));
    a.ability = 0;
    net_.send(a, atm::net2::Channel::ReliableOrdered);
    attackCooldown_ = weaponCooldown(mainDef.weapon); // same table as the server
    switch (mainDef.weapon) {
    case WeaponType::Bow:
    case WeaponType::Staff: {
      const bool magic = mainDef.weapon == WeaponType::Staff;
      if (magic) {
        playActionAnim(selfAnim_, act::Cast, mainDef.weapon);
        if (sfx_) sfx_->play(GameSound::SwordSwing);
      } else {
        playActionAnim(selfAnim_, act::BowShoot, mainDef.weapon);
        if (sfx_) sfx_->play(GameSound::BowShoot);
      }
      // Launch flash: sparks from in front of the chest, along the aim.
      const glm::vec3 aim = shotDir;
      Particles::Burst fl;
      fl.block = atm::voxel::blocks::Lamp;
      fl.tint = magic ? rgba(200, 140, 255) : rgba(255, 226, 140);
      fl.count = magic ? 18 : 12;
      fl.speed = 2.5f;
      fl.up = 0.5f;
      fl.size = 0.07f;
      fl.life = 0.25f;
      fl.gravity = 0.0f;
      fl.drag = 4.0f;
      fl.spread = 0.12f;
      particles_.burst(prediction_.current().pos + glm::dvec3(0.0, 1.25, 0.0) + glm::dvec3(aim) * 0.7, fl);
      break;
    }
    default: {
      // Glowing slash arc in front of the player, tinted by the blade.
      const std::string_view wn = mainDef.name;
      const uint32_t arcTint = wn.find("crystal") != std::string_view::npos ? rgba(120, 230, 255)
                               : wn.find("iron") != std::string_view::npos  ? rgba(235, 240, 255)
                                                                            : rgba(255, 220, 130);
      particles_.slashArc(prediction_.current().pos + glm::dvec3(0.0, 1.05, 0.0), camYaw_, arcTint);
    }
      playActionAnim(selfAnim_, act::Swing, mainDef.weapon);
      if (sfx_) sfx_->play(GameSound::SwordSwing);
      break;
    }
  }

  // Secondary on armour / food in the selected slot: wear it / eat it.
  if (input_.held("secondary") && placeCooldown_ <= 0.0f) {
    const ItemStack &stack = inventory_[size_t(hotbar_)];
    if (stack.item && stack.count > 0) {
      const ItemDef &d = itemDef(stack.item);
      if (d.kind == ItemKind::Armour || d.kind == ItemKind::Food) {
        proto::Equip e;
        e.inventorySlot = uint8_t(hotbar_);
        e.equipSlot = d.kind == ItemKind::Food ? kEquipConsume : uint8_t(d.slot);
        e.unequip = false;
        net_.send(e, atm::net2::Channel::ReliableOrdered);
        placeCooldown_ = 0.6f;
        playActionAnim(selfAnim_, act::Place, mainDef.weapon);
        if (sfx_) sfx_->play(GameSound::UiClick);
      }
    }
  }

  // Placing: secondary on a block face with a block item selected.
  if (input_.held("secondary") && hasTarget_ && placeCooldown_ <= 0.0f) {
    const ItemStack &stack = inventory_[size_t(hotbar_)];
    if (stack.item != 0 && stack.count > 0 && itemDef(stack.item).kind == ItemKind::Block) {
      proto::BlockAction a;
      a.action = 1;
      a.x = targetBlock_.x;
      a.y = targetBlock_.y;
      a.z = targetBlock_.z;
      a.face = uint8_t(targetFace_);
      a.hotbarSlot = uint8_t(hotbar_);
      net_.send(a, atm::net2::Channel::ReliableOrdered);
      placeCooldown_ = 0.2f;
      playActionAnim(selfAnim_, act::Place, mainDef.weapon);
    }
  }
}


} // namespace ao::client
