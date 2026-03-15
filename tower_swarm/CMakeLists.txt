# =============================================================================
# tower_swarm — Tower defense prototype using Attome Engine / SDL3
# =============================================================================
cmake_minimum_required(VERSION 3.21)
project(TowerSwarm)

attome_add_game(TowerSwarm
  src/tower_swarm_main.cpp
  src/TowerSwarmGame.cpp
  src/InputManager.cpp
  src/CameraController.cpp
  src/entities/BaseEntity.cpp
  src/entities/CreatureContainer.cpp
  src/entities/EnemyContainer.cpp
  src/entities/ProjectileContainer.cpp
  src/entities/PickupContainer.cpp
  src/entities/TileContainer.cpp
  src/systems/PathGrid.cpp
  src/levels/WaveSpawner.cpp
  src/levels/LevelManager.cpp
  src/levels/SaveState.cpp
  src/screens/HUD.cpp
  src/shop/WaveBuffShop.cpp
  src/shop/RelicSystem.cpp
)
