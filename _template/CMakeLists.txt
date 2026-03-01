# =============================================================================
# _template — per-game CMakeLists
# Replace "TemplateGame" with your game name.
# This file is auto-updated by Scripts/new-game.ps1
# =============================================================================
cmake_minimum_required(VERSION 3.21)
project(TemplateGame)

# Pull in the engine + attome_add_game() helper
# This is included automatically when building from games/CMakeLists.txt.
# No action needed here.

attome_add_game(TemplateGame
    src/main.cpp
    # add more source files here:
    # src/player.cpp
    # src/world.cpp
)
