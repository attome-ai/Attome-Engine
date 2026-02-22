# =============================================================================
# planet_basic — Basic version of the planet game
# =============================================================================
cmake_minimum_required(VERSION 3.21)
project(PlanetBasic)

attome_add_game(PlanetBasic
    ../engine/gameBasic/planet_game_basic.cpp
)
