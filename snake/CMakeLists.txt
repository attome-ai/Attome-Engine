# =============================================================================
# snake — Classic Snake game using Attome Engine / SDL3
# =============================================================================
cmake_minimum_required(VERSION 3.21)
project(Snake)

attome_add_game(Snake
    src/main.cpp
)
