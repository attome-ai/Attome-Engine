# Mobile Platform Notes

The Attome Engine currently targets **Windows** and **Web (Emscripten/WASM)**.

Mobile support (Android / iOS) is planned for future milestones.

## Android (future)
- Will require the Android NDK + CMake android toolchain file.
- SDL3 has official Android support via `android/` in the SDL3 source.
- Entry point will need to use `SDL_main`.

## iOS (future)
- Requires Xcode + iOS CMake toolchain.
- SDL3 supports iOS natively.

When ready, add your platform CMakeLists here and reference it from the game's root CMakeLists.txt.
