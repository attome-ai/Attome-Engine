# Mobile Platform Notes

## Android ✅ Implemented

Android support is fully set up using the **Android NDK + vcpkg + Gradle** approach.

### Prerequisites (one-time install)
1. **Android Studio** — installs SDK tools
2. **NDK (Side by Side)**: Android Studio → SDK Manager → SDK Tools → NDK (Side by Side) → install r26+
3. Set `ANDROID_NDK_HOME` env variable to your NDK path
4. Set `VCPKG_ROOT` env variable to your vcpkg path

### Build
```powershell
# From the repo root:
.\Scripts\build-android.ps1 -Game snake

# For emulator (x86_64):
.\Scripts\build-android.ps1 -Game snake -ABI x86_64
```

### Project Layout
```
games/platforms/android/      ← Gradle/Android Studio project
Scripts/build-android.ps1     ← automated build + APK script
games/CMakeLists.txt          ← ANDROID target handled here
```

### How it works
- CMake cross-compiles each game as `libmain.so` (shared lib)
- SDL3's `SDLActivity` (Java) boots the native code
- `GameActivity.java` is a thin wrapper — no game logic needed there
- Assets are copied to the Gradle project automatically by CMake

---

## iOS (future)
- Requires Xcode + iOS CMake toolchain.
- SDL3 supports iOS natively.
