# Tower Swarm — Full Production TODO
## Development → Enterprise → All Platforms

> **Reference:** [TowerSwarm-GDD.md](./TowerSwarm-GDD.md) — read this first
> **Engine:** ATMEngine C++20 + SDL3
> **Targets:** Steam (Windows x64, Linux x64, macOS), Web (WASM → Browser), Android (Google Play), iOS (future)
> **Last updated:** 2026-03-15

---

## Platform Matrix

| Platform | Build | Store | Status |
|---|---|---|---|
| Windows x64 | CMake + MSVC | Steam | Primary |
| Linux x64 | CMake + GCC/Clang | Steam | Primary |
| macOS arm64 + x64 | CMake + Clang | Steam | Secondary |
| Web (WASM) | Emscripten | NoobyGame (browser) | Primary |
| Android | SDL3 Android NDK | Google Play | Primary |
| iOS | SDL3 iOS | App Store | Future v2.0 |
| Steam Deck | Linux x64 (same binary) | Steam | Verified |

---

## Status Legend
```
[ ] Not started
[~] In progress
[x] Done
[!] Blocked
```

---

## Table of Contents

1. [Foundation & Tooling](#1-foundation--tooling)
2. [Native PC Build Pipeline](#2-native-pc-build-pipeline-steam-target)
2B. [Android Build Pipeline](#2b-android-build-pipeline-google-play)
2C. [Web / Browser Build Pipeline](#2c-web--browser-build-pipeline-wasm)
2D. [macOS Build Pipeline](#2d-macos-build-pipeline)
3. [Core Game — Engine Layer](#3-core-game--engine-layer)
4. [Core Game — Gameplay Systems](#4-core-game--gameplay-systems)
5. [Core Game — Characters & Enemies](#5-core-game--characters--enemies)
6. [Core Game — Shops & Economy](#6-core-game--shops--economy)
7. [Core Game — UI, HUD & Screens](#7-core-game--ui-hud--screens)
8. [Core Game — Audio](#8-core-game--audio)
9. [Art & Visual Production](#9-art--visual-production)
10. [Audio Production](#10-audio-production)
11. [Steamworks Integration](#11-steamworks-integration)
12. [Backend Infrastructure](#12-backend-infrastructure)
13. [QA & Testing](#13-qa--testing)
14. [Performance & Optimization](#14-performance--optimization)
15. [Accessibility & Localization](#15-accessibility--localization)
16. [Legal & Compliance](#16-legal--compliance)
17. [Steam Store & Marketing](#17-steam-store--marketing)
18. [Launch Operations](#18-launch-operations)
19. [Post-Launch & Live Ops](#19-post-launch--live-ops)

---

## 1. Foundation & Tooling

### 1.1 Repository & Version Control
- [ ] Initialize Git repository with `.gitignore` for C++, CMake, WASM artifacts
- [ ] Define branching strategy: `main` (stable) / `develop` (integration) / `feature/*` / `hotfix/*`
- [ ] Set up protected `main` branch (require PR + 1 review before merge)
- [ ] Create `CHANGELOG.md` with Keep-a-Changelog format
- [ ] Create `CONTRIBUTING.md` with coding standards
- [ ] Set up Git LFS for binary assets (sprites, audio, fonts)
- [ ] Create `.editorconfig` for consistent formatting (indent, line endings)
- [ ] Add `clang-format` config (`.clang-format`) for C++ style enforcement

### 1.2 Build System
- [ ] Root `CMakeLists.txt`: separate targets for `TowerSwarm_Native`, `TowerSwarm_WASM`
- [ ] `CMakePresets.json`: presets for Debug / RelWithDebInfo / Release / WASM
- [ ] `vcpkg.json` manifest for all C++ dependencies (SDL3, Steamworks, spdlog, fmt, nlohmann-json)
- [ ] Windows build: MSVC 2022 + Ninja generator confirmed working
- [ ] Linux build: GCC 13+ / Clang 17+ confirmed working
- [ ] WASM build: Emscripten 3.x + SDL3 port confirmed working
- [ ] CMake install target: packages game + assets into distributable folder
- [ ] Cross-compilation docs: how to build Linux target from Windows (WSL2 toolchain)

### 1.3 CI/CD Pipeline (GitHub Actions)

**Per-PR jobs (fast feedback):**
- [ ] **Build: Windows x64** — MSVC + Ninja, Debug + Release
- [ ] **Build: Linux x64** — GCC 13, Debug + Release
- [ ] **Build: WASM** — Emscripten, Release only
- [ ] **Build: Android arm64** — NDK r26, assembleRelease
- [ ] **Test job** — unit tests on Linux Release, fail PR if tests fail
- [ ] **Static analysis** — `clang-tidy` on changed C++ files

**Nightly jobs:**
- [ ] **Build: macOS universal** — Xcode 15, arm64 + x86_64, lipo into universal binary
- [ ] **Memory check** — `valgrind` on Linux Debug build, fail if leak detected
- [ ] **Performance benchmark** — headless Level 1–30 run, assert FPS ≥ 60

**Release jobs (triggered on `main` tag `vX.Y.Z`):**
- [ ] **Steam upload** — Windows + Linux + macOS → SteamPipe `default` branch
- [ ] **WASM deploy** — WASM bundle → NoobyGame CDN
- [ ] **Android upload** — AAB → Google Play Internal Testing track
- [ ] **GitHub Release** — create release with changelog + all build artifacts attached
- [ ] **Notify** — Discord webhook on release success/failure

Build artifacts: auto-upload ZIP of all builds per CI run (90-day retention)

### 1.4 Crash Reporting
- [ ] Integrate **Crashpad** (cross-platform crash handler for native builds)
- [ ] Configure crash dump upload endpoint (self-hosted or Sentry)
- [ ] Add `--crash-handler` subprocess launch in `main()`
- [ ] Symbol upload step in CI (`.pdb` on Windows, DWARF on Linux)
- [ ] Crash dashboard: able to see top crashes by frequency, OS, build version
- [ ] In-game crash dialog: "Game crashed. Send report?" with opt-in telemetry consent

### 1.5 Analytics & Telemetry
- [ ] Define event schema: `game_start`, `level_start`, `level_complete(level, stars, time)`, `level_fail`, `creature_evolved(tier)`, `merge_done`, `shop_purchase(item, cost)`, `session_end`
- [ ] Implement lightweight event queue (fire-and-forget HTTP POST to analytics endpoint)
- [ ] GDPR-compliant opt-in: first-launch consent dialog (EU players)
- [ ] Analytics backend: ingest endpoint → PostgreSQL events table → Grafana dashboard
- [ ] Funnel analysis: level 1 start → level 1 complete → level 5 → level 10 (retention funnel)

### 1.6 Developer Tooling
- [ ] In-game debug console (toggle with backtick key): `set_level N`, `give_essence N`, `set_tier N`, `skip_wave`, `god_mode`
- [ ] Debug overlay (F1): FPS, entity counts, memory usage, current wave stats
- [ ] Level editor stub: ability to hand-author level definitions as JSON (used by `LevelDefinition` loader)
- [ ] Asset hot-reload: changes to sprite files reflected without recompile (debug builds only)
- [ ] Headless mode: `--headless --run-to-level 30` for automated smoke tests

---

## 2. Native PC Build Pipeline (Steam Target)

### 2.1 Window & Display
- [ ] `DisplayManager`: detect available resolutions, apply fullscreen/windowed/borderless
- [ ] Default to native resolution fullscreen on first launch
- [ ] Resolution options: 1280×720, 1600×900, 1920×1080, 2560×1440, 3840×2160
- [ ] UI scaling: render at 1080p internally, scale to display resolution
- [ ] HiDPI support (4K monitors): UI elements scale × 2 at 4K
- [ ] Window icon: set `SDL_SetWindowIcon` with Tower Swarm logo
- [ ] Alt+Enter toggles fullscreen/windowed
- [ ] Save display settings to `settings.ini` (persist across sessions)

### 2.2 Input System (Native)
- [ ] Keyboard: all actions rebindable
- [ ] Mouse: left click = place/select, right click = sell/cancel, scroll = zoom
- [ ] Gamepad: full controller support via SDL3 gamepad API
  - [ ] D-pad / left stick: navigate menus and camera
  - [ ] A/Cross: confirm / place creature
  - [ ] B/Circle: cancel / sell
  - [ ] Y/Triangle: open shop / inter-level screen
  - [ ] Shoulder buttons: cycle creature types
  - [ ] Start: pause
- [ ] Gamepad cursor: virtual cursor driven by right stick for precise creature placement
- [ ] Input config saved to `settings.ini`
- [ ] Key rebinding screen in Settings menu

### 2.3 File System & Save
- [ ] Save directory: `%APPDATA%\TowerSwarm\` (Windows), `~/.local/share/TowerSwarm/` (Linux)
- [ ] Save file: `save.json` (see GDD Section 15 for schema)
- [ ] Settings file: `settings.ini` (display, audio, input, language)
- [ ] Backup save: rotate last 3 save files (`save.json`, `save.bak1.json`, `save.bak2.json`)
- [ ] Auto-save: after every level complete and inter-level shop close
- [ ] Manual save: save button in pause menu
- [ ] Save corruption guard: validate JSON on load, fall back to `.bak1` if corrupt
- [ ] Steam Cloud save sync (see Section 11)

### 2.4 Asset Loading (Native)
- [ ] Asset directory: `./assets/` relative to executable
- [ ] `AssetManager`: load PNG sprites into SDL_Surface → texture atlas at startup
- [ ] `AudioManager`: load OGG files into SDL3 audio streams at startup
- [ ] `FontManager`: load TTF fonts at startup (pixel font + UI font)
- [ ] Loading screen with progress bar during asset load
- [ ] Asset manifest: `assets/manifest.json` lists all required files (validate on startup, abort with error if missing)
- [ ] Asset pack: CI bundles all assets into game directory for distribution
- [ ] Asset versioning: manifest includes hash per file (detect corrupted installs)

### 2.5 Packaging & Distribution (SteamPipe)
- [ ] `steampipe/app_build.vdf`: app build script for Steam upload
- [ ] `steampipe/depot_windows.vdf`: Windows x64 depot definition
- [ ] `steampipe/depot_linux.vdf`: Linux x64 depot definition
- [ ] Steam launch options configured: `TowerSwarm.exe` with no args required
- [ ] SteamPipe upload step in CI: `steamcmd +login ... +run_app_build ... +quit`
- [ ] Beta branch in Steamworks: `default` (public), `beta` (testers), `dev` (internal)
- [ ] Windows redistributables: MSVC runtime bundled via VC++ Redist installer or static linking

---

---

## 2B. Android Build Pipeline (Google Play)

### 2B.1 Android Build Setup
- [ ] SDL3 Android port: confirm SDL3 `android-project/` template compiles
- [ ] NDK version: NDK r26+ required for C++20 support — lock in `local.properties`
- [ ] `CMakeLists.txt` Android target: `add_library(TowerSwarm SHARED ...)` (SDL3 requires shared library)
- [ ] `build.gradle` configured: `minSdk 26` (Android 8.0+), `targetSdk 34`, `abiFilters "arm64-v8a", "x86_64"`
- [ ] Asset packaging: copy `assets/` into Android `app/src/main/assets/` during build
- [ ] Gradle build: `./gradlew assembleRelease` produces signed AAB
- [ ] CI: Android build job in GitHub Actions using `ubuntu-latest` + Android SDK action
- [ ] Keystore: signing keystore stored in GitHub Secrets, never committed
- [ ] Google Play Internal Testing track: auto-deploy from CI on `main` tag

### 2B.2 Android Display & Resolution
- [ ] SDL3 window creation: request native resolution, `SDL_WINDOW_FULLSCREEN`
- [ ] Forced landscape: `android:screenOrientation="sensorLandscape"` in `AndroidManifest.xml`
- [ ] Safe area insets: detect notch + navigation bar insets, offset HUD elements
- [ ] UI scaling: auto-scale for phone (360dp) vs tablet (600dp+) — tablets show more of world
- [ ] Density-aware rendering: render at 720p logic, scale to device DPI via SDL3

### 2B.3 Android Touch Controls
- [ ] Single tap on empty cell: open creature type picker → confirm → place
- [ ] Single tap on creature: select (show creature panel)
- [ ] Long press on creature: drag to reposition
- [ ] Double tap on creature: sell (confirm dialog)
- [ ] Two-finger drag: pan camera
- [ ] Pinch: zoom in/out
- [ ] Two-finger double-tap: zoom to fit all creatures
- [ ] Virtual D-pad (optional, togglable in settings): on-screen directional for camera pan
- [ ] Touch target sizes: all interactive elements minimum 48×48dp (Material Design guideline)
- [ ] Touch feedback: ripple effect on button press, haptic vibration (50ms) on creature placed/evolved/merged
- [ ] Wave buff shop: full-screen card swipe (swipe up to pick a card)

### 2B.4 Android Performance
- [ ] Target devices: Snapdragon 7xx (mid-range) at 60 FPS, Snapdragon 4xx (entry) at 30 FPS
- [ ] GPU: use OpenGL ES 3.2 renderer (SDL3 selects automatically)
- [ ] RAM budget: 300MB for game + assets (Android kills apps over ~500MB on mid-range)
- [ ] Texture compression: ASTC for Android (use `.astc.ktx` sprites instead of PNG, bake at build time)
- [ ] Audio: use SDL3 audio, `AAudio` backend (low-latency, default on Android 8+)
- [ ] Battery: cap render loop to 60 FPS using `SDL_DelayNS`; do not run at uncapped speed
- [ ] Thermal throttle: detect `BATTERY_PLUGGED_*` intent; drop to 30 FPS target when unplugged + hot

### 2B.5 Android Lifecycle
- [ ] `SDL_APP_WILLENTERBACKGROUND`: pause game, pause audio, save state snapshot
- [ ] `SDL_APP_DIDENTERFOREGROUND`: resume game, restore audio, confirm save state
- [ ] `SDL_APP_TERMINATING`: flush save state to disk immediately
- [ ] Handle phone call interruption: same as background (pause + save)
- [ ] Screen wake lock: `SDL_DisableScreenSaver()` during active gameplay, re-enable on pause screen

### 2B.6 Android Input (Hardware)
- [ ] Physical back button: opens pause menu (do NOT exit game)
- [ ] Physical keyboard (if attached): same bindings as desktop
- [ ] Gamepad via Bluetooth: same SDL3 gamepad mappings as desktop

### 2B.7 Google Play Store
- [ ] Google Play Developer account ($25 one-time)
- [ ] App bundle (AAB) — not APK — required for Play Store 2024+
- [ ] App signing: enroll in Google Play App Signing (upload key + signing cert)
- [ ] Store listing: title, short description (80 chars), full description (4000 chars)
- [ ] Screenshots: phone (min 2, max 8) + 7-inch tablet (min 1) + 10-inch tablet (min 1)
- [ ] Feature graphic: 1024×500px
- [ ] App icon: 512×512px PNG (round + adaptive icon for Android 8+)
- [ ] Content rating: complete questionnaire → expected rating IARC 7+ / Everyone
- [ ] Privacy policy URL: required for Play Store submission
- [ ] Data safety form: declare what data is collected (Steam ID, analytics if opted-in)
- [ ] Free + IAP model: base game free OR premium $2.99 (decide before submission)
- [ ] Google Play Games sign-in: integrate for cloud saves + achievements on Android
- [ ] Release tracks: Internal → Closed Testing → Open Testing → Production

### 2B.8 Google Play Games Integration
- [ ] Google Play Games SDK: `com.google.android.gms:play-services-games-v2`
- [ ] Sign-in: `PlayGames.getGamesSignInClient()` — silent sign-in on app start
- [ ] Achievements: mirror Steam achievement list (40 achievements defined in Play Console)
- [ ] Leaderboards: `PlayGames.getLeaderboardsClient()` — "Max Level Reached", "Daily Challenge"
- [ ] Cloud saves: `PlayGames.getSnapshotsClient()` — sync `save.json` (same schema as Steam Cloud)
- [ ] Conflict resolution: highest `max_level` wins on cloud conflict

### 2B.9 Android Monetization (if free-to-play on mobile)
- [ ] In-app purchase: "Shard Pack" options (500 / 1200 / 2500 Shards)
- [ ] Google Play Billing Library 6+: `BillingClient.Builder`
- [ ] Purchase verification: server-side Google Play Developer API validation (no client trust)
- [ ] Receipt storage: `tower_swarm_purchases(user_id, order_id, product_id, amount, verified_at)`
- [ ] Restore purchases: "Restore" button in settings (required for compliance)
- [ ] Refund handling: Google Play refunds → webhook → revoke Shards (grace period 7 days)
- [ ] Note: if premium price on mobile ($2.99), IAP removed entirely

---

## 2C. Web / Browser Build Pipeline (WASM)

### 2C.1 Build
- [ ] Emscripten 3.x: `emcmake cmake ... -DCMAKE_BUILD_TYPE=Release`
- [ ] Output: `tower_swarm.wasm` + `tower_swarm.js` (ES module) + `tower_swarm.data` (assets)
- [ ] Asset pack: `--preload-file assets/` → all assets embedded in `.data` file
- [ ] Memory: `-sINITIAL_MEMORY=536870912` (512MB), `-sALLOW_MEMORY_GROWTH=1`
- [ ] WASM SIMD: `-msimd128` for 20–40% speedup on supported browsers
- [ ] Multithreading: `-sUSE_PTHREADS=1` with SharedArrayBuffer (requires COOP/COEP headers)
- [ ] Optimizations: `-O3 -flto` for release builds
- [ ] Binary size: target < 8MB for `.wasm` compressed — profile and strip unused code

### 2C.2 Angular NoobyGame Integration
- [ ] `tower-swarm-wasm.constants.ts`: WASM path, canvas ID, loading config
- [ ] `game-catalog.constants.ts`: entry for Tower Swarm (name, route, thumbnail, description)
- [ ] `tower-swarm-game.component.ts`: Angular component wrapping WASM canvas
  - [ ] Lifecycle: `ngOnInit` → download WASM → init → `ngOnDestroy` → cleanup
  - [ ] Resize observer: adapt canvas to container size on window resize
  - [ ] Visibility API: pause WASM loop when tab is hidden
- [ ] Route: `/games/tower-swarm` with lazy-loaded module
- [ ] Loading screen: Angular spinner + progress bar while WASM + assets download
- [ ] Error boundary: show "Failed to load game. Try refreshing." if WASM init fails

### 2C.3 Browser-Specific Input
- [ ] Keyboard events: `SDL_EVENT_KEY_DOWN/UP` wired via Emscripten event bridge
- [ ] Mouse events: all SDL3 mouse events mapped
- [ ] Touch events: `SDL_EVENT_FINGER_DOWN/UP/MOTION` — same handling as Android touch
- [ ] Context menu: `event.preventDefault()` on right-click to suppress browser menu
- [ ] Fullscreen: `canvas.requestFullscreen()` bound to F11 or button in HUD
- [ ] Pointer lock: optional for precise camera drag

### 2C.4 Browser Save State
- [ ] `localStorage`: persist `save.json` string (key: `tower_swarm_save`)
- [ ] Size limit: `localStorage` 5–10MB per origin — save.json < 50KB so safe
- [ ] Auto-save after level complete + inter-level shop close
- [ ] "Import save" / "Export save" buttons: let users back up their save file
- [ ] Cross-device sync: logged-in users get cloud save sync via backend API

### 2C.5 Web Hosting & CDN
- [ ] Host on NoobyGame platform subdomain: `noobygame.attome.com/games/tower-swarm`
- [ ] COOP/COEP headers required for SharedArrayBuffer (WASM threads):
  - [ ] `Cross-Origin-Opener-Policy: same-origin`
  - [ ] `Cross-Origin-Embedder-Policy: require-corp`
- [ ] Brotli compression: serve `.wasm` with `Content-Encoding: br` (50–60% size reduction)
- [ ] CDN caching: `Cache-Control: public, max-age=31536000, immutable` for versioned assets
- [ ] Service Worker: cache WASM + assets for offline play after first load

### 2C.6 Browser Performance Targets
- [ ] Chrome 120+ (desktop): 60 FPS through Level 60
- [ ] Firefox 120+ (desktop): 60 FPS through Level 40
- [ ] Safari 17+ (desktop/iOS): 30 FPS through Level 30 (Safari WebAssembly slower)
- [ ] Chrome Android: 30 FPS through Level 20
- [ ] Load time: < 5 seconds on 50 Mbps connection (WASM + assets < 25MB total)

---

## 2D. macOS Build Pipeline

### 2D.1 macOS Build Setup
- [ ] Xcode 15+ with macOS 14 SDK
- [ ] CMake target: `arm64` (Apple Silicon) + `x86_64` (Intel) → universal binary via `lipo`
- [ ] SDL3 macOS: Metal renderer (SDL3 defaults to Metal on macOS — fast)
- [ ] App bundle: `TowerSwarm.app` with correct `Info.plist`
- [ ] Code signing: Apple Developer Program ($99/year), sign with Developer ID cert
- [ ] Notarization: Apple notarization required for distribution outside Mac App Store
  - [ ] `xcrun notarytool submit TowerSwarm.zip --apple-id ... --team-id ...`
  - [ ] Notarization CI step: notarize on every release build
- [ ] Hardened runtime: `-runtime` flag required for notarization
- [ ] Gatekeeper: test that app opens without "unidentified developer" warning

### 2D.2 macOS-Specific Input
- [ ] Scroll wheel: two-finger trackpad scroll → camera pan (mapped to SDL scroll events)
- [ ] Pinch gesture: zoom via `NSGestureRecognizer` → SDL custom event
- [ ] Right-click: `Ctrl+click` on Magic Mouse is right-click — test this
- [ ] Cmd+Q: quit game (send `SDL_EVENT_QUIT`)
- [ ] Cmd+M: minimize window
- [ ] Cmd+F: toggle fullscreen (macOS convention, in addition to F11)

### 2D.3 macOS Steam Distribution
- [ ] macOS depot in Steamworks: `depot_macos.vdf`
- [ ] Universal binary uploaded to Steam macOS depot
- [ ] Test on both Apple Silicon (M-series) and Intel Mac
- [ ] Steam Overlay: confirm works on macOS (Metal renderer)

---

## 3. Core Game — Engine Layer

### 3.1 Entity Containers
- [ ] `CreatureContainer` — Hybrid, all SoA arrays from GDD §15 + `swapSlots` + `resizeArrays`
- [ ] `EnemyContainer` — Dynamic, all SoA arrays
- [ ] `ProjectileContainer` — Dynamic, pooled (10,000 slots pre-allocated)
- [ ] `ParticleContainer` — Dynamic, pooled (50,000 slots pre-allocated)
- [ ] `PickupContainer` (essence orbs) — Dynamic, pooled (5,000 slots)
- [ ] `TileContainer` — Static, tilemap rendering
- [ ] `WallContainer` — Static, destructible walls
- [ ] `BaseEntity` — Static, single instance

### 3.2 World & Camera
- [ ] `CameraController`: pan (WASD + middle-click drag), zoom (scroll wheel), clamp to world bounds
- [ ] Smooth lerp on camera movement (`lerp_factor = 8.0f * dt`)
- [ ] Camera follows cursor when at screen edge (edge-scroll, configurable sensitivity)
- [ ] Mini-map (bottom-right corner): shows world overview, all creatures + enemies, camera viewport rect
- [ ] Camera zoom: 0.5× to 2.0× range, smooth scale
- [ ] World size: 5120×2880 (4× window at 1280×720)

### 3.3 Spatial Grid & Queries
- [ ] Confirm `queryCircle` works correctly for creature attack range
- [ ] Confirm `queryRect` works for camera-frustum culling
- [ ] Verify grid updates on creature repositioning
- [ ] Benchmark: 100,000 entities, queryCircle per creature per frame → must be < 1ms total

### 3.4 Rendering
- [ ] Texture atlas: all sprites packed into single 2048×2048 atlas at startup
- [ ] Batch render: all entities grouped by texture + z-index per frame
- [ ] Z-index assignment: tiles(0–10), walls(11–20), enemies(21–30), creatures(31–40), projectiles(41–50), particles(51–60), pickups(61–70), UI(200–255)
- [ ] Additive blend pass for glow effects (high-tier creatures, base)
- [ ] SDL3 renderer target: hardware-accelerated (GPU), fall back to software
- [ ] VSync toggle in settings (default on)

### 3.5 PathGrid (A* on 64px grid)
- [ ] Grid dimensions: `world_width / 64` × `world_height / 64`
- [ ] Walkable cell map: updated when walls placed/destroyed
- [ ] A* implementation: returns list of cell waypoints
- [ ] Path caching: enemy type paths cached per spawn-point → base (recalculated only on wall change)
- [ ] Path validation before wall placement: reject wall if it fully blocks base path
- [ ] Flyer type: bypasses PathGrid entirely

---

## 4. Core Game — Gameplay Systems

### 4.1 Level Manager
- [ ] State machine: `MAIN_MENU` / `LEVEL_SELECT` / `PRE_LEVEL` / `PLAYING` / `WAVE_CLEAR` / `LEVEL_CLEAR` / `LEVEL_FAILED`
- [ ] `LevelScaler::generate(N)` → `LevelDefinition` (all formulas from GDD §4)
- [ ] Wave spawner: dequeue enemies per wave, respects `inter_spawn_delay`
- [ ] Between-wave grace timer: `max(3, 8 - floor(level/10))` seconds
- [ ] Level complete → calculate stars, trigger `LEVEL_CLEAR`
- [ ] Level fail → `LEVEL_FAILED`, preserve pre-level save snapshot for retry
- [ ] `SaveState::snapshot()` called at level start (retry restores this)
- [ ] Level milestone events: hand-authored for levels 1, 5, 10, 25, 50, 100

### 4.2 Combat System
- [ ] Creature target acquisition: `queryCircle(x, y, range)` → pick nearest alive enemy
- [ ] Re-target: if `target_id` handle invalid or out of range → re-acquire next frame
- [ ] Attack: fire projectile toward target, reset `attack_cd`
- [ ] Projectile hit: `queryCircle(proj_x, proj_y, 8px)` → apply `damage` to all hits up to `pierce` count
- [ ] Damage application: `enemy_hp -= damage`, death check, essence drop, kill credit
- [ ] Base damage: enemy reaching base calls `BaseHealthSystem::take_damage(dmg)`, triggers fail check

### 4.3 Creature Movement AI
- [ ] Staggered scheduler: each creature recalculates every 3s, stagger start times to spread load
- [ ] Threat vector: weighted sum of enemy positions within 200/400/600px radii
- [ ] Support vector: mild repulsion from other creatures within 96px
- [ ] Desired position: clamp to valid grid cells (not base zone, not occupied, not wall)
- [ ] If desired position significantly different (> 64px) → enter `MOVING` state
- [ ] Movement via A*: find path to desired cell, advance along waypoints
- [ ] Smooth interpolation over 1.5 seconds per waypoint
- [ ] No attacking while in `MOVING` state
- [ ] Player drag override: click + drag creature to new cell, 0.5s stun on drop

### 4.4 Evolution System
- [ ] Kill tracking per creature slot (cumulative across all levels)
- [ ] Check threshold: `floor(10 × 2.5^(tier-1))` kills needed per tier
- [ ] On threshold reached: `state = EVOLVING`, `evolve_timer = 0`
- [ ] Evolve animation: 0.8s pulse (1.0× → 1.5× → new_size), color shift to tier color
- [ ] Stat recalculation at tier change (all 5 stats from GDD formulas)
- [ ] Floating text: "[Character] → [Stage Name] TIER N"
- [ ] Screen-edge glow: 1-second colored flash
- [ ] Sound: tier-appropriate fanfare
- [ ] Track evolutions-this-level for inter-level stats display

### 4.5 Merge System
- [ ] Every 2 seconds: scan for adjacent same-type + same-tier pairs via `queryRect`
- [ ] Eligible pair: show pulsing amber link between them
- [ ] Auto-merge: trigger if both in `IDLE` state for 6 consecutive seconds
- [ ] Manual merge: drag one creature onto eligible adjacent creature
- [ ] Merge animation: both slide to midpoint (0.8s), flash white, new creature appears
- [ ] New creature: `kills = (kills_a + kills_b) / 2`, check if threshold already met
- [ ] Merge reward: +10 essence, +1 "merges_this_level" counter
- [ ] Merge blocked when: `ATTACKING`, `MOVING`, `EVOLVING`, or `MERGING` state

### 4.6 Essence & Pickup System
- [ ] `PickupEntity` spawned at enemy death position: floats upward 30px over 2s, then auto-collected
- [ ] Auto-collection on approach: if any creature within 80px, pickup flies toward it and is collected
- [ ] Essence balance stored in `GameState::essence`
- [ ] Interest: end of level if essence ≥ 100, grant +5% rounded up
- [ ] Placement cost check: grey out creature seeds in selector if insufficient essence
- [ ] Sell: right-click creature → confirm dialog → remove entity, restore 50% of seed cost

### 4.7 Wave Buff Shop (In-Gameplay)
- [ ] Card pool: 12 cards defined in `WaveBuffShop::CARD_POOL[]`
- [ ] Draw: 3 unique random cards per wave clear (no duplicates in same draw)
- [ ] UI: slide up from bottom, show 3 cards with name + description + icon
- [ ] Timer: grace period countdown shown, shop closes when timer hits 0 (card chosen or skipped)
- [ ] Selected card: apply `BuffEffect` to `GameState::active_buffs[]`
- [ ] Buff effects tick per frame or per event, expire after their `duration_waves` count
- [ ] "Skip" option: close shop without choosing (no penalty)

---

## 5. Core Game — Characters & Enemies

### 5.1 Character Definition System
- [ ] `CharacterDefinition` struct: id, name, rarity, base_stats, stage_names[3], stage_abilities[3], signature_ability, upgrade_nodes[5]
- [ ] `CharacterDefinitions.h`: all 10 characters fully defined in code
- [ ] `CharacterRoster`: player's owned characters, upgrade ranks, current kills, current tier
- [ ] Character unlock check: `UnlockSystem::is_unlocked(character_id)` queries save state
- [ ] Locked characters shown in selector with lock icon + "Unlocks at Level N" tooltip

### 5.2 All 10 Characters Implemented
- [ ] **Brix** — Shooter — `attack_range=220, base_dmg=12, rate=1.5/s`
  - [ ] Signature: Avalanche (15s CD — boulder knockback line 150px)
  - [ ] Stage 4 perk: shots pierce 1 enemy
  - [ ] Stage 7 perk: shots pierce 3, +20% range
  - [ ] Stage 10 perk: shots detonate on impact (60px splash)
- [ ] **Flara** — Splasher — `attack_range=180, base_dmg=8, rate=0.8/s, splash_radius=80px`
  - [ ] Signature: Conflagration (20s CD — 300px firestorm, 5× damage)
  - [ ] Stage 4 perk: burning ground 2s AoE after hit
  - [ ] Stage 7 perk: burning ground 4s + slow
  - [ ] Stage 10 perk: 3 simultaneous blast targets
- [ ] **Mossling** — Support — `aura_radius=96px, aura_attack_speed=+5%, aura_damage=0%`
  - [ ] Signature: Overgrowth (25s CD — resets attack CDs of all nearby creatures)
  - [ ] Stage 4: aura +10% atk speed + +8% damage
  - [ ] Stage 7: aura heals creatures 2 HP/s
  - [ ] Stage 10: aura radius 200px + slows nearby enemies
- [ ] **Glitch** — Trapper — `slow_field_radius=60px, slow_amount=50%, slow_duration=3s`
  - [ ] Signature: System Crash (18s CD — freeze all enemies 250px for 2.5s)
  - [ ] Stage 4: orbs reduce enemy damage output
  - [ ] Stage 7: orbs detonate after 4s, burst damage
  - [ ] Stage 10: orbs chain on detonate
- [ ] **Ironjaw** — Charger — `charge_range=300px, charge_dmg=30, knockback=80px`
  - [ ] Signature: Override (22s CD — 4s frenzy, 3× attack speed + unlimited movement)
  - [ ] Stage 4: charge hits 3 enemies in line
  - [ ] Stage 7: charge leaves shockwave trail
  - [ ] Stage 10: charge is a rampage, hits all in path
- [ ] **Wraith** — Sniper — `attack_range=500px, base_dmg=40, rate=0.3/s`
  - [ ] Signature: Death Mark (30s CD — marked enemy dies in 4s regardless of HP)
  - [ ] Stage 4: arrows ignore 30% armor
  - [ ] Stage 7: instakill enemies below 15% HP
  - [ ] Stage 10: kills chain bolt to nearest enemy
- [ ] **Crystalis** — Hybrid — `attack_range=280, base_dmg=15, aura_range_boost=+15%`
  - [ ] Signature: Prismatic Nova (20s CD — 360° beam burst hits all onscreen enemies)
  - [ ] Stage 4: beams refract to 2 targets
  - [ ] Stage 7: beams refract to 4 targets
  - [ ] Stage 10: beams bounce infinitely until enemy dies
- [ ] **Vex** — Chaos — `random_ability_interval=5s, ability_pool_size=3→6→9`
  - [ ] Signature: Entropy Storm (25s CD — all random abilities fire simultaneously)
  - [ ] Stage 4: +2 abilities to pool (lightning strike, clone 3s)
  - [ ] Stage 7: all abilities 2× stronger
  - [ ] Stage 10: abilities chain (each triggers next)
- [ ] **Orin** — Titan — `passive_base_shield_chance=5%→15%→25%`
  - [ ] Signature: Temporal Ward (60s CD — all enemies frozen 5s, creatures continue attacking)
  - [ ] Stage 6: shield upgrades to 15%, emits damage aura
  - [ ] Stage 10: 25% shield + revive 1 creature per level
  - [ ] Unlock gate: Level 50 3-star OR 500 Shards
- [ ] **Null** — Nullifier — `drain_radius=180px, drain_damage=10%→25%`
  - [ ] Signature: Consumption (45s CD — absorbs nearest enemy, gains HP)
  - [ ] Stage 6: drains 25% damage + 15% speed from enemies
  - [ ] Stage 10: enemies in range deal 0 damage; their kills credit Null
  - [ ] Unlock gate: Level 100 OR 800 Shards

### 5.3 All 8 Enemy Types Implemented
- [ ] **Grub** — `hp=30, speed=90, reward=5` — direct vector path
- [ ] **Hulk** — `hp=250, speed=35, reward=20` — 50% frontal damage reduction
- [ ] **Scuttle** — `hp=12, speed=110, reward=2` — spawns in packs of 15–30
- [ ] **Driftwing** — `hp=60, speed=70, reward=12` — ignores PathGrid, direct vector
- [ ] **Divide** — `hp=80, speed=55, reward=15` — on death: spawn 2 children at 40% HP
- [ ] **Vanguard** — `hp=150, speed=50, reward=18` — front shield: 80% resist (track facing direction)
- [ ] **Mender** — `hp=40, speed=40, reward=10` — heal 8 HP/s to all allies within 120px
- [ ] **Siege Lord (Boss)** — `hp=50×base, phases=3` — phase triggers at 66%/33%
  - [ ] Phase 1→2: spawn 20 Grubs, +30% speed
  - [ ] Phase 2→3: spawn 10 Hulks, charge behavior toward base
  - [ ] Biome visual variants: Moss / Ember / Frost / Infernal / Void (5 boss skins)
- [ ] New-enemy introduction: 2s pause on first encounter, zoom + banner, description line
- [ ] Enemy AI branching: `switch(enemy_type)` in `EnemyAI::update()`

---

## 6. Core Game — Shops & Economy

### 6.1 Wave Buff Shop (12 cards)
- [ ] Implement all 12 buff cards with correct `BuffEffect` implementations:
  - [ ] Surge: `all_creatures_attack_speed *= 1.25f` for 4 waves
  - [ ] Fortify: `base_hp += 15` one-time
  - [ ] Frenzied Blood: `essence_per_kill += 1` for 3 waves
  - [ ] Slow Tide: `next_wave_enemy_speed *= 0.65f`
  - [ ] Foresight: clear next wave's elite modifier
  - [ ] Mend: `all_creature_hp = all_creature_hp_max * 0.5` (add 50% not set)
  - [ ] Wild Seed: `spawn_random_tier2_creature()` on open map cell
  - [ ] Echo Strike: `projectile_echo_chance = 0.20f` (repeated at 0.3s delay) for 3 waves
  - [ ] Essence Cache: `essence += essence * 0.30f`
  - [ ] Iron Skin: `all_creatures_damage_received *= 0.80f` for 2 waves
  - [ ] Apex Hunter: highest-kill creature `damage *= 1.50f` for 1 wave
  - [ ] Void Pulse: every 10th kill explodes 80px for 1 wave
- [ ] Buff icons: unique 32×32 icon per card
- [ ] Active buff HUD strip: show icons of buffs currently active, with wave countdown

### 6.2 Inter-Level Shop — Bazaar Tab
- [ ] Rotation: 4 random character seeds per level (weighted by rarity)
- [ ] Rarity weights: Common 60%, Rare 25%, Epic 12%, Legendary 3%
- [ ] Cost formula per GDD §9
- [ ] Buy: add creature seed to `player_roster`, deduct essence
- [ ] Duplicate handling: "Already owned — buy for second unit?" prompt
- [ ] Reroll: costs 15 essence, refreshes all 4 slots (once per visit)
- [ ] "Can't afford" state: greyed out, shows shortage amount

### 6.3 Inter-Level Shop — Forge Tab
- [ ] Show all owned characters with their 5 upgrade nodes
- [ ] Upgrade node: clickable, shows current rank, next effect, cost
- [ ] Cost formula: `(current_rank + 1) × 15 × (1 + level × 0.05)` essence
- [ ] Max rank enforcement (V for Strike/Vitality, III for Reach/Tempo/Signature)
- [ ] Upgrade immediately applies to live `CreatureContainer` stats
- [ ] Visual indicator: "maxed" badge on fully upgraded nodes

### 6.4 Inter-Level Shop — Relic Tab
- [ ] Show 3 equip slots + full grid of unlocked relics
- [ ] Drag-and-drop relic into slot (or click → pick slot)
- [ ] Active relic effects: applied to `RelicSystem::apply_all()` at level start
- [ ] Locked relics: shown greyed with "Unlock in Armory — N Shards"
- [ ] Relics not yet obtained: completely hidden (don't show what you can't get)

### 6.5 Inter-Level Shop — Repair Tab
- [ ] 3 repair tiers: +20 HP (40 essence), +50 HP (90 essence), full (160 essence)
- [ ] Note: base HP still resets to 100 each level — repair is for next level only
- [ ] Show next-level starting HP preview

### 6.6 Armory (Meta Hub)
- [ ] Character Gallery: grid of 10 character portraits, locked/unlocked state, cost
- [ ] Character detail popup: lore text, evolution stages preview, stats summary
- [ ] Shard balance displayed top-right
- [ ] Unlock purchase flow: confirm dialog → deduct Shards → unlock character
- [ ] Passive Mastery tree: 8 masteries × 3 ranks, visual tree layout
- [ ] Mastery purchase: confirm → deduct Shards → apply permanently to `PlayerProfile`
- [ ] Cosmetics tab: skins for characters, base, particles, HUD themes
- [ ] Cosmetic preview: show character with selected skin in a rotating preview window
- [ ] All Armory state saved to `PlayerProfile.json` / backend sync

---

## 7. Core Game — UI, HUD & Screens

### 7.1 HUD (In-Gameplay)
- [ ] Top bar: `Level N — Wave X / Y` | countdown timer | essence (animated) | ★★★ threshold marks
- [ ] Base HP bar: bottom-center, shows HP/max, markers at 30% and 70% (star thresholds)
- [ ] Active buff strip: icons below top bar, wave countdown per buff
- [ ] Selected creature panel: character art, name, tier, kills, HP bar, attack range circle, evolution progress bar
- [ ] Creature type selector wheel (bottom-left): 6 type icons + cost, locked types greyed, selected type highlighted
- [ ] Kill feed (top-right): last 5 notable kills (boss kills, elite kills), 3s fade per entry
- [ ] Floating damage numbers: pool of 500 entities, white for normal, yellow for crit/evolve bonus
- [ ] Evolution banner: full-width flash, "[Name] EVOLVED → TIER N", 2.5s duration
- [ ] New enemy banner: "[Enemy Type]" with description, slides in from top, auto-dismisses after 4s
- [ ] Wave start banner: "WAVE X / Y" pulse, 1.5s
- [ ] Wave clear banner: "WAVE X CLEAR — N killed" with essence earned, 2s
- [ ] Boss wave intro banner: "SIEGE LORD APPROACHES" full-screen, 3s, with boss art

### 7.2 Level Select Screen
- [ ] Grid of level tiles: 5 per row, scrollable
- [ ] Each tile: level number, biome color, star rating (0–3 stars), locked state
- [ ] Locked tile: shows lock icon, "Complete Level N–1 to unlock"
- [ ] Hover tooltip: enemy types in level, wave count, boss type
- [ ] "Continue" floating button: navigates to highest unlocked level
- [ ] Biome section headers: "VERDANT FIELDS (1–10)", "ASHLANDS (11–20)", etc.
- [ ] Tile pop-in animation on first unlock (scale + star sparkle)
- [ ] Total stars counter: "★ 234 / ∞" at top

### 7.3 Pre-Level Screen
- [ ] Level number + biome name as header
- [ ] Level stats: wave count, enemy types (icons), boss type, elite indicator
- [ ] Player roster grid: show all owned creatures, tier badge, kill count badge
- [ ] Current relic slots: 3 icons visible
- [ ] Essence balance shown
- [ ] "[Deploy]" button → transitions to gameplay with level intro

### 7.4 Inter-Level Screen
- [ ] Slide-in animation from right over 0.4s
- [ ] Results panel: level N, star reveal animation (fill one-by-one with delay), stats table
- [ ] Stats: enemies killed, essence earned, base HP remaining %, time taken, evolutions this level, merges this level
- [ ] Best creature highlight: top creature by kills with large portrait + "Tier N, X kills"
- [ ] Shop tabs: Bazaar / Forge / Relics / Repair (tabbed navigation)
- [ ] Roster strip at bottom: scrollable row of all owned creatures
- [ ] Buttons: [Next Level →] [Replay Level] [Level Select]
- [ ] Transition back to gameplay: fade out + level intro animation

### 7.5 Main Menu
- [ ] Background: animated parallax scene from Verdant Fields biome
- [ ] Logo: Tower Swarm title with creature silhouettes
- [ ] Buttons: [Play] [Armory] [Leaderboard] [Daily Level] [Settings] [Quit]
- [ ] Player profile strip: player name, level, total stars, current season info
- [ ] Daily level badge: "Today's Level: N — Your Best: ★★☆"
- [ ] Version number bottom-right
- [ ] Steam news ticker (optional): pulls latest Steam announcement

### 7.6 Pause Menu
- [ ] Overlay: dark semi-transparent over gameplay (game freezes)
- [ ] Options: [Resume] [Retry Level] [Level Select] [Settings] [Quit to Desktop]
- [ ] Retry level: confirm dialog → restore pre-level save snapshot → restart
- [ ] Quit to desktop: confirm dialog → save state → exit

### 7.7 Settings Screen
- [ ] **Display tab**: resolution dropdown, fullscreen toggle, VSync toggle, brightness slider
- [ ] **Audio tab**: master volume, music volume, SFX volume — sliders with live preview
- [ ] **Input tab**: key rebinding table (16 actions), controller deadzone slider, edge-scroll sensitivity
- [ ] **Gameplay tab**: camera pan speed, damage number toggle, screen shake intensity (off/low/high), auto-merge toggle
- [ ] **Accessibility tab**: colorblind mode (Deuteranopia / Protanopia / Tritanopia), UI scale (90–150%), reduce-motion mode
- [ ] **Language tab**: language selector (see Section 15)
- [ ] All settings saved to `settings.ini` on close

### 7.8 Leaderboard Screen
- [ ] Tab: Daily / Weekly / All-Time
- [ ] Each entry: rank, player name (Steam persona), level reached / stars / time
- [ ] Current player highlighted in list (even if not top 20)
- [ ] Refresh button: re-fetch from backend
- [ ] Friend filter (future): show only Steam friends
- [ ] Share button: copy leaderboard screenshot to clipboard

---

## 8. Core Game — Audio

### 8.1 Audio Manager
- [ ] SDL3 audio stream setup: 44100 Hz, stereo, float32
- [ ] `AudioManager::play_sfx(id, volume, pitch_variation)` — play one-shot SFX
- [ ] `AudioManager::play_music(track_id)` — fade current track out, fade new track in (1s crossfade)
- [ ] `AudioManager::set_intensity(level)` — blend between calm/tense/intense/boss layers
- [ ] SFX pool: 32 simultaneous SFX channels
- [ ] Music looping: seamless loop point at bar boundary

### 8.2 Dynamic Music System
- [ ] Music state machine: CALM → TENSE (wave starts) → INTENSE (boss wave) → BOSS → VICTORY → FAIL
- [ ] Per-biome music base: 5 biome themes loaded dynamically
- [ ] Intensity layers crossfade based on `WaveSpawner::get_intensity()`
- [ ] Boss wave: cut to boss stinger track immediately on boss spawn
- [ ] Level complete: victory jingle plays, music ducks under it
- [ ] Level fail: fail jingle, music stops

---

## 9. Art & Visual Production

### 9.1 Art Direction Document
- [ ] Define visual style: pixel art (16×16 / 32×32 base tile), slightly animated
- [ ] Color palette per biome (5 palettes, 16 colors each)
- [ ] Creature visual language: shapes communicate role (circle=Shooter, star=Splasher, etc.)
- [ ] Enemy visual language: size communicates HP, color communicates type
- [ ] UI style: dark background, neon accents, clean readable fonts
- [ ] Reference moodboard assembled (artstyle references)

### 9.2 Character Sprites
- [ ] Brix: 5 visual tiers × idle (4 frames) + attack (6 frames) = 50 frames
- [ ] Flara: 5 visual tiers × idle + attack + explosion = 60 frames
- [ ] Mossling: 5 visual tiers × idle + aura pulse = 40 frames
- [ ] Glitch: 5 visual tiers × idle + ability = 50 frames
- [ ] Ironjaw: 5 visual tiers × idle + charge = 50 frames
- [ ] Wraith: 5 visual tiers × idle + shoot = 50 frames
- [ ] Crystalis: 5 visual tiers × idle + beam = 50 frames
- [ ] Vex: 5 visual tiers × idle + random ability = 60 frames
- [ ] Orin: 5 visual tiers × idle + special = 50 frames
- [ ] Null: 5 visual tiers × idle + drain = 50 frames
- [ ] Character portrait art: 10 × high-res bust portraits for menus (200×200px)
- [ ] Character card art: 10 × full body for Armory gallery (400×600px)

### 9.3 Enemy Sprites
- [ ] Grub: idle (4f) + walk (6f) + death (4f)
- [ ] Hulk: idle (4f) + walk (4f) + death (6f)
- [ ] Scuttle: idle (2f) + walk (4f) + death (2f) — small, 16×16
- [ ] Driftwing: idle (4f) + fly (6f) + death (4f)
- [ ] Divide: idle (4f) + walk (4f) + split (4f) + death
- [ ] Vanguard: idle (4f) + walk (4f) + shield raise (2f) + death (4f)
- [ ] Mender: idle (4f) + walk (4f) + heal pulse (4f) + death (4f)
- [ ] Siege Lord: 5 biome variants × idle (6f) + walk (6f) + ability (8f) + death (10f)
- [ ] Void variants: recolor/distortion filter for all 7 standard enemies

### 9.4 Environment & Tiles
- [ ] Verdant Fields tileset: grass, dirt path, stone, flower, tree trunk, bush (6 tiles × 4 variants)
- [ ] Ashlands tileset: sand, cracked rock, ember path, lava pool edge, cactus, ruin (6 × 4)
- [ ] Frostmarsh tileset: snow, ice path, frozen ground, icicle, snowdrift, frozen tree (6 × 4)
- [ ] Deepcore tileset: stone floor, magma crack, iron grate, stalactite, lava flow, crystal vein (6 × 4)
- [ ] Void tileset: dark matter, energy crack, starfield tile, void tendril, portal edge, nothing (6 × 4)
- [ ] Base / Nexus sprite: 3 damage states (healthy, cracked, critical) × 5 biome skins
- [ ] Wall sprite: intact + cracked + rubble (3 states) × 5 biomes
- [ ] Map background art: 5 large backgrounds (1280×720px each, one per biome)

### 9.5 VFX Sprites
- [ ] Explosion: 8 frames, 3 sizes (small 32px, medium 64px, large 128px)
- [ ] Evolution burst: 6 frames, 64px — 7 color variants (one per tier range)
- [ ] Merge flash: 4 frames, 96px
- [ ] Death dissolve: 6 frames, matches enemy size
- [ ] Essence orb: 4 frames floating + trail particle
- [ ] Projectile sprites: bolt, orb, arrow, charge trail, slow field, beam (6 types)
- [ ] Level complete: confetti 12 frames, star burst 6 frames
- [ ] Boss death: 16-frame massive explosion, screen-wide

### 9.6 UI Art
- [ ] 6 creature type icons (32×32): Shooter, Splasher, Trapper, Charger, Support, Sniper
- [ ] 8 enemy type icons (24×24) for enemy intro banners
- [ ] Relic icons (32×32): 20 relics
- [ ] Wave buff card artwork: 12 unique card illustrations (128×192px)
- [ ] Star icons: filled / empty / locked (3 states)
- [ ] Currency icons: essence drop, Shard gem
- [ ] Biome badge icons: 5 biome symbols
- [ ] HUD frame elements: top bar frame, creature panel frame, selector wheel
- [ ] Button states: normal / hover / pressed / disabled

### 9.7 Steam Store Art
- [ ] **Capsule Small** (231×87px): logo + creature silhouettes
- [ ] **Capsule Main** (616×353px): key art — creatures vs enemy horde, biome backdrop
- [ ] **Library Capsule** (600×900px): portrait format key art
- [ ] **Library Hero** (3840×1240px): wide cinematic banner
- [ ] **Page Background** (1438×810px): atmospheric biome scene
- [ ] **Bundle Banner**: 1280×720px art (for future bundles)
- [ ] **Screenshots**: minimum 5, recommended 10 (1920×1080px each, showing different systems)
- [ ] **Animated GIFs**: 3 GIFs showing evolution, merge, boss fight (for store page)
- [ ] **Game Icon** (256×256px): Steam library icon

---

## 10. Audio Production

### 10.1 Music Tracks
- [ ] Verdant Fields — Calm layer (2m30s loop)
- [ ] Verdant Fields — Tense layer (add-on, 2m30s)
- [ ] Verdant Fields — Intense layer (add-on, 2m30s)
- [ ] Ashlands — 3 intensity layers
- [ ] Frostmarsh — 3 intensity layers
- [ ] Deepcore — 3 intensity layers
- [ ] The Void — 3 intensity layers
- [ ] Boss stinger (universal, biome variations): 5 tracks
- [ ] Main menu theme: (1m30s, loops after intro)
- [ ] Level complete jingle (10s)
- [ ] Level fail jingle (6s)
- [ ] Victory fanfare (for 3-star perfect runs, 12s)
- [ ] Evolution fanfares: tier 1–3 (soft), tier 4–6 (mid), tier 7–9 (epic), tier 10+ (cinematic): 4 variants
- [ ] Merge chime: 3 variants (subtle, mid, epic for Tier 7+ merges)

### 10.2 Sound Effects
- [ ] Creature attacks: bolt_shoot, splash_explode, slow_field_deploy, charge_impact, support_pulse, sniper_shot — per character type
- [ ] Projectile hit: soft_hit, heavy_hit, splash_hit, pierce_hit
- [ ] Enemy sounds: grub_die, hulk_stomp, scuttle_swarm_ambient, driftwing_fly, boss_roar, boss_phase_shift
- [ ] UI sounds: button_click, button_hover, menu_open, menu_close, shop_purchase, shop_reject (can't afford), tab_switch
- [ ] Game events: wave_start, wave_clear, level_complete, level_fail, creature_placed, creature_sold
- [ ] Evolution: evolve_whoosh, evolve_chime (tier-appropriate)
- [ ] Merge: merge_slide, merge_flash, merge_complete
- [ ] Boss: boss_intro_roar, boss_phase_transition, boss_death
- [ ] Ambient: biome ambience loops (birds, wind, fire crackle, cave drips, void hum) — 5 tracks

### 10.3 Audio Quality Standards
- [ ] All SFX: 44100 Hz, 16-bit, stereo or mono as appropriate, OGG Vorbis q6
- [ ] All music: 44100 Hz, stereo, OGG Vorbis q8
- [ ] Music mastered to -14 LUFS integrated (streaming standard)
- [ ] SFX peak: no clipping above -3 dBTP
- [ ] All audio reviewed for licensing: original compositions or royalty-free

---

## 11. Steamworks Integration

### 11.1 Setup
- [ ] Create Steamworks developer account + pay app fee ($100)
- [ ] Create app in Steamworks dashboard, get App ID
- [ ] Download Steamworks SDK (latest stable)
- [ ] Add Steamworks SDK to `vcpkg` or manual include path
- [ ] Initialize `SteamAPI_Init()` on startup — handle failure (launch without Steam features)
- [ ] `SteamAPI_Shutdown()` on clean exit
- [ ] `SteamAPI_RunCallbacks()` called each frame in main loop
- [ ] App ID written to `steam_appid.txt` in game directory (for dev builds)

### 11.2 Steam Achievements (40 defined)
- [ ] Define all 40 achievements in Steamworks dashboard (name, description, icon)
- [ ] Implement `SteamAchievementsManager` class
- [ ] Hook achievements to in-game events:
  - [ ] `ACH_FIRST_BLOOD` — kill first enemy
  - [ ] `ACH_EVOLVER` — reach Tier 3 with any creature
  - [ ] `ACH_MERGER` — complete first merge
  - [ ] `ACH_TEN_FORWARD` — complete Level 10
  - [ ] `ACH_STARBOUND` — 3-star any level
  - [ ] `ACH_BOSS_SLAYER` — defeat 10 Siege Lords
  - [ ] `ACH_LEVEL_25` — complete Level 25
  - [ ] `ACH_PERFECT_DEFENSE` — complete level with 100% base HP
  - [ ] `ACH_COLLECTOR` — unlock 5 characters
  - [ ] `ACH_ARMY_OF_ONE` — win level with 1 creature type only
  - [ ] `ACH_MERGE_CHAIN` — 5 merges in single level
  - [ ] `ACH_EVOLUTION_GOD` — reach Tier 10 with any creature
  - [ ] `ACH_CENTURY` — complete Level 100
  - [ ] `ACH_LEGENDARY` — unlock Null
  - [ ] `ACH_NO_DAMAGE_RUN` — level complete, base untouched
  - [ ] `ACH_SPEED_RUN_L10` — complete Level 10 in under 8 minutes
  - [ ] `ACH_VOID_WALKER` — reach Void biome (Level 61)
  - [ ] `ACH_FULL_ROSTER` — unlock all 10 characters
  - [ ] `ACH_MAX_TIER` — reach Tier 20 with any creature
  - [ ] `ACH_RELICS_COLLECTOR` — unlock all 20 relics
  - [ ] ... (20 more covering: biome bosses, economy milestones, specific character achievements)
- [ ] Achievement unlock notification integrates with Steam overlay toast
- [ ] `SteamUserStats::StoreStats()` called after any stat update

### 11.3 Steam Leaderboards
- [ ] Create leaderboards in Steamworks: `global_max_level`, `daily_challenge_YYYYMMDD`, `weekly_stars`
- [ ] `SteamLeaderboardsManager`: upload score on level complete
- [ ] Download and display leaderboard entries in-game (top 20 + player rank)
- [ ] Score = `level_number × 1000 + stars × 100 + time_bonus` (deterministic sort)
- [ ] Tie-break: faster completion ranks higher

### 11.4 Steam Cloud Saves
- [ ] Register `save.json` in Steamworks cloud sync (auto-sync on exit/launch)
- [ ] Cloud quota check: warn player if approaching Steam Cloud limit
- [ ] Conflict resolution: newer timestamp wins (show UI if dates conflict)
- [ ] `ISteamRemoteStorage::FileWrite()` after each save
- [ ] Fallback: if Steam Cloud unavailable, local save only (graceful degradation)

### 11.5 Steam Overlay
- [ ] `SteamFriends::ActivateGameOverlay()` bound to Shift+Tab (Steam default)
- [ ] Screenshot hotkey (F12): captured via `ISteamScreenshots::TriggerScreenshot()`
- [ ] Screenshots auto-tagged with level number + star rating
- [ ] Rich Presence: `SetRichPresence("status", "Level N — Wave X/Y")`
- [ ] Rich Presence: `SetRichPresence("steam_display", "#Status_InGame")`

### 11.6 Steam Input
- [ ] Create `steam_input_manifest.vdf` with all actions defined
- [ ] Action sets: `GameControls` (gameplay), `MenuControls` (menus)
- [ ] Bind all 16 rebindable actions to Steam Input action names
- [ ] In-game: use `ISteamInput::GetAnalogActionData()` for camera movement
- [ ] Default controller glyphs: shown in tutorial and settings screen
- [ ] Support: Xbox, PlayStation, Nintendo Switch Pro, generic gamepad layouts

### 11.7 Steam Deck Certification
- [ ] Run on Steam Deck default settings (1280×800, 60Hz target)
- [ ] All UI readable at Steam Deck screen size (7" at 1280×800)
- [ ] No mouse-required actions: all gameplay possible with controller
- [ ] Gyro: optional gyro aiming for cursor (nice-to-have)
- [ ] Touch screen: basic touch support via SDL3 touch events (tap = click)
- [ ] FPS: stable 60 on Steam Deck APU (Zen 2 + RDNA 2)
- [ ] Battery: target < 15W TDP (Deck battery life > 2h)
- [ ] Verified badge checklist: submit for Valve's Steam Deck Verified review

### 11.8 Steam Trading Cards (Post-Launch)
- [ ] Define 8 trading cards (6 character portraits + 2 biome scenes)
- [ ] 4 badge levels (bronze → silver → gold → foil)
- [ ] 3 emoticons from game characters
- [ ] 3 profile backgrounds (biome art)
- [ ] Submit to Valve for review (requires 10,000+ owners threshold)

---

## 12. Backend Infrastructure

### 12.1 Server Setup
- [ ] Choose cloud provider (AWS / GCP / Azure / Hetzner)
- [ ] Production environment: 2× app servers behind load balancer
- [ ] Staging environment: 1× app server (mirrors production config)
- [ ] Database: PostgreSQL 16 managed instance (RDS or Supabase)
- [ ] Cache: Redis 7 instance for leaderboard reads
- [ ] CDN: CloudFront or Cloudflare for static assets (sprites, WASM bundle)
- [ ] SSL: Let's Encrypt auto-renew for all endpoints
- [ ] Firewall: only ports 80/443 open to public; DB port only from app servers
- [ ] VPN: all internal services behind VPN (WireGuard)

### 12.2 Database Schema
```sql
-- Implement all tables:
tower_swarm_runs        (id, user_id, level, stars, time_sec, creature_json, created_at)
tower_swarm_progress    (user_id, max_level, total_stars, player_level, shards, essence, roster_json, updated_at)
tower_swarm_level_best  (user_id, level_number, best_stars, best_time_sec)
tower_swarm_daily       (user_id, date, level_number, stars, time_sec)
tower_swarm_events      (id, user_id, event_name, event_data jsonb, created_at)
tower_swarm_achievements (user_id, achievement_id, unlocked_at)
```
- [ ] Migrations: all schema changes via migration files (Flyway or Liquibase)
- [ ] Indexes: user_id on all tables, (level_number, stars, time_sec) on runs for leaderboard
- [ ] Row-level security: users can only read/write their own rows
- [ ] Backup: automated daily backups with 30-day retention

### 12.3 API Implementation
- [ ] `POST /api/games/tower-swarm/level-complete` — validate + insert run, update progress
- [ ] `GET /api/games/tower-swarm/leaderboard` — cached query with Redis TTL 60s
- [ ] `GET /api/games/tower-swarm/daily-level` — `srand(date)` seed + level number
- [ ] `GET /api/games/tower-swarm/player/:id/progress` — full player progress object
- [ ] `POST /api/games/tower-swarm/sync-save` — overwrite player save JSON
- [ ] `GET /api/games/tower-swarm/replay/:run_id` — serve replay data
- [ ] Authentication: **Steam ticket auth** (`ISteamUser::GetAuthTicketForWebApi()` → verify on server)
- [ ] Rate limiting: per-endpoint limits (100 score submissions/day/user)
- [ ] Score validation: sanity checks (Level 200 in 2 minutes → reject, flag account)
- [ ] API versioning: `/api/v1/...` prefix (anticipate breaking changes)

### 12.4 Monitoring & Alerting
- [ ] Prometheus metrics: request count, latency p50/p95/p99, error rate, DB pool size
- [ ] Grafana dashboard: all key metrics visible
- [ ] Alerting: PagerDuty / Discord alert on: error rate > 1%, latency p95 > 500ms, DB CPU > 80%
- [ ] Uptime monitoring: external pings every 60s (UptimeRobot or similar)
- [ ] Log aggregation: structured JSON logs → Loki / CloudWatch
- [ ] Sentry: backend error tracking with stack traces

### 12.5 Authentication & Security
- [ ] Steam ticket verification on all authenticated endpoints
- [ ] JWT tokens issued after Steam ticket validation (TTL 24h)
- [ ] HTTPS only (redirect HTTP → HTTPS)
- [ ] SQL injection prevention: parameterized queries everywhere
- [ ] Input validation: all API inputs validated + sanitized
- [ ] CORS: allow only game domain origins
- [ ] Secrets management: all secrets in environment variables or secrets manager (never in code)
- [ ] Audit log: log all score submissions + account actions

---

## 13. QA & Testing

### 13.1 Unit Tests (C++)
- [ ] `LevelScalerTests`: verify difficulty formulas for levels 1, 10, 50, 100, 200
- [ ] `EvolutionSystemTests`: verify kill thresholds + tier-up at each tier, Tier 20 edge case
- [ ] `MergeSystemTests`: valid merge conditions, blocked conditions, kill inheritance
- [ ] `EssenceSystemTests`: pickup collection, interest calculation, sell refund
- [ ] `SaveStateTests`: serialize → deserialize creature roster, verify no data loss
- [ ] `PathGridTests`: valid path found, blocked path rejected, flyer bypass
- [ ] `WaveScalerTests`: enemy count formulas match expected values at key levels
- [ ] `BuffSystemTests`: all 12 buffs apply and expire correctly

### 13.2 Integration Tests
- [ ] Full level simulation: headless run Level 1 → win, verify save state written
- [ ] Headless run Level 1 → fail (base HP = 0), verify fail state + snapshot restore
- [ ] Evolution chain: create Tier 1 Brix, simulate 10 kills, verify Tier 2
- [ ] Merge chain: two Tier-3 Brix adjacent → auto-merge → Tier-4 with correct kills
- [ ] Economy round-trip: earn essence from kills, buy seed, verify balance and roster
- [ ] Steam API mock: verify achievement unlock, leaderboard submit, Cloud save called

### 13.3 Playtesting Rounds

**Round 1 — Internal (before Alpha)**
- [ ] 5 internal testers, Levels 1–10
- [ ] Goal: identify broken mechanics, crashes, progression blockers
- [ ] Feedback form: fun rating per level, confusion moments, crashes

**Round 2 — Alpha (before Beta)**
- [ ] 25 testers, Levels 1–25
- [ ] Goal: balance feedback, economy feel, creature viability
- [ ] Metrics: essence per level, wave clear rate, most-used characters, retry count

**Round 3 — Beta (before Gold)**
- [ ] 100 testers, open levels
- [ ] Goal: performance across hardware tiers, bugs, UX pain points
- [ ] Steam beta branch: testers access via Steam key
- [ ] Bug tracker public: testers submit bugs via in-game widget or GitHub Issues

**Round 4 — Steam Next Fest (if applicable)**
- [ ] Demo build: Levels 1–10, limited character roster (Brix, Flara, Mossling)
- [ ] Wishlists: track conversion from demo to wishlist

### 13.4 Performance Testing
- [ ] Benchmark: 60 FPS on minimum spec (i5-8400, GTX 1060 6GB, 8GB RAM, Windows 10)
- [ ] Benchmark: 60 FPS on Steam Deck (Zen 2 APU)
- [ ] Stress test: Level 200 headless run, verify FPS doesn't drop below 30
- [ ] Memory test: 2-hour session play, confirm no memory leak growth
- [ ] Load time: game cold launch + asset loading < 10 seconds on SSD
- [ ] Level transition time: inter-level screen loads in < 0.5 seconds

### 13.5 Platform Certification
- [ ] Windows: test on Win 10 21H2, Win 11 23H2 — no crashes
- [ ] Linux: test on Ubuntu 22.04, Fedora 38, SteamOS 3.x — no crashes
- [ ] Steam Deck: submit for Deck Verified review (see §11.7 checklist)
- [ ] Resolution test: 1280×720, 1920×1080, 2560×1440, 4K — UI correct at all
- [ ] Ultrawide test: 21:9 and 32:9 — game pillarboxed or supports natively

### 13.6 Regression Suite
- [ ] Automated regression: run full unit + integration test suite on every PR
- [ ] Screenshot regression: key screens captured, compared to baseline via perceptual hash
- [ ] Performance regression: FPS benchmark run in CI on GPU-capable agent, alert if drops >10%

---

## 14. Performance & Optimization

### 14.1 Entity System
- [ ] Profile: `ATMProfiler` hooks on update start/end per container
- [ ] Target: `CreatureContainer::update()` < 0.3ms for 200 creatures
- [ ] Target: `EnemyContainer::update()` < 2ms for 5,000 enemies
- [ ] Target: `ProjectileContainer::update()` < 0.5ms for 3,000 projectiles
- [ ] LOD: enemies > 600px from viewport skip per-frame AI (just advance on path vector)
- [ ] Batch spawn: cap at 50 enemy spawns per frame to avoid spike
- [ ] Deferred destroy: mark entities for removal, batch-remove at end of frame

### 14.2 Rendering
- [ ] Confirm batch renderer groups by texture + z-index correctly
- [ ] Profile: `engine_render_scene()` < 3ms for 10,000 visible entities at 1080p
- [ ] Occlusion: entities outside camera frustum not submitted to renderer
- [ ] Static cache: tilemap rendered once to off-screen texture, re-used until camera moves
- [ ] UI: render HUD to separate pass, not in main entity batch

### 14.3 Memory
- [ ] Maximum memory budget: 512MB RAM for game state + assets + engine
- [ ] Texture atlas: all sprites in single 2048×2048 RGBA atlas (~16MB)
- [ ] Audio streaming: music streams from disk, only SFX fully loaded in RAM
- [ ] Entity pool sizing: document pool sizes in `Constants.h` with memory cost annotations
- [ ] Memory profiler: Valgrind (Linux) + DrMemory (Windows) clean runs before each release

### 14.4 WASM-Specific
- [ ] Emscripten heap: 512MB allocated (`-sINITIAL_MEMORY=536870912`)
- [ ] WASM binary size: target < 8MB (compressed) for reasonable web load time
- [ ] Asset preloading: all sprites preloaded before game starts (show progress bar)
- [ ] Web worker: heavy pathfinding calculations offloaded to web worker (if needed)
- [ ] WebGL2: use WebGL2 renderer path for better performance

---

## 15. Accessibility & Localization

### 15.1 Accessibility
- [ ] **Colorblind modes**: Deuteranopia (red-green), Protanopia (red-blind), Tritanopia (blue-yellow) — recolor tier glow, enemy HP bars, star indicators
- [ ] **UI scale**: 90% / 100% / 125% / 150% — scales HUD + menu elements
- [ ] **Reduce motion**: disable screen shake, evolution pulse, camera smoothing
- [ ] **Font size**: small / medium / large for all in-game text
- [ ] **High contrast mode**: increase UI border thickness, darker backgrounds
- [ ] **Pause-and-play**: game can be fully paused at any time (no real-time-only decisions)
- [ ] **Subtitles**: any voiced content (if added) has subtitles

### 15.2 Localization
- [ ] **Priority languages**: English (base), Simplified Chinese, Brazilian Portuguese, Spanish (LATAM), Russian
- [ ] String extraction: all UI strings in `strings/en.json`, no hardcoded text in C++
- [ ] String IDs: snake_case keys (`hud_wave_label`, `shop_buy_button`, etc.)
- [ ] Numeric formatting: locale-aware for large numbers (100,000 vs 100.000)
- [ ] Font support: include CJK-compatible font for Chinese
- [ ] RTL support: not required for v1.0 (Arabic/Hebrew deferred)
- [ ] Translation process: export `en.json`, send to translators, import per language
- [ ] QA for each language: native speaker review before launch
- [ ] Steam store page: translated descriptions for all 5 languages

---

## 16. Legal & Compliance

### 16.1 Game IP
- [ ] Trademark search: "Tower Swarm" — confirm no conflicts in key markets (US, EU)
- [ ] File trademark application: "Tower Swarm" wordmark (US + EU, Nice Class 41 — games)
- [ ] Copyright: all game code, art, and audio registered as original works
- [ ] Studio legal entity: register business entity if not already done (LLC or equivalent)

### 16.2 End User License Agreement (EULA)
- [ ] Draft EULA covering: license grant (non-transferable), prohibited actions, disclaimer of warranties, limitation of liability, governing law
- [ ] Shown on first launch: "I Accept" required before playing
- [ ] Stored: EULA version logged in save state (re-show if EULA updated)
- [ ] Review by legal counsel

### 16.3 Privacy Policy
- [ ] Document: what data collected (usage analytics if opted in, Steam ID for leaderboards)
- [ ] GDPR compliance: lawful basis, data retention limits, right to erasure endpoint
- [ ] CCPA compliance: California privacy rights
- [ ] COPPA: age gate if collecting data from < 13 (Steam handles age at account level)
- [ ] Data deletion endpoint: `DELETE /api/games/tower-swarm/player/:id` — removes all player data
- [ ] Privacy policy URL: hosted publicly, linked from game main menu and Steam store page
- [ ] Review by legal counsel

### 16.4 Age Ratings
- [ ] **ESRB** (North America): submit via ESRB's IARC tool (automated, free via Steam)
- [ ] **PEGI** (Europe): submit via IARC — expected rating: PEGI 7
- [ ] **USK** (Germany): submit via IARC
- [ ] **CERO** (Japan, optional): direct submission if targeting Japanese market
- [ ] **ACB** (Australia): submit via IARC
- [ ] All ratings received and entered into Steamworks before launch

### 16.5 Music & Audio Licensing
- [ ] Confirm all music is original composition (no licensed tracks)
- [ ] If using third-party SFX library: verify license permits use in commercial game
- [ ] Credits screen: list all contributors (composer, sound designer, artists)

### 16.6 Third-Party Licenses
- [ ] SDL3: zlib license — no attribution required but good practice
- [ ] Steamworks SDK: Steamworks Terms of Service compliance
- [ ] nlohmann/json: MIT license — include in credits
- [ ] spdlog / fmt: MIT license — include in credits
- [ ] All fonts: verify commercial use license
- [ ] Third-party licenses file: `LICENSES.txt` bundled with game

---

## 17. Steam Store & Marketing

### 17.1 Steamworks Store Configuration
- [ ] Developer + Publisher name configured
- [ ] App name: "Tower Swarm"
- [ ] Short description (300 chars): punchy, covers core hook
- [ ] Long description (full HTML): gameplay overview, features list, biomes, characters teaser
- [ ] Tags: apply all relevant Steam tags (Tower Defense, Strategy, Roguelite, Indie, Pixel Graphics, 2D)
- [ ] Categories: Single-player, Steam Achievements, Steam Cloud, Steam Leaderboards, Steam Trading Cards, Full controller support, Steam Deck Verified
- [ ] System requirements: minimum + recommended (Windows 10/11, 4GB RAM, GTX 960 / RX 580)
- [ ] All age ratings uploaded to Steamworks
- [ ] Privacy policy URL added to store page
- [ ] EULA URL (optional, if externally hosted)

### 17.2 Store Page Assets
- [ ] All art from §9.7 uploaded and approved (capsules, hero, background)
- [ ] 10 screenshots uploaded: cover all major systems (evolution, merge, shop, boss, biomes)
- [ ] 1 launch trailer (60–90 seconds): gameplay → evolution moment → merge moment → boss → level complete
- [ ] 1 gameplay trailer (2–3 minutes): deeper dive for serious buyers

### 17.3 Pricing Strategy
- [ ] Base price: $9.99 USD (accessible impulse tier for Tower Defense genre)
- [ ] Regional pricing: follow Steamworks regional price conversion recommendations
- [ ] Launch discount: 20% for first 2 weeks (Steamworks launch discount widget)
- [ ] Future sale eligibility: minimum 30 days at full price before any sale

### 17.4 Press & Influencer Outreach
- [ ] Press kit: logo, screenshots, GIFs, 1-page fact sheet, elevator pitch, developer bio
- [ ] Press kit hosted: presskit() page or itch.io
- [ ] Press list: 50 gaming outlets (Kotaku, PC Gamer, RPS, IGN Indie, TouchArcade, etc.)
- [ ] Influencer list: 20 relevant YouTubers/streamers (Tower Defense genre, Indie game content)
- [ ] Review copies: send via Steam (no-cost review copies in Steamworks)
- [ ] Outreach timing: 6 weeks before launch for press, 2 weeks for streamers
- [ ] Embargo date: 48h before launch for press reviews

### 17.5 Wishlist Campaign
- [ ] Steam coming soon page live: minimum 3 months before launch
- [ ] Steam page visible and wishlisting enabled from day 1 of marketing
- [ ] Steam Next Fest participation: submit for event (held bi-annually)
- [ ] Demo for Next Fest: Levels 1–10, 3 characters (Brix, Flara, Mossling)
- [ ] Wishlist goal: 10,000 wishlists before launch (benchmark for successful indie launch)

### 17.6 Social Media
- [ ] Create: Twitter/X, TikTok, Discord server, YouTube channel
- [ ] Content calendar: 3 posts/week in the 3 months before launch
- [ ] Content types: evolution showcase GIFs, "did you know" mechanic posts, biome reveals, character spotlights
- [ ] Discord: community server with channels: #announcements, #feedback, #bug-reports, #screenshots, #general
- [ ] TikTok content: evolution moments, merge chains, boss fights (10–30 second clips)
- [ ] Reddit: post in r/indiegaming, r/towerdefense, r/gamedev when appropriate

### 17.7 Launch Day Checklist
- [ ] Store page fully complete (descriptions, all assets, all ratings)
- [ ] All achievements defined and tested
- [ ] Leaderboards active
- [ ] Steam Cloud sync tested
- [ ] Day-1 patch prepared and ready to upload (hotfix branch)
- [ ] Discord server ready with launch announcement
- [ ] Social posts scheduled (launch day, +1 day)
- [ ] Press release sent to outlets
- [ ] Steam announcement posted
- [ ] Backend servers scaled up (3× capacity for launch day traffic)
- [ ] On-call rotation: developer available 24h post-launch for emergency hotfix

---

## 18. Launch Operations

### 18.1 Release Build (Gold)
- [ ] Version tagged: `v1.0.0` on `main` branch
- [ ] All CI tests pass on release candidate
- [ ] Zero P0/P1 bugs open
- [ ] Performance benchmarks passed on minimum spec
- [ ] All legal requirements complete (ratings, EULA, privacy policy)
- [ ] Steam Deck Verified status received (or Playable minimum)
- [ ] Store page approved by Valve (submit 3 weeks before launch for review)
- [ ] SteamPipe: Windows + Linux depots uploaded to `default` branch
- [ ] Launch date set in Steamworks

### 18.2 Server Launch Prep
- [ ] Scale backend: auto-scaling configured for 10× normal traffic
- [ ] Load test: simulate 1,000 concurrent score submissions
- [ ] Database connections: PgBouncer pool sized for launch traffic
- [ ] Redis: cache warmed for daily leaderboard query
- [ ] CDN: WASM bundle pre-distributed to edge nodes

### 18.3 Day-1 Monitoring
- [ ] Grafana dashboard open: watch error rate, latency, CPU
- [ ] Steam review feed monitored: respond to negative reviews within 6h
- [ ] Discord #bug-reports monitored: fast acknowledgement of reported bugs
- [ ] Crash report queue monitored: any P0 crash → hotfix within 24h
- [ ] Steam forums monitored: pin known issues thread if needed

---

## 19. Post-Launch & Live Ops

### 19.1 Patch Cadence
- [ ] Hotfixes: within 48h for P0 (crash, data loss) bugs
- [ ] Minor patches: every 2 weeks (balance tweaks, bug fixes)
- [ ] Content patches: every 6–8 weeks (new characters, new biome content, new relics)
- [ ] Major updates: every 4–6 months (new systems, Season updates)

### 19.2 Content Roadmap (Post-Launch)
- [ ] **v1.1 — The Collector Update**: 2 new characters, 3 new relics, Armory cosmetics expansion
- [ ] **v1.2 — New Biome: Celestial Spire**: Levels 70–80 biome, 2 new enemy variants, new boss
- [ ] **v1.3 — Endless Mode**: A true infinite mode (unwinnable waves) with global high-score board
- [ ] **v1.4 — Challenge Runs**: Daily+ weekly curated challenge levels with unique modifiers
- [ ] **v2.0 — Co-Op** (if feasible): 2-player online co-op — each player controls half the creatures

### 19.3 Community Management
- [ ] Respond to all Steam reviews (positive + negative) within 1 week
- [ ] Monthly developer update post on Steam: progress, upcoming content, community highlights
- [ ] Community screenshot contest: monthly best-run contest with Shard rewards
- [ ] Bug tracker: public GitHub Issues or canny.io board for community bug reports
- [ ] Feature voting: canny.io board for community feature requests

### 19.4 Analytics Review Cycle
- [ ] Weekly: funnel review (Level 1 start → Level 10 complete conversion)
- [ ] Weekly: top-played characters, least-played characters → balance signal
- [ ] Weekly: wave buff card pick rates → underused cards flagged for buff
- [ ] Monthly: relic usage distribution → underused relics flagged for buff
- [ ] Monthly: essence economy review (are players regularly going broke? Hoarding?)
- [ ] Quarterly: retention curve (Day 1, Day 7, Day 30 return rate)

### 19.5 Long-Term Operations
- [ ] SSL certificate renewal automation (Let's Encrypt auto-renew)
- [ ] Dependency security updates: audit `vcpkg.json` + npm packages monthly
- [ ] Database maintenance: VACUUM + ANALYZE scheduled weekly
- [ ] Backup restore drill: test backup restore quarterly
- [ ] Season rollover: automated at season end (archive leaderboard, distribute rewards, reset)
- [ ] Store page refresh: update screenshots + description after major content updates

---

## Summary: Milestone Gates

```
MILESTONE 1 — PLAYABLE PROTOTYPE
  § 1   Foundation complete
  § 2   Native PC build (Windows) confirmed
  § 3   Engine layer complete
  § 4.1–4.4 Combat + evolution working
  § 5.2 Brix + Grub only
  Platforms: Windows only
  → Gate: Level 1 playable, winnable, no crashes

MILESTONE 2 — VERTICAL SLICE (Levels 1–10)
  § 4   All gameplay systems
  § 5   All 10 characters + 8 enemies
  § 6   All 3 shops
  § 7   Full HUD + screens
  § 8   Audio (placeholder OK)
  § 2C  WASM build running in browser
  Platforms: Windows + Browser
  → Gate: Levels 1–10 fun, internally playtested, balanced

MILESTONE 3 — ALPHA
  § 2B  Android build launching on device
  § 2D  macOS build compiling
  § 9   Art in progress (50%)
  § 10  Audio draft complete
  § 11  Steamworks basics (App ID, achievements stub)
  § 12  Backend API v1 live (staging)
  § 13  Unit tests passing + internal playtest Round 1
  Platforms: Windows + Android + Browser
  → Gate: 0 P0 bugs, 25-tester alpha on Steam beta + Android Internal Testing

MILESTONE 4 — BETA
  § 9   Art 100% complete
  § 10  Audio 100% complete (all SFX + music)
  § 2B  Android touch controls polished
  § 2C  WASM CDN deployed
  § 2D  macOS notarized universal binary
  § 11  Full Steamworks + Google Play Games integration
  § 15  Accessibility + EN localization
  § 16  Legal complete (EULA, privacy policy, all age ratings)
  § 13  Beta playtesting (100 testers across all platforms)
  § 14  Performance targets met on all platforms
  Platforms: Windows + Linux + macOS + Android + Browser (all green)
  → Gate: Steam Deck Verified, Android Play Store Open Testing, beta feedback addressed

MILESTONE 5 — GOLD (LAUNCH)
  § 17  Steam store page live + wishlisting (Steam)
        Google Play store listing ready (Android)
        NoobyGame page live (Browser)
  § 18  Launch operations ready (backend scaled, monitoring on)
  Platforms: ALL
  → Gate: All P0/P1 bugs resolved, Valve store approval received,
           Google Play review approved, WASM deploy confirmed

MILESTONE 6 — POST-LAUNCH
  § 19  Live ops running across all platforms
  → v1.1 content patch ships to ALL platforms simultaneously within 8 weeks
```

---

## Platform Checklist at Each Milestone

| Check | M1 | M2 | M3 | M4 | M5 |
|---|---|---|---|---|---|
| Windows builds & runs | ✅ | ✅ | ✅ | ✅ | ✅ |
| Linux builds & runs | — | — | ✅ | ✅ | ✅ |
| macOS universal binary | — | — | ✅ | ✅ | ✅ |
| WASM runs in Chrome | — | ✅ | ✅ | ✅ | ✅ |
| Android APK installs & runs | — | — | ✅ | ✅ | ✅ |
| Steam Deck Verified | — | — | — | ✅ | ✅ |
| Google Play Open Testing | — | — | — | ✅ | ✅ |
| Steam store page live | — | — | — | — | ✅ |
| Google Play Production | — | — | — | — | ✅ |
| NoobyGame browser live | — | ✅ | ✅ | ✅ | ✅ |

---

*Tower Swarm — Production TODO v1.1 | 2026-03-15*
*Platforms: Steam (Win/Linux/macOS/Deck) + Google Play (Android) + Browser (WASM)*
*Reference GDD: TowerSwarm-GDD.md | Design TODO: TowerSwarm-TODO.md*
