# Meteor Dodge (SDL3 + C++ + WASM)

This game uses the Attome C++ engine (`games/engine`) and is compiled to WebAssembly with Emscripten.

## Quick Start (One Command)

One-time emsdk setup (installs to `Scripts/emsdk` and saves `EMSDK`):

```bat
Scripts\download_emscript.bat
```

From `games/meteor_dodge`:

```powershell
.\build_web.ps1 -Play
```

Or from `AttomeAngular/webai`:

```powershell
npm run play:meteor
```

What this does:

- builds `meteor_dodge.js/.wasm`
- starts `noobygame` dev server (`npm run start:noobygame`) in a new terminal window
- opens `http://localhost:4303/games` when the server is ready

`build_web.ps1` assumes emsdk is installed in `Scripts/emsdk` and bootstraps from:

- `Scripts/emsdk/emsdk_env.bat`
- if `em++` is still missing after bootstrap, it auto-runs `emsdk.bat install latest` + `emsdk.bat activate latest` once and retries

## Build Only

Run:

```powershell
games/meteor_dodge/build_web.ps1
```

The script writes the runtime bundle to:

- `AttomeAngular/webai/projects/noobygame/public/wasm/meteor-dodge/meteor_dodge.js`
- `AttomeAngular/webai/projects/noobygame/public/wasm/meteor-dodge/meteor_dodge.wasm`

## Runtime Entry

Angular route `/games/meteor-dodge` loads:

- `/wasm/meteor-dodge/index.html`

The page then boots `meteor_dodge.js` and attaches SDL3 rendering to the canvas.
