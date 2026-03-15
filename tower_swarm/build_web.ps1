param(
  [switch]$Play
)

$ErrorActionPreference = 'Stop'

function Start-NoobyGameDevServer {
  param(
    [Parameter(Mandatory = $true)]
    [string]$RepoRootPath
  )

  $webAiDir = (Resolve-Path (Join-Path $RepoRootPath 'frontend/webai')).Path
  $gamesUrl = 'http://localhost:4303/games/tower-swarm'

  $serveCommand = "cd /d `"$webAiDir`" && npm run start:noobygame"
  $serveProcess = Start-Process -FilePath 'cmd.exe' -ArgumentList '/k', $serveCommand -PassThru

  Write-Host "Started noobygame dev server in a new terminal window (PID: $($serveProcess.Id))."
  Write-Host "Waiting for http://localhost:4303 ..."

  $isReady = $false
  for ($attempt = 0; $attempt -lt 75; $attempt++) {
    Start-Sleep -Milliseconds 800
    try {
      Invoke-WebRequest -Uri 'http://localhost:4303' -UseBasicParsing -TimeoutSec 2 | Out-Null
      $isReady = $true
      break
    } catch {
      # Keep polling until timeout.
    }
  }

  if ($isReady) {
    Start-Process $gamesUrl
    Write-Host "Opened $gamesUrl"
  } else {
    Write-Host "Server is still starting. Open $gamesUrl manually once ready."
  }
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$scriptPath = (Resolve-Path $MyInvocation.MyCommand.Path).Path
$gamesRoot = (Resolve-Path (Join-Path $scriptDir '..')).Path
$repoRoot = (Resolve-Path (Join-Path $gamesRoot '..')).Path
$emsdkEnvScript = Join-Path $repoRoot 'Scripts/emsdk/emsdk_env.bat'
$emsdkDir = Split-Path -Parent $emsdkEnvScript
$emsdkBat = Join-Path $emsdkDir 'emsdk.bat'

$outDir = Join-Path $repoRoot 'frontend/webai/projects/noobygame/public/wasm/tower-swarm'

$engineCpp = Join-Path $gamesRoot 'ATMEngine.cpp'
$engineInclude = $gamesRoot
$gameInclude = Join-Path $scriptDir 'src'

$sources = @(
  (Join-Path $scriptDir 'src/tower_swarm_main.cpp')
  (Join-Path $scriptDir 'src/TowerSwarmGame.cpp')
  (Join-Path $scriptDir 'src/InputManager.cpp')
  (Join-Path $scriptDir 'src/CameraController.cpp')
  (Join-Path $scriptDir 'src/entities/BaseEntity.cpp')
  (Join-Path $scriptDir 'src/entities/CreatureContainer.cpp')
  (Join-Path $scriptDir 'src/entities/EnemyContainer.cpp')
  (Join-Path $scriptDir 'src/entities/ProjectileContainer.cpp')
  (Join-Path $scriptDir 'src/entities/PickupContainer.cpp')
  (Join-Path $scriptDir 'src/entities/TileContainer.cpp')
  (Join-Path $scriptDir 'src/levels/WaveSpawner.cpp')
  (Join-Path $scriptDir 'src/levels/LevelManager.cpp')
  (Join-Path $scriptDir 'src/levels/SaveState.cpp')
  (Join-Path $scriptDir 'src/screens/HUD.cpp')
  (Join-Path $scriptDir 'src/shop/WaveBuffShop.cpp')
  $engineCpp
)

foreach ($src in $sources) {
  if (-not (Test-Path $src)) {
    throw "Source file missing: $src"
  }
}

if (-not (Test-Path $engineCpp)) {
  throw "Engine source missing: $engineCpp"
}

if (-not (Get-Command em++ -ErrorAction SilentlyContinue)) {
  if (-not (Test-Path $emsdkEnvScript)) {
    throw "em++ was not found in PATH and expected emsdk env script is missing: $emsdkEnvScript. Run Scripts\\download_emscript.bat first."
  }

  if (-not (Test-Path $emsdkBat)) {
    throw "emsdk bootstrap helper is missing: $emsdkBat. Run Scripts\\download_emscript.bat first."
  }

  $playFlag = if ($Play) { '-Play' } else { '' }

  if (-not $env:TOWER_SWARM_EMSDK_BOOTSTRAPPED) {
    Write-Host "em++ is not in PATH. Bootstrapping Emscripten from:"
    Write-Host " - $emsdkEnvScript"

    $cmdLine = "call `"$emsdkEnvScript`" >nul && set TOWER_SWARM_EMSDK_BOOTSTRAPPED=1 && powershell -NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`" $playFlag"
    & cmd.exe /c $cmdLine
    exit $LASTEXITCODE
  }

  if (-not $env:TOWER_SWARM_EMSDK_ACTIVATED) {
    Write-Host 'Emscripten environment is loaded, but em++ is still unavailable.'
    Write-Host "Running toolchain setup in: $emsdkDir"

    $cmdLine = "cd /d `"$emsdkDir`" && call `"$emsdkBat`" install latest && call `"$emsdkBat`" activate latest && call `"$emsdkEnvScript`" >nul && set TOWER_SWARM_EMSDK_BOOTSTRAPPED=1 && set TOWER_SWARM_EMSDK_ACTIVATED=1 && powershell -NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`" $playFlag"
    & cmd.exe /c $cmdLine
    exit $LASTEXITCODE
  }

  throw "em++ was not found in PATH after bootstrap + activate from $emsdkDir. Run Scripts\\download_emscript.bat and verify $emsdkDir\\upstream\\emscripten\\em++.bat exists."
}

New-Item -ItemType Directory -Force $outDir | Out-Null
$outDir = (Resolve-Path $outDir).Path

Write-Host 'Building Tower Swarm SDL3 WASM bundle...'
Write-Host "Output: $outDir"

$outFile = Join-Path $outDir 'tower_swarm.js'
$emArgs = @(
  @($sources)
  '-I'
  $engineInclude
  '-I'
  $gameInclude
  '-std=c++20'
  '-O2'
  '-sUSE_SDL=3'
  '-sALLOW_MEMORY_GROWTH=1'
  '-sASSERTIONS=1'
  '-sEXPORTED_RUNTIME_METHODS=["ccall","cwrap"]'
  '-sENVIRONMENT=web'
  '-o'
  $outFile
)

& em++ @emArgs

if ($LASTEXITCODE -ne 0) {
  throw "Tower Swarm WASM build failed with exit code $LASTEXITCODE"
}

Write-Host 'Done. Generated files:'
Write-Host " - $(Join-Path $outDir 'tower_swarm.js')"
Write-Host " - $(Join-Path $outDir 'tower_swarm.wasm')"

if ($Play) {
  Start-NoobyGameDevServer -RepoRootPath $repoRoot
}
