param(
  [switch]$Play
)

$ErrorActionPreference = 'Stop'

function Start-NoobyGameDevServer {
  param(
    [Parameter(Mandatory = $true)]
    [string]$RepoRootPath
  )

  $webAiDir = (Resolve-Path (Join-Path $RepoRootPath '../AttomeAngular/webai')).Path
  $gamesUrl = 'http://localhost:4303/games'

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
$repoRoot = (Resolve-Path (Join-Path $scriptDir '..')).Path
$attomeRoot = (Resolve-Path (Join-Path $repoRoot '..')).Path
$emsdkEnvScript = Join-Path $attomeRoot 'Scripts/emsdk/emsdk_env.bat'
$emsdkDir = Split-Path -Parent $emsdkEnvScript
$emsdkBat = Join-Path $emsdkDir 'emsdk.bat'
$outDir = Join-Path $repoRoot '../AttomeAngular/webai/projects/noobygame/public/wasm/meteor-dodge'

$mainCpp = Join-Path $scriptDir 'src/meteor_dodge_main.cpp'
$engineCpp = Join-Path $repoRoot 'engine/ATMEngine.cpp'
$engineDir = Join-Path $repoRoot 'engine'
$libDir = Join-Path $repoRoot 'lib'

if (-not (Get-Command em++ -ErrorAction SilentlyContinue)) {
  if (-not (Test-Path $emsdkEnvScript)) {
    throw "em++ was not found in PATH and expected emsdk env script is missing: $emsdkEnvScript. Run Scripts\download_emscript.bat first."
  }

  if (-not (Test-Path $emsdkBat)) {
    throw "emsdk bootstrap helper is missing: $emsdkBat. Run Scripts\download_emscript.bat first."
  }

  $playFlag = if ($Play) { '-Play' } else { '' }

  if (-not $env:METEOR_DODGE_EMSDK_BOOTSTRAPPED) {
    Write-Host "em++ is not in PATH. Bootstrapping Emscripten from:"
    Write-Host " - $emsdkEnvScript"

    $cmdLine = "call `"$emsdkEnvScript`" >nul && set METEOR_DODGE_EMSDK_BOOTSTRAPPED=1 && powershell -NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`" $playFlag"
    & cmd.exe /c $cmdLine
    exit $LASTEXITCODE
  }

  if (-not $env:METEOR_DODGE_EMSDK_ACTIVATED) {
    Write-Host 'Emscripten environment is loaded, but em++ is still unavailable.'
    Write-Host "Running toolchain setup in: $emsdkDir"

    $cmdLine = "cd /d `"$emsdkDir`" && call `"$emsdkBat`" install latest && call `"$emsdkBat`" activate latest && call `"$emsdkEnvScript`" >nul && set METEOR_DODGE_EMSDK_BOOTSTRAPPED=1 && set METEOR_DODGE_EMSDK_ACTIVATED=1 && powershell -NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`" $playFlag"
    & cmd.exe /c $cmdLine
    exit $LASTEXITCODE
  }

  throw "em++ was not found in PATH after bootstrap + activate from $emsdkDir. Run Scripts\download_emscript.bat and verify $emsdkDir\upstream\emscripten\em++.bat exists."
}

New-Item -ItemType Directory -Force $outDir | Out-Null
$outDir = (Resolve-Path $outDir).Path

Write-Host 'Building Meteor Dodge SDL3 WASM bundle...'
Write-Host "Output: $outDir"

$outFile = Join-Path $outDir 'meteor_dodge.js'
$emArgs = @(
  $mainCpp
  $engineCpp
  '-I'
  $engineDir
  '-I'
  $libDir
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
  throw "Meteor Dodge WASM build failed with exit code $LASTEXITCODE"
}

Write-Host 'Done. Generated files:'
Write-Host " - $(Join-Path $outDir 'meteor_dodge.js')"
Write-Host " - $(Join-Path $outDir 'meteor_dodge.wasm')"

if ($Play) {
  Start-NoobyGameDevServer -RepoRootPath $repoRoot
}
