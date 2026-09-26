# Builds and runs the standalone test programs (engine + Attome Online).
#
# The tests are separate executables (atm_tests.exe, ao_tests.exe): no test
# code is compiled into the game, server or bots, so running them never
# affects the game or its performance. The game does not need to be closed.
#
#   .\run_tests.ps1              build (Release) and run all tests
#   .\run_tests.ps1 -NoBuild     run the last build only
#
# Exit code 0 = all tests passed.

param([switch]$NoBuild, [string]$BuildDir = "build-online", [string]$Config = "Release")

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not $NoBuild) {
  # Make sure the test targets exist in this build tree (no-op if already on).
  cmake $BuildDir -DATTOME_BUILD_TESTS=ON | Out-Null
  cmake --build $BuildDir --config $Config --target atm_tests ao_tests -- /m /v:minimal
  if ($LASTEXITCODE -ne 0) { Write-Host "Test build failed." -ForegroundColor Red; exit 1 }
}

ctest --test-dir $BuildDir -C $Config --output-on-failure
if ($LASTEXITCODE -eq 0) { Write-Host "All tests passed." -ForegroundColor Green }
else { Write-Host "Some tests FAILED (see above)." -ForegroundColor Red }
exit $LASTEXITCODE
