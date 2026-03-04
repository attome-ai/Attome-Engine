package org.attome.game;

import org.libsdl.app.SDLActivity;

/**
 * GameActivity — thin wrapper around SDL3's SDLActivity.
 *
 * SDL3 handles the entire game loop; this class just:
 *  - Declares the native libraries to load (libmain.so built by CMake)
 *  - Lets SDL3's Java glue take over from there
 *
 * Touch input is forwarded to SDL as mouse/finger events automatically.
 */
public class GameActivity extends SDLActivity {

    /**
     * Return the names of shared libraries to load.
     * SDL will load these after its own runtime.
     * "main" → libmain.so (OUTPUT_NAME "main" set in CMakeLists.txt)
     */
    @Override
    protected String[] getLibraries() {
        return new String[] {
            "SDL3",          // SDL3 shared lib (from vcpkg / SDL3 AAR)
            "main"           // your game's libmain.so
        };
    }
}
