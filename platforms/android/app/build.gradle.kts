// =============================================================================
// app/build.gradle.kts — Attome Game Android App Module
//
// GAME_NAME is passed from build-android.ps1 as a Gradle property:
//   gradlew assembleDebug -PGAME_NAME=snake
//
// It controls:
//   - Which game's CMakeLists.txt is built
//   - The applicationId / label
// =============================================================================
plugins {
    alias(libs.plugins.android.application)
}

val gameName: String = project.findProperty("GAME_NAME")?.toString() ?: "hello-world"
val gamesRoot: String = project.findProperty("GAMES_ROOT")?.toString()
    ?: rootProject.projectDir.parentFile.parentFile.absolutePath

android {
    namespace = "org.attome.game"
    compileSdk = 34

    defaultConfig {
        applicationId = "org.attome.game.$gameName"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        // CMake / NDK config
        externalNativeBuild {
            cmake {
                cppFlags("-std=c++20")
                arguments(
                    "-DGAME_NAME=$gameName",
                    "-DGAMES_ROOT=$gamesRoot",
                    // vcpkg toolchain is chained via VCPKG_CHAINLOAD in build-android.ps1
                    // and passed through cmake invocation; Gradle picks it up from
                    // the cmake block below.
                    "-DANDROID_STL=c++_shared",
                    "-DANDROID_PLATFORM=android-24"
                )
            }
        }

        ndk {
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }

    externalNativeBuild {
        cmake {
            path = file("CMakeLists.txt")
            version = "3.22.1"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    // Make the built .so visible to Gradle's packaging step
    sourceSets {
        getByName("main") {
            jniLibs.srcDirs("src/main/jniLibs")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}
