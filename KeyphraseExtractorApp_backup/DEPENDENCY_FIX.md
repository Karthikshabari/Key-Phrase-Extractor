# Dependency Conflict Resolution

This document explains the changes made to resolve the dependency conflicts in the Keyphrase Extractor App.

## Original Error

```
npm ERR! code ERESOLVE
npm ERR! ERESOLVE unable to resolve dependency tree
npm ERR!
npm ERR! While resolving: keyphraseextractorapp@1.0.0
npm ERR! Found: expo@49.0.23
npm ERR! node_modules/expo
npm ERR!   expo@"~49.0.15" from the root project
npm ERR!
npm ERR! Could not resolve dependency:
npm ERR! peer expo@"^48.0.17" from @expo/webpack-config@18.1.4
npm ERR! node_modules/@expo/webpack-config
npm ERR!   @expo/webpack-config@"^18.0.1" from the root project
```

## Solution

The issue was a version mismatch between Expo 49 and @expo/webpack-config 18, which requires Expo 48. We had two options:

1. Downgrade Expo to version 48
2. Upgrade both packages to compatible versions

We chose option 2 and upgraded to Expo 50 with webpack-config 19.

## Changes Made

1. Updated `@expo/webpack-config` from `^18.0.1` to `^19.0.0`
2. Updated `expo` from `~49.0.15` to `~50.0.0`
3. Updated all Expo-related packages to match Expo 50:
   - `expo-linking`: `~5.0.2` → `~6.0.0`
   - `expo-status-bar`: `~1.6.0` → `~1.11.0`

4. Updated React and React Native to match Expo 50:
   - `react`: `18.2.0` → `18.3.0`
   - `react-dom`: `18.2.0` → `18.3.0`
   - `react-native`: `0.72.6` → `0.73.4`

5. Updated React Native packages:
   - `react-native-safe-area-context`: `4.6.3` → `4.8.2`
   - `react-native-screens`: `~3.22.0` → `~3.29.0`
   - `react-native-web`: `~0.19.6` → `~0.19.10`

6. Updated React Navigation packages:
   - `@react-navigation/native`: `^6.1.9` → `^6.1.14`
   - `@react-navigation/native-stack`: `^6.9.17` → `^6.9.22`
   - `@react-navigation/stack`: `^6.3.20` → `^6.3.25`

7. Updated other dependencies:
   - `@react-native-async-storage/async-storage`: `1.18.2` → `1.21.0`
   - `react-native-paper`: `^5.11.1` → `^5.12.3`
   - `@types/react`: `~18.2.14` → `~18.2.45`
   - `typescript`: `^5.1.3` → `^5.3.0`

8. Changed installation method from `--legacy-peer-deps` to `--force`

9. Created a `clean_install.bat` script to perform a clean installation by:
   - Removing node_modules folder
   - Removing package-lock.json
   - Installing dependencies with `--force` flag

## Installation Instructions

To install the app with the fixed dependencies:

1. Run the clean installation script:
   ```
   clean_install.bat
   ```

2. Run the app:
   ```
   run_app.bat
   ```

## Why This Works

The `--force` flag tells npm to proceed with the installation even if there are peer dependency conflicts. This is necessary because some packages in the React Native ecosystem may not have updated their peer dependency requirements yet.

By upgrading to the latest compatible versions of all packages, we minimize the risk of runtime errors despite forcing the installation.
