@echo off
echo ===== Keyphrase Extractor App Fix Script =====

echo Cleaning up node_modules...
rmdir /s /q node_modules

echo Cleaning up package-lock.json...
del package-lock.json

echo Cleaning up yarn.lock if it exists...
if exist yarn.lock del yarn.lock

echo Cleaning up .expo folder if it exists...
if exist .expo rmdir /s /q .expo

echo Installing dependencies with legacy peer deps...
npm install --legacy-peer-deps

echo Installing react-native-vector-icons...
npm install react-native-vector-icons@10.0.0 --legacy-peer-deps

echo Updating expo-linking and react-native...
npm install expo-linking@~6.2.2 react-native@0.73.6 --legacy-peer-deps

echo ===== Fix completed! =====
echo.
echo Please run 'npx expo start --web' to start the app
echo.
pause
