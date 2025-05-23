@echo off
echo ===== Keyphrase Extractor App Expo Downgrade Script =====

echo Cleaning up node_modules...
rmdir /s /q node_modules

echo Cleaning up package-lock.json...
del package-lock.json

echo Cleaning up .expo folder if it exists...
if exist .expo rmdir /s /q .expo

echo Installing Expo 49 (more stable version)...
npm install expo@~49.0.15 --force

echo Installing compatible dependencies...
npm install @react-native-async-storage/async-storage@1.18.2 --force
npm install expo-constants@~14.4.2 --force
npm install expo-file-system@~15.4.5 --force
npm install expo-haptics@~12.4.0 --force
npm install expo-linear-gradient@~12.3.0 --force
npm install expo-linking@~5.0.2 --force
npm install expo-status-bar@~1.6.0 --force
npm install expo-task-manager@~11.3.0 --force
npm install react@18.2.0 react-dom@18.2.0 --force
npm install react-native@0.72.6 --force
npm install react-native-gesture-handler@~2.12.0 --force
npm install react-native-reanimated@~3.3.0 --force
npm install react-native-screens@~3.22.0 --force
npm install @types/react@~18.2.14 --force

echo Installing @react-native-community/cli...
npm install @react-native-community/cli@latest --save-dev --force

echo ===== Downgrade completed! =====
echo.
echo Please run 'npx expo start --web' to start the app
echo.
pause
