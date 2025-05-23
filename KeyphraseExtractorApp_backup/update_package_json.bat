@echo off
echo ===== Updating package.json to Expo 49 =====

echo Creating backup of package.json...
copy package.json package.json.bak

echo Updating package.json...
echo {> package.json
echo   "name": "keyphraseextractorapp",>> package.json
echo   "version": "1.0.0",>> package.json
echo   "main": "index.ts",>> package.json
echo   "scripts": {>> package.json
echo     "start": "expo start",>> package.json
echo     "android": "expo run:android",>> package.json
echo     "ios": "expo run:ios",>> package.json
echo     "web": "expo start --web">> package.json
echo   },>> package.json
echo   "dependencies": {>> package.json
echo     "@expo/webpack-config": "^18.0.1",>> package.json
echo     "@react-native-async-storage/async-storage": "1.18.2",>> package.json
echo     "@react-navigation/native": "^6.1.9",>> package.json
echo     "@react-navigation/native-stack": "^6.9.17",>> package.json
echo     "@react-navigation/stack": "^6.3.20",>> package.json
echo     "axios": "^1.6.2",>> package.json
echo     "buffer": "^6.0.3",>> package.json
echo     "crypto-browserify": "^3.12.1",>> package.json
echo     "expo": "~49.0.15",>> package.json
echo     "expo-constants": "~14.4.2",>> package.json
echo     "expo-file-system": "~15.4.5",>> package.json
echo     "expo-haptics": "~12.4.0",>> package.json
echo     "expo-linear-gradient": "~12.3.0",>> package.json
echo     "expo-linking": "~5.0.2",>> package.json
echo     "expo-screen-capture": "~5.3.0",>> package.json
echo     "expo-status-bar": "~1.6.0",>> package.json
echo     "expo-task-manager": "~11.3.0",>> package.json
echo     "react": "18.2.0",>> package.json
echo     "react-dom": "18.2.0",>> package.json
echo     "react-native": "0.72.6",>> package.json
echo     "react-native-gesture-handler": "~2.12.0",>> package.json
echo     "react-native-paper": "^5.11.1",>> package.json
echo     "react-native-reanimated": "~3.3.0",>> package.json
echo     "react-native-safe-area-context": "4.6.3",>> package.json
echo     "react-native-screens": "~3.22.0",>> package.json
echo     "react-native-vector-icons": "^10.0.0",>> package.json
echo     "react-native-web": "~0.19.6",>> package.json
echo     "stream-browserify": "^3.0.0",>> package.json
echo     "vm-browserify": "^1.1.2">> package.json
echo   },>> package.json
echo   "devDependencies": {>> package.json
echo     "@babel/core": "^7.20.0",>> package.json
echo     "@react-native-community/cli": "latest",>> package.json
echo     "@types/react": "~18.2.14",>> package.json
echo     "typescript": "^5.1.3">> package.json
echo   },>> package.json
echo   "private": true,>> package.json
echo   "browser": {>> package.json
echo     "crypto": "crypto-browserify",>> package.json
echo     "stream": "stream-browserify",>> package.json
echo     "buffer": "buffer",>> package.json
echo     "vm": "vm-browserify">> package.json
echo   }>> package.json
echo }>> package.json

echo ===== Package.json updated to Expo 49 =====
echo.
echo Please run 'npm install --force' to install the updated dependencies
echo.
pause
