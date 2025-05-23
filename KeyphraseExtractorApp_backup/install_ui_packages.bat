@echo off
echo Installing UI enhancement packages...
cd KeyphraseExtractorApp
npm install expo-linear-gradient@~12.7.2 expo-file-system@~16.0.5 react-native-reanimated@~3.6.2 react-native-gesture-handler@~2.14.0 --force
echo UI packages installed successfully!
echo.
echo Now try running the app again with:
echo npm run web
echo.
pause
