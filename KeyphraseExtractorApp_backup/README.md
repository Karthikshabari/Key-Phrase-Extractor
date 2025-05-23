# Keyphrase Extractor App

A React Native application for extracting keyphrases from text using a backend API.

## Features

- Extract keyphrases from any text
- View extracted keyphrases with confidence scores
- Sort keyphrases by score or alphabetically
- Save extraction history
- Configure API settings
- Debug mode for troubleshooting

## Getting Started

### Prerequisites

- Node.js (v14 or later)
- npm or yarn
- Expo CLI

### Installation

1. Clone this repository
2. Run the clean installation script (recommended):
   ```
   clean_install.bat
   ```

   Or use the regular installation script:
   ```
   install_dependencies.bat
   ```

   Or manually install dependencies:
   ```
   cd KeyphraseExtractorApp
   npm install --force
   ```

3. Install polyfill packages for web compatibility:
   ```
   install_vm_polyfill.bat
   ```
   or
   ```
   cd KeyphraseExtractorApp
   npm install --save vm-browserify --force
   ```

### Running the App

Use the run script:
```
run_app.bat
```

Or run manually:

#### Web
```
npm run web
```

#### Android
```
npm run android
```

#### iOS
```
npm run ios
```

For mobile apps, scan the QR code with the Expo Go app on your device.

### API Configuration

Before using the app, you need to configure the API URL:

1. Start your keyphrase extraction API backend
2. The default API URL is set to: `https://8e6f-34-82-156-92.ngrok-free.app`
3. If your API URL changes, you can update it in the Settings screen

## API Response Format

The app expects the API to return keyphrases in the following format:

```json
{
  "keyphrases": [
    ["keyphrase1", 0.85],
    ["keyphrase2", 0.72],
    ["keyphrase3", 0.65]
  ]
}
```

Each keyphrase is an array with:
- First element: the keyphrase text (string)
- Second element: the confidence score (number between 0 and 1)

## Troubleshooting

If you encounter issues:

1. Enable Debug Mode in the Settings screen to see detailed logs
2. Check the browser console for error messages
3. Verify your API is running and accessible
4. Make sure your API returns data in the expected format

## Project Structure

```
KeyphraseExtractorApp/
├── App.tsx                 # Main application component
├── src/
│   ├── components/         # Reusable UI components
│   │   ├── Header.tsx
│   │   ├── KeyphraseItem.tsx
│   │   └── LoadingOverlay.tsx
│   ├── screens/            # App screens
│   │   ├── HomeScreen.tsx
│   │   ├── ResultsScreen.tsx
│   │   ├── HistoryScreen.tsx
│   │   └── SettingsScreen.tsx
│   ├── services/           # API and other services
│   │   └── api.ts
│   └── utils/              # Utility functions
│       └── storage.ts
```

## API Integration

The app expects the backend API to have an endpoint at `/extract_keyphrases` that accepts POST requests with a JSON body containing a `text` field and returns a JSON response with a `keyphrases` array of objects, each with `keyphrase` and `score` fields.

Example request:
```json
{
  "text": "Your text to extract keyphrases from"
}
```

Example response:
```json
{
  "keyphrases": [
    {
      "keyphrase": "example keyphrase",
      "score": 0.85
    },
    {
      "keyphrase": "another keyphrase",
      "score": 0.72
    }
  ]
}
```

## License

This project is licensed under the MIT License.
