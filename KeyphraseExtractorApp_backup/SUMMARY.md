# Keyphrase Extractor App - Summary

## Overview

This is a React Native application built with Expo that serves as a frontend for your keyphrase extraction system. The app can be run as both a mobile app (iOS/Android) and a web application.

## Key Features

- Extract keyphrases from any text
- View extracted keyphrases with confidence scores
- Sort keyphrases by score or alphabetically
- Save extraction history
- Configure API settings

## Project Structure

```
KeyphraseExtractorApp/
├── App.tsx                 # Main application component with navigation
├── src/
│   ├── components/         # Reusable UI components
│   │   ├── Header.tsx      # App header with navigation actions
│   │   ├── KeyphraseItem.tsx # Component to display a keyphrase
│   │   └── LoadingOverlay.tsx # Loading indicator for API calls
│   ├── screens/            # App screens
│   │   ├── HomeScreen.tsx  # Main screen with text input
│   │   ├── ResultsScreen.tsx # Display extracted keyphrases
│   │   ├── HistoryScreen.tsx # View past extractions
│   │   └── SettingsScreen.tsx # Configure API URL
│   ├── services/           # API and other services
│   │   └── api.ts          # API integration for keyphrase extraction
│   └── utils/              # Utility functions
│       └── storage.ts      # Local storage for history and settings
├── README.md               # Documentation
├── install_dependencies.bat # Script to install dependencies
└── run_app.bat             # Script to run the app
```

## Installation and Setup

1. Run the installation script:
   ```
   install_dependencies.bat
   ```

2. Run the app:
   ```
   run_app.bat
   ```

3. Configure the API URL in the Settings screen to point to your backend API.

## API Integration

The app expects your backend API to have an endpoint at `/extract_keyphrases` that accepts POST requests with a JSON body containing a `text` field and returns a JSON response with a `keyphrases` array of objects, each with `keyphrase` and `score` fields.

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

## Dependency Fixes

The original project had dependency conflicts between Expo and React Navigation versions. The following changes were made to fix these issues:

1. Updated package.json with compatible versions:
   - Downgraded Expo from 52.x to 49.0.15
   - Updated React Navigation to v6.x
   - Updated other dependencies to match these versions

2. Modified navigation imports in all screen components

3. Created installation script with `--legacy-peer-deps` flag to handle remaining peer dependency issues

## Next Steps

1. Test the app with your backend API
2. Customize the UI to match your branding
3. Add additional features as needed
4. Consider deploying the app to app stores or as a web application
