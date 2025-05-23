// This file is used to load icon fonts for web
import { Platform } from 'react-native';

// Only load fonts on web platform
if (Platform.OS === 'web') {
  // Load all icon fonts
  const IconFontStyles = `
    @font-face {
      src: url(https://cdn.jsdelivr.net/npm/react-native-vector-icons@10.0.0/Fonts/MaterialCommunityIcons.ttf) format('truetype');
      font-family: 'MaterialCommunityIcons';
      font-display: block;
    }
    @font-face {
      src: url(https://cdn.jsdelivr.net/npm/react-native-vector-icons@10.0.0/Fonts/MaterialIcons.ttf) format('truetype');
      font-family: 'MaterialIcons';
      font-display: block;
    }
    @font-face {
      src: url(https://cdn.jsdelivr.net/npm/react-native-vector-icons@10.0.0/Fonts/FontAwesome.ttf) format('truetype');
      font-family: 'FontAwesome';
      font-display: block;
    }
    @font-face {
      src: url(https://cdn.jsdelivr.net/npm/react-native-vector-icons@10.0.0/Fonts/Ionicons.ttf) format('truetype');
      font-family: 'Ionicons';
      font-display: block;
    }
  `;

  // Create stylesheet
  const style = document.createElement('style');
  style.type = 'text/css';
  if (style.styleSheet) {
    style.styleSheet.cssText = IconFontStyles;
  } else {
    style.appendChild(document.createTextNode(IconFontStyles));
  }

  // Inject stylesheet
  document.head.appendChild(style);
}
