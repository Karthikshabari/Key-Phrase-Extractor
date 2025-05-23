import { Platform } from 'react-native';

// Define font families
export const fontConfig = {
  web: {
    regular: {
      fontFamily: 'Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
      fontWeight: '400' as const,
    },
    medium: {
      fontFamily: 'Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
      fontWeight: '500' as const,
    },
    light: {
      fontFamily: 'Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
      fontWeight: '300' as const,
    },
    thin: {
      fontFamily: 'Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
      fontWeight: '100' as const,
    },
  },
  ios: {
    regular: {
      fontFamily: 'System',
      fontWeight: '400' as const,
    },
    medium: {
      fontFamily: 'System',
      fontWeight: '500' as const,
    },
    light: {
      fontFamily: 'System',
      fontWeight: '300' as const,
    },
    thin: {
      fontFamily: 'System',
      fontWeight: '100' as const,
    },
  },
  android: {
    regular: {
      fontFamily: 'sans-serif',
      fontWeight: 'normal' as const,
    },
    medium: {
      fontFamily: 'sans-serif-medium',
      fontWeight: 'normal' as const,
    },
    light: {
      fontFamily: 'sans-serif-light',
      fontWeight: 'normal' as const,
    },
    thin: {
      fontFamily: 'sans-serif-thin',
      fontWeight: 'normal' as const,
    },
  },
};

// Define typography scale
export const typography = {
  h1: {
    fontSize: 28,
    lineHeight: 36,
    fontWeight: Platform.OS === 'ios' ? '700' : 'bold',
    letterSpacing: 0.25,
  },
  h2: {
    fontSize: 24,
    lineHeight: 32,
    fontWeight: Platform.OS === 'ios' ? '600' : 'bold',
    letterSpacing: 0,
  },
  h3: {
    fontSize: 20,
    lineHeight: 28,
    fontWeight: Platform.OS === 'ios' ? '600' : 'bold',
    letterSpacing: 0.15,
  },
  subtitle1: {
    fontSize: 16,
    lineHeight: 24,
    fontWeight: Platform.OS === 'ios' ? '500' : 'normal',
    letterSpacing: 0.15,
  },
  subtitle2: {
    fontSize: 14,
    lineHeight: 22,
    fontWeight: Platform.OS === 'ios' ? '500' : 'normal',
    letterSpacing: 0.1,
  },
  body1: {
    fontSize: 16,
    lineHeight: 24,
    fontWeight: 'normal',
    letterSpacing: 0.5,
  },
  body2: {
    fontSize: 14,
    lineHeight: 22,
    fontWeight: 'normal',
    letterSpacing: 0.25,
  },
  button: {
    fontSize: 14,
    lineHeight: 20,
    fontWeight: Platform.OS === 'ios' ? '600' : 'bold',
    letterSpacing: 1.25,
    textTransform: 'uppercase',
  },
  caption: {
    fontSize: 12,
    lineHeight: 16,
    fontWeight: 'normal',
    letterSpacing: 0.4,
  },
  overline: {
    fontSize: 10,
    lineHeight: 16,
    fontWeight: Platform.OS === 'ios' ? '500' : 'normal',
    letterSpacing: 1.5,
    textTransform: 'uppercase',
  },
};

// Get the font configuration for the current platform
export const getFontConfig = () => {
  if (Platform.OS === 'web') {
    return fontConfig.web;
  } else if (Platform.OS === 'ios') {
    return fontConfig.ios;
  } else {
    return fontConfig.android;
  }
};
