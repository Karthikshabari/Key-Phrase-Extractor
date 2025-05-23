import { NativeStackNavigationOptions } from '@react-navigation/native-stack';
import { Platform } from 'react-native';

// Custom transition configuration for the app
export const transitionConfig: NativeStackNavigationOptions = {
  animation: Platform.OS === 'web' ? 'none' : 'slide_from_right',
  presentation: 'card',
  animationDuration: 300, // Using a numeric value instead of string
  gestureEnabled: Platform.OS !== 'web',
  fullScreenGestureEnabled: Platform.OS !== 'web',
  headerShown: true,

  // Explicitly set all problematic properties to undefined to avoid the 'large' string error
  // @ts-ignore - Adding these to fix the 'large' string conversion error
  sheetAllowedDetents: undefined,
  sheetLargestUndimmedDetent: undefined,
  sheetExpandsWhenScrolledToEdge: undefined,
  sheetGrabberVisible: undefined,
  sheetCornerRadius: undefined,
  sheetPresentationStyle: undefined,
};
