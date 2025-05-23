// Import navigation patch to fix the "large" string conversion error
import './src/utils/navigationPatch';

import React, { useEffect, useState } from 'react';
import { Platform } from 'react-native';
import { StatusBar } from 'expo-status-bar';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { Provider as PaperProvider, MD3LightTheme } from 'react-native-paper';
import { DefaultTheme } from '@react-navigation/native';
import { Keyphrase } from './src/services/api';
import { getApiUrl } from './src/utils/storage';
import { updateApiUrl } from './src/services/api';

// Import icon fonts for web
import './src/utils/iconFonts';

// Import screens
import HomeScreen from './src/screens/HomeScreen';
import ResultsScreen from './src/screens/ResultsScreen';
import HistoryScreen from './src/screens/HistoryScreen';
import SettingsScreen from './src/screens/SettingsScreen';

// Import components
import Header from './src/components/Header';
import SplashScreen from './src/components/SplashScreen';

// Import navigation configuration
import { transitionConfig } from './src/navigation/TransitionConfig';

// Define the navigation stack parameter list
export type RootStackParamList = {
  Home: undefined;
  Results: { keyphrases: Keyphrase[]; text: string };
  History: undefined;
  Settings: undefined;
};

// Create the navigation stack
const Stack = createNativeStackNavigator<RootStackParamList>();

// Create a custom theme with refined colors
const theme = {
  ...MD3LightTheme,
  colors: {
    ...MD3LightTheme.colors,
    // Primary colors - softer purple palette
    primary: '#7B68EE',        // Medium slate blue - softer than deep purple
    primaryContainer: '#F0EBFF', // Very light purple for containers

    // Secondary colors - teal/aqua for contrast
    secondary: '#00B8A9',      // Softer teal as secondary color
    secondaryContainer: '#E6F9F7', // Very light teal for containers

    // Tertiary colors - soft pink for an analogous feel
    tertiary: '#FF85A2',       // Soft pink accent for highlights
    tertiaryContainer: '#FFF0F3', // Very light pink for containers

    // Accent color - gold for luxury touches
    accent: '#FFD166',         // Gold accent
    accentContainer: '#FFF8E6', // Very light gold for containers

    // Neutral colors
    surface: '#FFFFFF',        // White surface
    surfaceVariant: '#F8F9FA', // Very light gray for variant surfaces
    background: '#F8F9FA',     // Light gray background
    error: '#E63946',          // Refined error color

    // Text colors
    onPrimary: '#FFFFFF',      // White text on primary
    onSecondary: '#FFFFFF',    // White text on secondary
    onTertiary: '#FFFFFF',     // White text on tertiary
    onSurface: '#343A40',      // Dark gray text on surface (not pure black)
    onSurfaceVariant: '#6C757D', // Medium gray text for less emphasis
    onBackground: '#343A40',   // Dark gray text on background
  },
  roundness: 12,               // More rounded corners for a softer look
  animation: {
    scale: 1.0,                // Standard animation scale
  },
  fonts: {
    ...MD3LightTheme.fonts,
    // We'll use system fonts with custom weights
    labelLarge: {
      ...MD3LightTheme.fonts.labelLarge,
      fontWeight: '500' as const,
    },
    titleLarge: {
      ...MD3LightTheme.fonts.titleLarge,
      fontWeight: '700' as const,
      fontSize: 28,
      lineHeight: 36,
    },
    titleMedium: {
      ...MD3LightTheme.fonts.titleMedium,
      fontWeight: '600' as const,
      fontSize: 20,
      lineHeight: 28,
    },
    bodyLarge: {
      ...MD3LightTheme.fonts.bodyLarge,
      fontSize: 16,
      lineHeight: 24,
      letterSpacing: 0.5,
    },
    bodyMedium: {
      ...MD3LightTheme.fonts.bodyMedium,
      fontSize: 14,
      lineHeight: 22,
      letterSpacing: 0.25,
    },
  }
};

// Navigation theme with matching colors
const navigationTheme = {
  ...DefaultTheme,
  colors: {
    ...DefaultTheme.colors,
    primary: theme.colors.primary,
    background: theme.colors.background,
    card: theme.colors.surface,
    text: theme.colors.onSurface,
    border: 'rgba(0,0,0,0.05)',
    notification: theme.colors.tertiary,
  },
};

export default function App() {
  const [isReady, setIsReady] = useState(false);
  const [showSplash, setShowSplash] = useState(true);

  // Load saved API URL when the app starts
  useEffect(() => {
    const loadApiUrl = async () => {
      try {
        const savedUrl = await getApiUrl();
        if (savedUrl) {
          updateApiUrl(savedUrl);
        }
      } catch (error) {
        console.error('Error loading API URL:', error);
      } finally {
        setIsReady(true);
      }
    };

    loadApiUrl();
  }, []);

  // Handle splash screen completion
  const handleSplashComplete = () => {
    setShowSplash(false);
  };

  if (!isReady) {
    return null; // Or a simple loading indicator
  }

  // Show splash screen on first load
  if (showSplash) {
    return <SplashScreen onFinish={handleSplashComplete} />;
  }

  return (
    <SafeAreaProvider>
      <PaperProvider theme={theme}>
        <NavigationContainer theme={navigationTheme}>
          <Stack.Navigator
            initialRouteName="Home"
            screenOptions={{
              header: (props) => <Header {...props} />,
              ...transitionConfig,
              // Override problematic settings that might cause the "large" string conversion error
              presentation: 'card',
              animationDuration: 300,
              // @ts-ignore - Explicitly set these to undefined to prevent default values
              sheetAllowedDetents: undefined,
              sheetExpandsWhenScrolledToEdge: undefined,
              sheetLargestUndimmedDetent: undefined,
              sheetGrabberVisible: undefined,
              sheetCornerRadius: undefined,
              sheetPresentationStyle: undefined,
            }}
          >
            <Stack.Screen
              name="Home"
              component={HomeScreen}
              options={{
                title: 'Keyphrase Extractor',
                // @ts-ignore - Explicitly set these to undefined to prevent default values
                sheetAllowedDetents: undefined,
                sheetLargestUndimmedDetent: undefined,
                sheetExpandsWhenScrolledToEdge: undefined,
                sheetGrabberVisible: undefined,
                sheetCornerRadius: undefined,
                sheetPresentationStyle: undefined,
                animationDuration: 300,
              }}
            />
            <Stack.Screen
              name="Results"
              component={ResultsScreen}
              options={{
                title: 'Results',
                // @ts-ignore - Explicitly set these to undefined to prevent default values
                sheetAllowedDetents: undefined,
                sheetLargestUndimmedDetent: undefined,
                sheetExpandsWhenScrolledToEdge: undefined,
                sheetGrabberVisible: undefined,
                sheetCornerRadius: undefined,
                sheetPresentationStyle: undefined,
                animationDuration: 300,
              }}
            />
            <Stack.Screen
              name="History"
              component={HistoryScreen}
              options={{
                title: 'History',
                // @ts-ignore - Explicitly set these to undefined to prevent default values
                sheetAllowedDetents: undefined,
                sheetLargestUndimmedDetent: undefined,
                sheetExpandsWhenScrolledToEdge: undefined,
                sheetGrabberVisible: undefined,
                sheetCornerRadius: undefined,
                sheetPresentationStyle: undefined,
                animationDuration: 300,
              }}
            />
            <Stack.Screen
              name="Settings"
              component={SettingsScreen}
              options={{
                title: 'Settings',
                // @ts-ignore - Explicitly set these to undefined to prevent default values
                sheetAllowedDetents: undefined,
                sheetLargestUndimmedDetent: undefined,
                sheetExpandsWhenScrolledToEdge: undefined,
                sheetGrabberVisible: undefined,
                sheetCornerRadius: undefined,
                sheetPresentationStyle: undefined,
                animationDuration: 300,
              }}
            />
          </Stack.Navigator>
          <StatusBar style="light" backgroundColor="transparent" translucent />
        </NavigationContainer>
      </PaperProvider>
    </SafeAreaProvider>
  );
}
