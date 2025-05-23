import React, { useEffect, useRef } from 'react';
import { View, StyleSheet, Animated, Easing } from 'react-native';
import { ActivityIndicator, Text, useTheme, Surface } from 'react-native-paper';
import { LinearGradient } from 'expo-linear-gradient';

interface LoadingOverlayProps {
  visible: boolean;
  message?: string;
}

/**
 * Loading overlay component to display during API calls
 */
const LoadingOverlay: React.FC<LoadingOverlayProps> = ({
  visible,
  message = 'Processing...'
}) => {
  const theme = useTheme();

  // Animation values
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const scaleAnim = useRef(new Animated.Value(0.9)).current;
  const spinAnim = useRef(new Animated.Value(0)).current;

  // Spin animation for extra flair
  const spin = spinAnim.interpolate({
    inputRange: [0, 1],
    outputRange: ['0deg', '360deg']
  });

  useEffect(() => {
    if (visible) {
      // Show animations
      Animated.parallel([
        Animated.timing(fadeAnim, {
          toValue: 1,
          duration: 300,
          useNativeDriver: true,
          easing: Easing.out(Easing.cubic)
        }),
        Animated.timing(scaleAnim, {
          toValue: 1,
          duration: 300,
          useNativeDriver: true,
          easing: Easing.out(Easing.back(1.5))
        })
      ]).start();

      // Continuous spin animation
      Animated.loop(
        Animated.timing(spinAnim, {
          toValue: 1,
          duration: 2000,
          useNativeDriver: true,
          easing: Easing.linear
        })
      ).start();
    } else {
      // Hide animations
      Animated.parallel([
        Animated.timing(fadeAnim, {
          toValue: 0,
          duration: 200,
          useNativeDriver: true,
        }),
        Animated.timing(scaleAnim, {
          toValue: 0.9,
          duration: 200,
          useNativeDriver: true,
        })
      ]).start();

      // Stop spin animation
      spinAnim.stopAnimation();
    }
  }, [visible]);

  if (!visible) return null;

  return (
    <Animated.View
      style={[
        styles.container,
        { opacity: fadeAnim }
      ]}
    >
      <Animated.View
        style={{
          transform: [{ scale: scaleAnim }]
        }}
      >
        <Surface style={styles.loadingBox} elevation={5}>
          <LinearGradient
            colors={[theme.colors.primaryContainer, theme.colors.surfaceVariant]}
            style={styles.gradient}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
          />
          <Animated.View style={[styles.spinnerContainer, { transform: [{ rotate: spin }] }]}>
            <ActivityIndicator
              size="large"
              color={theme.colors.primary}
              style={styles.spinner}
            />
          </Animated.View>
          <Text
            style={[styles.message, { color: theme.colors.onSurface }]}
            variant="bodyLarge"
          >
            {message}
          </Text>
        </Surface>
      </Animated.View>
    </Animated.View>
  );
};

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 1000,
    backdropFilter: 'blur(3px)',
  },
  loadingBox: {
    padding: 24,
    borderRadius: 20,
    alignItems: 'center',
    minWidth: 220,
    minHeight: 160,
    justifyContent: 'center',
    overflow: 'hidden',
  },
  gradient: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
  },
  spinnerContainer: {
    marginBottom: 16,
    padding: 8,
  },
  spinner: {
    transform: [{ scale: 1.2 }]
  },
  message: {
    marginTop: 8,
    textAlign: 'center',
    fontWeight: '500',
  },
});

export default LoadingOverlay;
