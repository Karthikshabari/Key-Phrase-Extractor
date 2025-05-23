import React, { useEffect, useRef } from 'react';
import { Animated, StyleSheet, Pressable, ViewStyle } from 'react-native';
import { Surface, useTheme } from 'react-native-paper';
import { lightImpact } from '../utils/haptics';

interface AnimatedCardProps {
  children: React.ReactNode;
  onPress?: () => void;
  style?: ViewStyle;
  contentStyle?: ViewStyle;
  elevation?: number;
  delay?: number;
  animateOnMount?: boolean;
}

const AnimatedCard: React.FC<AnimatedCardProps> = ({
  children,
  onPress,
  style,
  contentStyle,
  elevation = 2,
  delay = 0,
  animateOnMount = true,
}) => {
  const theme = useTheme();
  const scaleAnim = useRef(new Animated.Value(animateOnMount ? 0.95 : 1)).current;
  const opacityAnim = useRef(new Animated.Value(animateOnMount ? 0 : 1)).current;
  const translateYAnim = useRef(new Animated.Value(animateOnMount ? 20 : 0)).current;
  
  // Animate in when component mounts
  useEffect(() => {
    if (!animateOnMount) return;
    
    Animated.parallel([
      Animated.timing(opacityAnim, {
        toValue: 1,
        duration: 500,
        delay,
        useNativeDriver: true,
      }),
      Animated.timing(translateYAnim, {
        toValue: 0,
        duration: 500,
        delay,
        useNativeDriver: true,
      }),
      Animated.spring(scaleAnim, {
        toValue: 1,
        delay,
        useNativeDriver: true,
        tension: 100,
        friction: 8,
      }),
    ]).start();
  }, []);
  
  // Handle press animation
  const handlePressIn = () => {
    if (!onPress) return;
    
    Animated.spring(scaleAnim, {
      toValue: 0.98,
      useNativeDriver: true,
      tension: 100,
      friction: 5,
    }).start();
  };
  
  const handlePressOut = () => {
    if (!onPress) return;
    
    Animated.spring(scaleAnim, {
      toValue: 1,
      useNativeDriver: true,
      tension: 100,
      friction: 5,
    }).start();
  };
  
  const handlePress = () => {
    if (!onPress) return;
    
    lightImpact();
    onPress();
  };
  
  return (
    <Animated.View
      style={[
        styles.container,
        {
          opacity: opacityAnim,
          transform: [
            { scale: scaleAnim },
            { translateY: translateYAnim },
          ],
        },
        style,
      ]}
    >
      <Surface style={[styles.surface, { elevation }]}>
        {onPress ? (
          <Pressable
            onPressIn={handlePressIn}
            onPressOut={handlePressOut}
            onPress={handlePress}
            style={[styles.pressable, contentStyle]}
            android_ripple={{ color: theme.colors.rippleColor }}
          >
            {children}
          </Pressable>
        ) : (
          <Animated.View style={[styles.content, contentStyle]}>
            {children}
          </Animated.View>
        )}
      </Surface>
    </Animated.View>
  );
};

const styles = StyleSheet.create({
  container: {
    marginVertical: 8,
  },
  surface: {
    borderRadius: 16,
    overflow: 'hidden',
  },
  pressable: {
    width: '100%',
    padding: 16,
  },
  content: {
    width: '100%',
    padding: 16,
  },
});

export default AnimatedCard;
