import React, { useEffect, useRef } from 'react';
import { Animated, StyleSheet, Pressable, View, ViewStyle } from 'react-native';
import { Surface, useTheme } from 'react-native-paper';
import { lightImpact } from '../utils/haptics';

interface AnimatedListItemProps {
  children: React.ReactNode;
  onPress?: () => void;
  style?: ViewStyle;
  index?: number;
  elevation?: number;
}

const AnimatedListItem: React.FC<AnimatedListItemProps> = ({
  children,
  onPress,
  style,
  index = 0,
  elevation = 1,
}) => {
  const theme = useTheme();
  const scaleAnim = useRef(new Animated.Value(0.95)).current;
  const opacityAnim = useRef(new Animated.Value(0)).current;
  const translateYAnim = useRef(new Animated.Value(20)).current;
  
  // Animate in when component mounts
  useEffect(() => {
    const delay = index * 100; // Stagger animation based on index
    
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
      toValue: 0.97,
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
            style={styles.pressable}
            android_ripple={{ color: theme.colors.rippleColor }}
          >
            {children}
          </Pressable>
        ) : (
          <View style={styles.content}>{children}</View>
        )}
      </Surface>
    </Animated.View>
  );
};

const styles = StyleSheet.create({
  container: {
    marginVertical: 6,
    marginHorizontal: 12,
  },
  surface: {
    borderRadius: 16,
    overflow: 'hidden',
  },
  pressable: {
    width: '100%',
  },
  content: {
    width: '100%',
  },
});

export default AnimatedListItem;
