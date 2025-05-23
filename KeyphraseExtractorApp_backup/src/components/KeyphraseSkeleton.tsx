import React, { useEffect } from 'react';
import { View, StyleSheet, Animated } from 'react-native';
import { Surface, useTheme } from 'react-native-paper';

interface KeyphraseSkeletonProps {
  count?: number;
}

const KeyphraseSkeleton: React.FC<KeyphraseSkeletonProps> = ({ count = 5 }) => {
  const theme = useTheme();
  const shimmerAnimation = new Animated.Value(0);
  
  useEffect(() => {
    const startShimmerAnimation = () => {
      Animated.loop(
        Animated.timing(shimmerAnimation, {
          toValue: 1,
          duration: 1500,
          useNativeDriver: false,
        })
      ).start();
    };
    
    startShimmerAnimation();
    
    return () => {
      shimmerAnimation.stopAnimation();
    };
  }, []);
  
  const getAnimatedStyle = (delay: number) => {
    const translateX = shimmerAnimation.interpolate({
      inputRange: [0, 1],
      outputRange: [-300, 300],
    });
    
    return {
      transform: [{ translateX }],
      opacity: shimmerAnimation.interpolate({
        inputRange: [0, 0.2, 0.8, 1],
        outputRange: [0.3, 0.6, 0.6, 0.3],
      }),
      backgroundColor: theme.colors.primary,
    };
  };
  
  const renderSkeletonItem = (index: number) => {
    return (
      <Surface key={index} style={styles.container} elevation={1}>
        <View style={styles.contentContainer}>
          <View style={styles.keyphraseContainer}>
            <View style={[styles.chip, { backgroundColor: theme.colors.surfaceVariant }]}>
              <Animated.View style={[styles.shimmer, getAnimatedStyle(index * 100)]} />
            </View>
          </View>
          
          <View style={styles.scoreContainer}>
            <View style={[styles.scoreText, { backgroundColor: theme.colors.surfaceVariant }]}>
              <Animated.View style={[styles.shimmer, getAnimatedStyle(index * 100 + 50)]} />
            </View>
            <View style={[styles.scoreBarContainer, { backgroundColor: theme.colors.surfaceVariant }]}>
              <Animated.View style={[styles.shimmer, getAnimatedStyle(index * 100 + 100)]} />
            </View>
          </View>
        </View>
      </Surface>
    );
  };
  
  return (
    <View style={styles.skeletonContainer}>
      {Array.from({ length: count }).map((_, index) => renderSkeletonItem(index))}
    </View>
  );
};

const styles = StyleSheet.create({
  skeletonContainer: {
    padding: 8,
  },
  container: {
    margin: 8,
    borderRadius: 16,
    overflow: 'hidden',
    position: 'relative',
  },
  contentContainer: {
    padding: 16,
  },
  keyphraseContainer: {
    marginBottom: 12,
  },
  chip: {
    alignSelf: 'flex-start',
    borderRadius: 8,
    height: 40,
    width: '70%',
    position: 'relative',
    overflow: 'hidden',
  },
  scoreContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  scoreText: {
    width: 50,
    height: 24,
    marginRight: 12,
    borderRadius: 4,
    position: 'relative',
    overflow: 'hidden',
  },
  scoreBarContainer: {
    flex: 1,
    height: 10,
    borderRadius: 6,
    position: 'relative',
    overflow: 'hidden',
  },
  shimmer: {
    width: '30%',
    height: '100%',
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
  },
});

export default KeyphraseSkeleton;
