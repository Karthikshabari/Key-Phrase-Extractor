import React from 'react';
import { StyleSheet, Animated, View, Platform, Pressable } from 'react-native';

// Conditionally import gesture handler components
let RectButton: any = View;
let Swipeable: any = ({ children }: { children: React.ReactNode }) => <View>{children}</View>;

// Only import on native platforms
if (Platform.OS !== 'web') {
  try {
    const GestureHandler = require('react-native-gesture-handler');
    RectButton = GestureHandler.RectButton;
    Swipeable = GestureHandler.Swipeable;
  } catch (error) {
    console.warn('react-native-gesture-handler not available');
  }
}
import { IconButton, useTheme } from 'react-native-paper';
import { HistoryItem } from '../utils/storage';
import { lightImpact } from '../utils/haptics';

interface SwipeableHistoryItemProps {
  item: HistoryItem;
  onDelete: (id: string) => void;
  onShare: (item: HistoryItem) => void;
  children: React.ReactNode;
}

const SwipeableHistoryItem: React.FC<SwipeableHistoryItemProps> = ({
  item,
  onDelete,
  onShare,
  children,
}) => {
  const theme = useTheme();
  let swipeableRef = React.useRef<Swipeable>(null);

  const renderRightActions = (
    progress: Animated.AnimatedInterpolation<number>,
    dragX: Animated.AnimatedInterpolation<number>
  ) => {
    const trans = dragX.interpolate({
      inputRange: [-80, 0],
      outputRange: [0, 80],
      extrapolate: 'clamp',
    });

    const opacity = dragX.interpolate({
      inputRange: [-80, -60, 0],
      outputRange: [1, 0.8, 0],
      extrapolate: 'clamp',
    });

    return (
      <Animated.View
        style={[
          styles.rightAction,
          {
            transform: [{ translateX: trans }],
            opacity,
          },
        ]}
      >
        <RectButton
          style={[styles.actionButton, { backgroundColor: theme.colors.error }]}
          onPress={() => {
            lightImpact();
            swipeableRef.current?.close();
            onDelete(item.id);
          }}
        >
          <IconButton icon="delete" size={24} iconColor="#fff" />
        </RectButton>
      </Animated.View>
    );
  };

  const renderLeftActions = (
    progress: Animated.AnimatedInterpolation<number>,
    dragX: Animated.AnimatedInterpolation<number>
  ) => {
    const trans = dragX.interpolate({
      inputRange: [0, 80],
      outputRange: [-80, 0],
      extrapolate: 'clamp',
    });

    const opacity = dragX.interpolate({
      inputRange: [0, 60, 80],
      outputRange: [0, 0.8, 1],
      extrapolate: 'clamp',
    });

    return (
      <Animated.View
        style={[
          styles.leftAction,
          {
            transform: [{ translateX: trans }],
            opacity,
          },
        ]}
      >
        <RectButton
          style={[styles.actionButton, { backgroundColor: theme.colors.primary }]}
          onPress={() => {
            lightImpact();
            swipeableRef.current?.close();
            onShare(item);
          }}
        >
          <IconButton icon="share" size={24} iconColor="#fff" />
        </RectButton>
      </Animated.View>
    );
  };

  // On web, just render the children without swipe functionality
  if (Platform.OS === 'web') {
    return <View style={styles.container}>{children}</View>;
  }

  // On native platforms, use Swipeable
  return (
    <Swipeable
      ref={swipeableRef}
      friction={2}
      leftThreshold={80}
      rightThreshold={80}
      renderRightActions={renderRightActions}
      renderLeftActions={renderLeftActions}
      onSwipeableOpen={() => lightImpact()}
    >
      {children}
    </Swipeable>
  );
};

const styles = StyleSheet.create({
  rightAction: {
    alignItems: 'center',
    justifyContent: 'center',
    width: 80,
  },
  leftAction: {
    alignItems: 'center',
    justifyContent: 'center',
    width: 80,
  },
  actionButton: {
    width: 60,
    height: 60,
    borderRadius: 30,
    alignItems: 'center',
    justifyContent: 'center',
    elevation: 2,
    shadowColor: 'rgba(0,0,0,0.3)',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.2,
    shadowRadius: 1.5,
  },
});

export default SwipeableHistoryItem;
