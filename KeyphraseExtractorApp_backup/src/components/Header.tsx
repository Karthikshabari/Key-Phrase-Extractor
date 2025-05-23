import React from 'react';
import { Appbar, useTheme, Text } from 'react-native-paper';
import { getHeaderTitle } from '@react-navigation/elements';
import { NativeStackHeaderProps } from '@react-navigation/native-stack/lib/typescript/src/types';
import { StyleSheet, Platform, StatusBar, View, SafeAreaView } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';

/**
 * Custom header component for the app
 */
const Header = ({ navigation, route, options, back }: NativeStackHeaderProps) => {
  const title = getHeaderTitle(options, route.name);
  const theme = useTheme();

  return (
    <View style={styles.headerContainer}>
      <LinearGradient
        colors={['#6A5ACD', '#9370DB']}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 0 }}
        style={styles.headerGradient}
      />
      <View style={styles.headerPattern}>
        {Array.from({ length: 5 }).map((_, i) => (
          <View key={i} style={[styles.patternCircle, { left: `${20 * i}%`, top: `${15 * (i % 3)}%` }]} />
        ))}
      </View>

      <SafeAreaView style={styles.safeArea}>
        <View style={styles.headerContent}>
          {/* Left side - Back button or empty space */}
          <View style={styles.leftSection}>
            {back ? (
              <Appbar.BackAction
                onPress={navigation.goBack}
                color="#fff"
                size={26}
                style={styles.backButton}
              />
            ) : (
              <View style={styles.logoContainer}>
                <LinearGradient
                  colors={['rgba(255,255,255,0.3)', 'rgba(255,255,255,0.1)']}
                  style={styles.logoGradient}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 1 }}
                />
                <Text style={styles.logoText}>KE</Text>
              </View>
            )}
          </View>

          {/* Middle - Title */}
          <View style={styles.titleContainer}>
            <Text style={styles.title}>{title}</Text>
          </View>

          {/* Right side - Action buttons */}
          <View style={styles.rightSection}>
            {route.name === 'Home' && (
              <Appbar.Action
                icon="text-box-multiple-outline"
                onPress={() => navigation.navigate('History')}
                color="#fff"
                size={26}
                style={styles.actionButton}
                accessibilityLabel="View History"
              />
            )}
            {route.name === 'History' && (
              <Appbar.Action
                icon="home"
                onPress={() => navigation.navigate('Home')}
                color="#fff"
                size={26}
                style={styles.actionButton}
              />
            )}
            {route.name === 'Results' && (
              <>
                <Appbar.Action
                  icon="bookmark-outline"
                  onPress={() => {
                    // This will be implemented later to save results
                    console.log('Save results');
                  }}
                  color="#fff"
                  size={26}
                  style={styles.actionButton}
                />
                <Appbar.Action
                  icon="home"
                  onPress={() => navigation.navigate('Home')}
                  color="#fff"
                  size={26}
                  style={styles.actionButton}
                />
              </>
            )}
          </View>
        </View>

        {/* Decorative element */}
        <View style={styles.decorationContainer}>
          <View style={styles.decorationLine} />
          <View style={styles.decorationDot} />
          <View style={styles.decorationLine} />
        </View>
      </SafeAreaView>
    </View>
  );
};

const styles = StyleSheet.create({
  headerContainer: {
    height: Platform.OS === 'ios' ? 110 : 80,
    position: 'relative',
    elevation: 4,
    shadowOpacity: 0.3,
    shadowRadius: 4,
    shadowOffset: { width: 0, height: 2 },
    zIndex: 1000,
  },
  headerGradient: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
  },
  headerPattern: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    opacity: 0.1,
    overflow: 'hidden',
  },
  patternCircle: {
    position: 'absolute',
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: '#fff',
    opacity: 0.2,
  },
  safeArea: {
    flex: 1,
  },
  headerContent: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 8,
    height: '100%',
    paddingTop: Platform.OS === 'ios' ? 0 : StatusBar.currentHeight,
  },
  leftSection: {
    width: 60,
    alignItems: 'center',
    justifyContent: 'center',
  },
  titleContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  rightSection: {
    width: 60,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
  },
  title: {
    fontWeight: '700',
    fontSize: 20,
    letterSpacing: 0.5,
    color: '#fff',
    textAlign: 'center',
    textShadowColor: 'rgba(0, 0, 0, 0.2)',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 2,
  },
  logoContainer: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    alignItems: 'center',
    justifyContent: 'center',
    position: 'relative',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.2)',
    elevation: 2,
    shadowColor: 'rgba(0, 0, 0, 0.2)',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.3,
    shadowRadius: 2,
  },
  logoGradient: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    borderRadius: 20,
  },
  logoText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 16,
  },
  backButton: {
    marginLeft: 0,
  },
  actionButton: {
    marginHorizontal: 4,
  },
  decorationContainer: {
    position: 'absolute',
    bottom: 4,
    left: 0,
    right: 0,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
  },
  decorationDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: 'rgba(255, 255, 255, 0.8)',
    marginHorizontal: 8,
  },
  decorationLine: {
    height: 2,
    width: 24,
    backgroundColor: 'rgba(255, 255, 255, 0.4)',
  }
});

export default Header;
