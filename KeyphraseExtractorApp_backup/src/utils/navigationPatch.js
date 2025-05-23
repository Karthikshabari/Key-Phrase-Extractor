/**
 * This file contains patches for react-navigation to fix the "large" string conversion error
 * It should be imported at the top of App.tsx
 * Enhanced for compatibility with react-native-screens ~4.4.0
 */

// Define a comprehensive list of problematic properties
const PROBLEMATIC_PROPS = [
  'sheetAllowedDetents',
  'sheetLargestUndimmedDetent',
  'sheetExpandsWhenScrolledToEdge',
  'sheetGrabberVisible',
  'sheetCornerRadius',
  'sheetPresentationStyle',
  'stackPresentation',
  'stackAnimation',
  'detents',
  'largestUndimmedDetent',
  'cornerRadius',
  'allowedDetents',
  'modalPresentationStyle',
  'modalTransitionStyle'
];

// Define valid string values that should not be removed
const VALID_STRING_VALUES = [
  'card', 'modal', 'transparentModal', 'containedModal',
  'containedTransparentModal', 'formSheet', 'fullScreenModal',
  'default', 'fade', 'flip', 'none', 'slide_from_right', 'slide_from_left',
  'slide_from_bottom', 'slide_from_top'
];

// Helper function to sanitize props
function sanitizeProps(props) {
  if (!props) return props;

  const safeProps = { ...props };

  // Process all props to prevent string to float conversion errors
  Object.keys(safeProps).forEach(key => {
    // Handle specific problematic props
    if (PROBLEMATIC_PROPS.includes(key)) {
      if (safeProps[key] === 'large' ||
          (typeof safeProps[key] === 'string' &&
           !VALID_STRING_VALUES.includes(safeProps[key]))) {
        delete safeProps[key];
      }
    }

    // Handle any prop with 'sheet' in the name
    if (key.includes('sheet') && typeof safeProps[key] === 'string' &&
        !VALID_STRING_VALUES.includes(safeProps[key])) {
      delete safeProps[key];
    }

    // Ensure numeric properties are actually numbers
    if ((key === 'animationDuration' || key.includes('Duration')) &&
        typeof safeProps[key] === 'string') {
      safeProps[key] = parseInt(safeProps[key], 10) || 300;
    }
  });

  return safeProps;
}

// Patch a component with the sanitization logic
function patchComponent(componentName) {
  if (!global.RNScreens || !global.RNScreens[componentName]) return;

  const originalComponent = global.RNScreens[componentName];

  global.RNScreens[componentName] = (props) => {
    return originalComponent(sanitizeProps(props));
  };

  console.log(`[NavigationPatch] Patched ${componentName}`);
}

// List of components to patch
const COMPONENTS_TO_PATCH = [
  'Screen',
  'ScreenStack',
  'ScreenStackHeaderConfig',
  'ScreenContainer',
  'FullWindowOverlay',
  'NativeScreen',
  'NativeScreenContainer',
  'NativeScreenNavigationContainer'
];

// Apply patches to all known components
if (global.RNScreens) {
  COMPONENTS_TO_PATCH.forEach(componentName => {
    patchComponent(componentName);
  });

  // Also patch any other components that exist
  Object.keys(global.RNScreens).forEach(key => {
    if (typeof global.RNScreens[key] === 'function' &&
        !COMPONENTS_TO_PATCH.includes(key) &&
        key !== 'default') {
      patchComponent(key);
    }
  });

  console.log('[NavigationPatch] Applied enhanced patches to prevent "large" string conversion errors');
}

export default {};
