/**
 * Utility functions for handling icons safely
 */

// List of known valid material-community icons
const VALID_MATERIAL_COMMUNITY_ICONS = [
  'pencil',
  'text-search',
  'format-list-bulleted',
  'key-variant',
  'key-outline',
  'label-outline',
  'sort-alphabetical-ascending',
  'sort-alphabetical-descending',
  'key-star',
  'text-box-search',
  'counter',
  'alert-circle-outline',
  'history',
  'lightbulb-on-outline',
  'ruler',
  'file-document-outline',
  'domain',
  'information-outline',
  'chevron-up',
  'chevron-down',
  'tag-text'
];

// Fallback icons to use if the requested icon is not available
const FALLBACK_ICONS = {
  'text-box-edit-outline': 'pencil',
  'text-box-edit': 'pencil'
};

/**
 * Get a safe icon name that is guaranteed to exist
 * @param {string} iconName - The requested icon name
 * @returns {string} - A valid icon name
 */
export const getSafeIconName = (iconName) => {
  // If the icon is valid, return it
  if (VALID_MATERIAL_COMMUNITY_ICONS.includes(iconName)) {
    return iconName;
  }
  
  // If we have a fallback for this icon, use it
  if (FALLBACK_ICONS[iconName]) {
    return FALLBACK_ICONS[iconName];
  }
  
  // Default fallback
  return 'pencil';
};
