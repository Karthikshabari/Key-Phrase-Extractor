// This file provides a web implementation for MaterialIcons
import React from 'react';
import { Text } from 'react-native';

// Icon mapping for common icons
const ICON_MAP = {
  'home': '\ue88a',
  'settings': '\ue8b8',
  'history': '\ue889',
  'search': '\ue8b6',
  'add': '\ue145',
  'edit': '\ue3c9',
  'delete': '\ue872',
  'menu': '\ue5d2',
  'close': '\ue5cd',
  'check': '\ue5ca',
  'arrow_back': '\ue5c4',
  'arrow_forward': '\ue5c8',
  'arrow_upward': '\ue5d8',
  'arrow_downward': '\ue5db',
  'refresh': '\ue5d5',
  'more_vert': '\ue5d4',
  'more_horiz': '\ue5d3',
  'star': '\ue838',
  'star_border': '\ue83a',
  'favorite': '\ue87d',
  'favorite_border': '\ue87e',
  'info': '\ue88e',
  'warning': '\ue002',
  'error': '\ue000',
  'help': '\ue887',
  'person': '\ue7fd',
  'people': '\ue7ef',
  'account_circle': '\ue853',
  'mail': '\ue0e1',
  'phone': '\ue0cd',
  'message': '\ue0c9',
  'notifications': '\ue7f4',
  'visibility': '\ue8f4',
  'visibility_off': '\ue8f5',
  'lock': '\ue897',
  'lock_open': '\ue898',
  'share': '\ue80d',
  'cloud_upload': '\ue2c3',
  'cloud_download': '\ue2c0',
  'file_download': '\ue884',
  'file_upload': '\ue2c6',
  'attach_file': '\ue226',
  'link': '\ue157',
  'access_time': '\ue192',
  'calendar_today': '\ue935',
  'event': '\ue878',
  'location_on': '\ue55f',
  'place': '\ue55f',
  'directions': '\ue52e',
  'navigation': '\ue55d',
  'call': '\ue0b0',
  'email': '\ue0be',
  'send': '\ue163',
  'bookmark': '\ue866',
  'bookmark_border': '\ue867',
  'camera': '\ue3af',
  'photo': '\ue410',
  'image': '\ue3f4',
  'videocam': '\ue04b',
  'music_note': '\ue405',
  'playlist_play': '\ue05f',
  'mic': '\ue029',
  'volume_up': '\ue050',
  'volume_down': '\ue04d',
  'volume_mute': '\ue04e',
  'volume_off': '\ue04f',
};

// Create a component that renders the icon as a font
const MaterialIcon = ({ name, color, size, ...rest }) => {
  // Get the icon character from the map or use a default
  const iconChar = ICON_MAP[name] || '\ue5cd'; // default icon
  
  return (
    <Text
      style={{
        fontFamily: 'MaterialIcons',
        fontSize: size,
        color,
        ...rest.style,
      }}
      {...rest}
    >
      {iconChar}
    </Text>
  );
};

// Export the component
MaterialIcon.displayName = 'MaterialIcon';
export default MaterialIcon;
