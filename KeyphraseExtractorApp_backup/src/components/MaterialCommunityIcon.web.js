// This file provides a web implementation for MaterialCommunityIcons
import React from 'react';
import { Text } from 'react-native';

// Icon mapping for common icons
const ICON_MAP = {
  'home': '\uf2dc',
  'history': '\uf2da',
  'cog': '\uf493',
  'text-box-plus': '\uf137c',
  'format-list-bulleted': '\uf279',
  'star': '\uf4ce',
  'star-half-full': '\uf4d0',
  'thumb-up': '\uf513',
  'thumb-up-outline': '\uf514',
  'thumbs-up-down': '\uf515',
  'chart-bar': '\uf128',
  'numeric-1-circle': '\uf3a9',
  'numeric-2-circle': '\uf3b5',
  'numeric-3-circle': '\uf3be',
  'tag-multiple': '\uf4ff',
  'tag-outline': '\uf4fa',
  'key-variant': '\uf30a',
  'key-outline': '\uf309',
  'key-plus': '\uf30b',
  'key-star': '\uf30c',
  'key-chain-variant': '\uf30d',
  'label-outline': '\uf315',
  'text-box-search-outline': '\uf138a',
  'ruler': '\uf45f',
  'file-document-outline': '\uf21e',
  'text-search': '\uf13b7',
  'clock-outline': '\uf150',
  'newspaper-variant-outline': '\uf9fe',
  'domain': '\uf1d7',
  'key-search': '\uf30e',
  'text-box-multiple-outline': '\uf1380',
  'counter': '\uf19d',
  'information-outline': '\uf2fd',
  'text-box-search': '\uf138b',
  'sort-alphabetical-ascending': '\uf4ba',
  'sort-alphabetical-descending': '\uf4bb',
  'delete': '\uf1c0',
  'share': '\uf496',
  'bookmark-outline': '\uf0c2',
  'arrow-left': '\uf04d',
  'settings': '\uf493',
  'plus': '\uf415',
  'check': '\uf12c',
  'close': '\uf156',
  'alert': '\uf026',
  'information': '\uf2fc',
  'help-circle': '\uf2d7',
  'magnify': '\uf349',
  'file-document': '\uf21e',
  'file-export': '\uf21f',
  'download': '\uf1da',
  'upload': '\uf552',
  'dots-vertical': '\uf1d9',
  'dots-horizontal': '\uf1d8',
  'menu': '\uf35c',
  'chevron-down': '\uf140',
  'chevron-up': '\uf143',
  'chevron-left': '\uf141',
  'chevron-right': '\uf142',
  'arrow-right': '\uf04e',
  'arrow-up': '\uf050',
  'arrow-down': '\uf04f',
  'refresh': '\uf450',
  'filter': '\uf232',
  'sort': '\uf4c1',
  'sort-ascending': '\uf4c2',
  'sort-descending': '\uf4c3',
  'calendar': '\uf0ed',
  'clock': '\uf150',
  'account': '\uf004',
  'account-circle': '\uf009',
  'email': '\uf1ee',
  'phone': '\uf3f2',
  'web': '\uf59f',
  'link': '\uf337',
  'heart': '\uf2d1',
  'heart-outline': '\uf2d3',
  'bookmark': '\uf0c0',
  'bookmark-outline': '\uf0c2',
  'eye': '\uf208',
  'eye-off': '\uf209',
  'lock': '\uf33e',
  'lock-open': '\uf33f',
  'key': '\uf306',
  'content-save': '\uf18f',
  'content-copy': '\uf18f',
  'content-paste': '\uf18e',
  'content-cut': '\uf190',
  'trash-can': '\uf53b',
  'trash-can-outline': '\uf53c',
  'pencil': '\uf3eb',
  'pencil-outline': '\uf3ec',
  'plus-circle': '\uf417',
  'minus-circle': '\uf376',
  'plus-circle-outline': '\uf418',
  'minus-circle-outline': '\uf377',
  'alert-circle': '\uf028',
  'alert-circle-outline': '\uf027',
  'information-outline': '\uf2fd',
  'help-circle-outline': '\uf2d8',
  'check-circle': '\uf12d',
  'check-circle-outline': '\uf12e',
  'close-circle': '\uf159',
  'close-circle-outline': '\uf15a',
  'star-outline': '\uf4d2',
  'star-half': '\uf4d1',
  'tag': '\uf4f9',
  'tag-outline': '\uf4fa',
};

// Create a component that renders the icon as a font
const MaterialCommunityIcon = ({ name, color, size, ...rest }) => {
  // Get the icon character from the map or use a default
  const iconChar = ICON_MAP[name] || '\uf15b'; // default icon

  return (
    <Text
      style={{
        fontFamily: 'MaterialCommunityIcons',
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
MaterialCommunityIcon.displayName = 'MaterialCommunityIcon';
export default MaterialCommunityIcon;
