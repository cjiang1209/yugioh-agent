// ygopro-core location constants — mirrors yugioh_core/constants.py.
// OVERLAY (0x80) is a flag, not a base location; mask with `loc & LOCATION_BASE_MASK`
// to extract the underlying zone.
export const LOCATION_DECK = 0x01;
export const LOCATION_HAND = 0x02;
export const LOCATION_MZONE = 0x04;
export const LOCATION_SZONE = 0x08;
export const LOCATION_GRAVE = 0x10;
export const LOCATION_BANISHED = 0x20;
export const LOCATION_EXTRA = 0x40;
export const LOCATION_OVERLAY = 0x80;

export const LOCATION_BASE_MASK = 0x7f;
