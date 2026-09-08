/**
 * Width of each side panel of the duel screen. ZoneViewer insets its backdrop
 * by this on both sides so the panels stay visible while it is open.
 *
 * The panels do not shrink, so the board takes whatever is left and scrolls
 * horizontally once that is less than it needs. Every pixel here is a pixel
 * off the board, and what it buys is width for the card names and action
 * descriptions, which sit on one line and ellipse.
 */
export const PANEL_WIDTH = 320;
