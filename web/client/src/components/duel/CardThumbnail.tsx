import { LocationBadge } from "./LocationBadge";

const CARD_IMAGE_BASE = "https://images.ygoprodeck.com/images/cards_small";

interface CardThumbnailProps {
  cardCode: number;
  width: number;
  height: number;
  borderColor: string;
  /** Fallback content when cardCode is 0. If omitted, hides on missing image. */
  fallback?: React.ReactNode;
  /** Engine location for the LocationBadge overlay. Omit to skip badge. */
  location?: number;
  /** LocationBadge size in px. */
  badgeSize?: number;
  alt?: string;
  boxShadow?: string;
  borderRadius?: number;
}

export function CardThumbnail({
  cardCode,
  width,
  height,
  borderColor,
  fallback,
  location,
  badgeSize = 16,
  alt = "",
  boxShadow,
  borderRadius = 2,
}: CardThumbnailProps) {
  const hasImage = cardCode > 0;

  if (!hasImage) {
    return (
      <div
        style={{
          width: `${width}px`,
          height: `${height}px`,
          flexShrink: 0,
          borderRadius: `${borderRadius}px`,
          background: "rgba(255,255,255,0.04)",
          border: `1px solid ${borderColor}`,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        {fallback}
      </div>
    );
  }

  return (
    <div
      style={{
        position: "relative",
        width: `${width}px`,
        height: `${height}px`,
        flexShrink: 0,
      }}
    >
      <img
        src={`${CARD_IMAGE_BASE}/${cardCode}.jpg`}
        alt={alt}
        style={{
          width: `${width}px`,
          height: `${height}px`,
          objectFit: "cover",
          borderRadius: `${borderRadius}px`,
          border: `1px solid ${borderColor}`,
          boxShadow,
        }}
        onError={e => {
          (e.target as HTMLImageElement).style.display = "none";
        }}
      />
      {hasImage && <LocationBadge location={location} size={badgeSize} />}
    </div>
  );
}
