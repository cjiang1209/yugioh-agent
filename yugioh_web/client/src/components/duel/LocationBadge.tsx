import { Hand, Layers, Swords, ScrollText, Skull, Ghost, Star } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import {
  LOCATION_BANISHED,
  LOCATION_BASE_MASK,
  LOCATION_DECK,
  LOCATION_EXTRA,
  LOCATION_GRAVE,
  LOCATION_HAND,
  LOCATION_MZONE,
  LOCATION_SZONE,
} from "../../../../shared/locations";

interface BadgeSpec {
  Icon: LucideIcon;
  label: string;
  color: string;
}

function badgeSpec(location: number | undefined): BadgeSpec | null {
  if (!location) return null;
  const base = location & LOCATION_BASE_MASK;
  switch (base) {
    case LOCATION_HAND:     return { Icon: Hand,       label: "Hand",        color: "#00f5ff" };
    case LOCATION_DECK:     return { Icon: Layers,     label: "Deck",        color: "#00f5ff" };
    case LOCATION_MZONE:    return { Icon: Swords,     label: "Monster Zone",color: "#00f5ff" };
    case LOCATION_SZONE:    return { Icon: ScrollText, label: "Spell/Trap",  color: "#00f5ff" };
    case LOCATION_GRAVE:    return { Icon: Skull,      label: "Graveyard",   color: "#ff2d78" };
    case LOCATION_BANISHED: return { Icon: Ghost,      label: "Banished",    color: "#b44fff" };
    case LOCATION_EXTRA:    return { Icon: Star,       label: "Extra Deck",  color: "#ffd700" };
    default:                return null;
  }
}

interface LocationBadgeProps {
  location: number | undefined;
  size?: number;
}

export function LocationBadge({ location, size = 16 }: LocationBadgeProps) {
  const spec = badgeSpec(location);
  if (!spec) return null;
  const iconPx = Math.round(size * 0.625);
  return (
    <div
      title={spec.label}
      style={{
        position: "absolute",
        bottom: "2px",
        right: "2px",
        width: `${size}px`,
        height: `${size}px`,
        borderRadius: "3px",
        background: "rgba(0,0,0,0.78)",
        border: `1px solid ${spec.color}`,
        color: spec.color,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        boxShadow: "0 0 4px rgba(0,0,0,0.6)",
        pointerEvents: "none",
      }}
    >
      <spec.Icon size={iconPx} strokeWidth={2.25} />
    </div>
  );
}
