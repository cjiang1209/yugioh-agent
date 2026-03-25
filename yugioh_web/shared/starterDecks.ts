// ─── Pre-built Starter Decks ─────────────────────────────────────────────────
// Using real YGOPRODeck card IDs for authentic card data

export interface DeckDefinition {
  name: string;
  description: string;
  cardIds: number[];
}

// Yugi's classic deck (Dark Magician focused)
export const YUGI_DECK: DeckDefinition = {
  name: "Yugi's Deck",
  description: "The King of Games' signature deck featuring the Dark Magician and powerful spells.",
  cardIds: [
    // Dark Magician x3
    46986414, 46986414, 46986414,
    // Dark Magician Girl x2
    38033121, 38033121,
    // Buster Blader x2
    78193831, 78193831,
    // Summoned Skull x2
    70781052, 70781052,
    // Kuriboh x3
    40640057, 40640057, 40640057,
    // Celtic Guardian x2
    91152256, 91152256,
    // Giant Soldier of Stone x2
    13039848, 13039848,
    // Mystical Elf x2
    15025844, 15025844,
    // Dark Magician of Chaos x1
    40737112,
    // Magician of Faith x2
    31560081, 31560081,
    // Spells
    // Dark Magic Attack x2
    2314238, 2314238,
    // Swords of Revealing Light x2
    72302403, 72302403,
    // Monster Reborn x1
    83764718,
    // Pot of Greed x1
    55144522,
    // Change of Heart x1
    4031928,
    // Dark Hole x1
    53129443,
    // Mystical Space Typhoon x2
    5318639, 5318639,
    // Traps
    // Mirror Force x2
    44095762, 44095762,
    // Magic Cylinder x2
    62279055, 62279055,
    // Spellbinding Circle x1
    18807108,
    // Trap Hole x2
    4206964, 4206964,
    // Seven Tools of the Bandit x1
    3819470,
  ],
};

// Blue-Eyes competitive deck (provided passcodes)
export const BLUE_EYES_DECK: DeckDefinition = {
  name: "Blue-Eyes Deck",
  description: "The ultimate Blue-Eyes White Dragon competitive deck.",
  cardIds: [
    // Main Deck (40 cards)
    89631139, 89631139, 89631139, // Blue-Eyes White Dragon x3
    38517737, 38517737, 38517737, // The White Stone of Legend x3
    64202399,                     // Blue-Eyes Alternative White Dragon x1
    57043986, 57043986,           // Sage with Eyes of Blue x2
    45467446, 45467446,           // Maiden with Eyes of Blue x2
    71039903, 71039903, 71039903, // Dragon Spirit of White x3
    79814787, 79814787,           // Blue-Eyes Solid Dragon x2
    8240199,  8240199,  8240199,  // Effect Veiler x3
    88241506, 88241506, 88241506, // Ash Blossom & Joyous Spring x3
    48800175, 48800175, 48800175, // Trade-In x3
    38120068, 38120068,           // Cards of Consonance x2
    6853254,  6853254,  6853254,  // Silver's Cry x3
    41620959, 41620959,           // The Melody of Awakening Dragon x2
    24094653, 24094653,           // Polymerization x2
    83764718,                     // Monster Reborn x1
    2295440,                      // Return of the Dragon Lords x1
    5318639,  5318639,            // Mystical Space Typhoon x2
    24224830, 24224830,           // Burst Stream of Destruction x2
    // Extra Deck (11 cards)
    23995346,                     // Blue-Eyes Ultimate Dragon x1
    56532353, 56532353,           // Azure-Eyes Silver Dragon x2
    2129638,  2129638,            // Blue-Eyes Twin Burst Dragon x2
    43228023,                     // Blue-Eyes Spirit Dragon x1
    59822133, 59822133,           // Neo Blue-Eyes Ultimate Dragon x2
    40908371, 40908371,           // Blue-Eyes Alternative Ultimate Dragon x2
    89604813,                     // Borreload Savage Dragon x1
  ],
};

// Kaiba's classic deck (Blue-Eyes White Dragon focused)
export const KAIBA_DECK: DeckDefinition = BLUE_EYES_DECK;

export const STARTER_DECKS = [YUGI_DECK, KAIBA_DECK];
