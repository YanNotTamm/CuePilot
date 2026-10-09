"""Genre profiles and structural cueing rules.

Each profile encodes the typical section order for the genre plus the
configurable default cue slot map (Section 8). `section_order` is used as a
prior by the structure detector and the cue generator. `cue_strategy` maps
each physical slot (1-8) to a semantic CueRole, following the genre-aware
recommended cue tables from Genre_pattern.md sections 19-42.

Following Genre_pattern.md section 71, the MVP set covers: EDM, House,
Progressive House, Hip-Hop, R&B, Jersey Club, Baile Funk, Indobounce, Hipdut,
Open Format -- plus the additional profiles CuePilot already shipped.
"""

from __future__ import annotations

from ..models import CueRole, GenreProfile, SectionType

# 8-slot standard (Genre_pattern.md section 7):
#   A=START, B=INTRO, C=VOCAL/HOOK, D=BUILD, E=DROP, F=BREAKDOWN,
#   G=SECOND_DROP, H=OUTRO.
DEFAULT_SLOT_MAP: dict[SectionType, int] = {
    SectionType.INTRO: 2,
    SectionType.VOCAL: 3,
    SectionType.BUILD: 4,
    SectionType.DROP: 5,
    SectionType.BREAKDOWN: 6,
    SectionType.SECOND_DROP: 7,
    SectionType.OUTRO: 8,
}

DEFAULT_CUE_WEIGHTS: dict[str, float] = {
    "phrase_boundary": 0.25,
    "energy_change": 0.20,
    "spectral_change": 0.15,
    "rhythm_change": 0.15,
    "vocal_onset": 0.10,
    "beat_alignment": 0.10,
    "genre_prior": 0.05,
}

# Standard 8-slot strategy: A=START, B=INTRO, C=VOCAL, D=BUILD, E=DROP,
# F=BREAKDOWN, G=SECOND_DROP, H=OUTRO (Genre_pattern.md section 7).
STANDARD_CUE_STRATEGY: dict[int, CueRole] = {
    1: CueRole.START,
    2: CueRole.INTRO,
    3: CueRole.VOCAL,
    4: CueRole.BUILD,
    5: CueRole.DROP,
    6: CueRole.BREAKDOWN,
    7: CueRole.SECOND_DROP,
    8: CueRole.OUTRO,
}

DEFAULT_DETECTION_WEIGHTS: dict[str, float] = {
    "energy": 0.30,
    "vocal": 0.20,
    "rhythm": 0.20,
    "bass": 0.15,
    "phrase": 0.15,
}

# Indonesian descriptions for the genre picker (frontend switches by language).
DESCRIPTION_ID: dict[str, str] = {
    "afrobeats": "Perkusi, vokal, chorus, groove Afrika Barat.",
    "amapiano": "Log drum, perkusi, melodi piano. 108-115 BPM.",
    "baile": "Perkusi, vokal, groove utama. Edit gaya BLB / NYXX.",
    "bass_house": "Bobot bass berat, kick tegas, melodi minimal.",
    "bassline_bounce": "Bass berat, groove memantul. Fokus ke energi frekuensi rendah & kick-bass.",
    "big_room": "Panggung utama festival, drop masif, breakdown minimal.",
    "breakbeat": "Tidak selalu four-on-the-floor. Fokus onset & periodisitas ritmis.",
    "dancehall": "Riddim, vokal, hook, break.",
    "drum_and_bass": "Drum breakbeat, bass mengalir, nuansa half-time. 160-180 BPM.",
    "dubstep": "140 BPM nuansa half-time, bass wobble, drop.",
    "edm": "Big room, progressive, festival. Struktur build/drop.",
    "electro_house": "Saw agresif, bass berat, build besar.",
    "future_house": "Bass memantul, vokal pop, drop berbasis groove.",
    "hardstyle": "Kick keras, drop klimaks, klimaks kedua.",
    "hip_hop": "Hook bukan drop. Struktur verse/hook, tanpa build/drop EDM.",
    "hipdut": "Hibrida hip-hop + dangdut + elektronik. Persepsi double-time.",
    "house": "House umum. Intro drum, groove, vokal, breakdown.",
    "indobounce": "Bass keras / bounce Indonesia. Drop kuat, ritme repetitif.",
    "jersey_club": "Pola kick, stutter, vocal chop, seksi pendek bertenaga.",
    "jungle": "Breakbeat, bass berat, nuansa ragga.",
    "open_format": "Campuran banyak genre. Pemetaan fleksibel, utamakan vokal/hook/energi.",
    "pop": "Pop mainstream. Verse/chorus, breakdown melodis.",
    "progressive_house": "Transisi panjang, layering bertahap, breakdown panjang. Frase 16-32 bar.",
    "reggaeton": "Irama dembow, vokal, hook, perubahan ritme.",
    "rnb": "Fokus vokal, chorus, instrumental. Hindari cue berlebihan.",
    "speed_garage": "2-step cepat, bass shuffle, energi garage.",
    "tech_house": "Kick, bass, perkusi dan hook vokal. Tanpa breakdown melodis berat.",
    "techno": "Tanpa drop konvensional; memakai energy peak dan perubahan groove.",
    "trance": "Melodis, breakdown panjang, drop euphoric.",
    "trap": "Bass 808, hi-hat roll, nuansa half-time 65-75 BPM.",
    "uk_garage": "Irama 2-step, shuffle, bass, vokal.",
}


def _profile(
    name: str,
    display_name: str,
    description: str,
    bpm_range: tuple[float, float],
    section_order: list[SectionType],
    cue_strategy: dict[int, CueRole] | None = None,
    detection_weights: dict[str, float] | None = None,
    energy_profile: str = "build_drop",
    transition_style: str = "phrase",
    preferred_phrase_bars: list[int] | None = None,
    slot_map: dict[SectionType, int] | None = None,
    description_id: str = "",
) -> GenreProfile:
    return GenreProfile(
        name=name,
        display_name=display_name,
        description=description,
        description_id=description_id or DESCRIPTION_ID.get(name, ""),
        bpm_range=bpm_range,
        section_order=section_order,
        slot_map=slot_map or dict(DEFAULT_SLOT_MAP),
        cue_weights=dict(DEFAULT_CUE_WEIGHTS),
        cue_strategy=cue_strategy or dict(STANDARD_CUE_STRATEGY),
        detection_weights=detection_weights or dict(DEFAULT_DETECTION_WEIGHTS),
        preferred_phrase_bars=preferred_phrase_bars or [8, 16, 32],
        energy_profile=energy_profile,
        transition_style=transition_style,
    )


PROFILES: dict[str, GenreProfile] = {}


def _register(profile: GenreProfile) -> None:
    PROFILES[profile.name] = profile


_register(
    _profile(
        "edm",
        "EDM",
        "Big room, progressive, festival. Build/drop structure.",
        (120.0, 132.0),
        [
            SectionType.INTRO,
            SectionType.VOCAL,
            SectionType.BUILD,
            SectionType.DROP,
            SectionType.BREAKDOWN,
            SectionType.SECOND_DROP,
            SectionType.OUTRO,
        ],
        cue_strategy=dict(STANDARD_CUE_STRATEGY),
        detection_weights={"energy": 0.35, "vocal": 0.15, "rhythm": 0.25, "bass": 0.10, "phrase": 0.15},
        energy_profile="build_drop",
    )
)

_register(
    _profile(
        "house",
        "House",
        "Generic house. Drum intro, groove, vocal, breakdown.",
        (115.0, 132.0),
        [
            SectionType.INTRO,
            SectionType.BUILD,
            SectionType.DROP,
            SectionType.BREAKDOWN,
            SectionType.SECOND_DROP,
            SectionType.OUTRO,
        ],
        cue_strategy=dict(STANDARD_CUE_STRATEGY),
        detection_weights={"energy": 0.30, "vocal": 0.20, "rhythm": 0.20, "bass": 0.15, "phrase": 0.15},
        energy_profile="build_drop",
    )
)

_register(
    _profile(
        "tech_house",
        "Tech House",
        "Kick, bass, percussion and vocal hooks. No heavy melodic breakdowns.",
        (124.0, 130.0),
        [
            SectionType.INTRO,
            SectionType.BUILD,
            SectionType.DROP,
            SectionType.BREAKDOWN,
            SectionType.SECOND_DROP,
            SectionType.OUTRO,
        ],
        cue_strategy=dict(STANDARD_CUE_STRATEGY),
        detection_weights={"energy": 0.30, "vocal": 0.20, "rhythm": 0.25, "bass": 0.15, "phrase": 0.10},
        energy_profile="build_drop",
    )
)

_register(
    _profile(
        "techno",
        "Techno",
        "No conventional drop; uses energy peaks and groove changes.",
        (125.0, 150.0),
        [
            SectionType.INTRO,
            SectionType.BUILD,
            SectionType.DROP,
            SectionType.BREAKDOWN,
            SectionType.SECOND_DROP,
            SectionType.OUTRO,
        ],
        cue_strategy={
            1: CueRole.START,
            2: CueRole.INTRO,
            3: CueRole.ENERGY_PEAK,
            4: CueRole.TRANSITION,
            5: CueRole.ENERGY_PEAK,
            6: CueRole.BREAKDOWN,
            7: CueRole.SECOND_DROP,
            8: CueRole.OUTRO,
        },
        detection_weights={"energy": 0.40, "vocal": 0.05, "rhythm": 0.30, "bass": 0.10, "phrase": 0.15},
        energy_profile="energy_peak",
    )
)

_register(
    _profile(
        "hip_hop",
        "Hip-Hop",
        "Hook ≠ drop. Verse/hook structure, no EDM build/drop.",
        (70.0, 100.0),
        [
            SectionType.INTRO,
            SectionType.VOCAL,
            SectionType.CHORUS,
            SectionType.VOCAL,
            SectionType.CHORUS,
            SectionType.OUTRO,
        ],
        cue_strategy={
            1: CueRole.START,
            2: CueRole.INTRO,
            3: CueRole.VOCAL,
            4: CueRole.HOOK,
            5: CueRole.HOOK,
            6: CueRole.BREAKDOWN,
            7: CueRole.SECOND_DROP,
            8: CueRole.OUTRO,
        },
        detection_weights={"energy": 0.15, "vocal": 0.45, "rhythm": 0.15, "bass": 0.15, "phrase": 0.10},
        energy_profile="hook_based",
    )
)

_register(
    _profile(
        "baile",
        "Baile Funk",
        "Percussion, vocal, main groove. BLB / NYXX style edits.",
        (130.0, 150.0),
        [
            SectionType.INTRO,
            SectionType.VOCAL,
            SectionType.DROP,
            SectionType.VOCAL,
            SectionType.SECOND_DROP,
            SectionType.OUTRO,
        ],
        cue_strategy={
            1: CueRole.START,
            2: CueRole.INTRO,
            3: CueRole.VOCAL,
            4: CueRole.PRE_DROP,
            5: CueRole.DROP,
            6: CueRole.BREAKDOWN,
            7: CueRole.SECOND_DROP,
            8: CueRole.OUTRO,
        },
        detection_weights={"energy": 0.30, "vocal": 0.30, "rhythm": 0.20, "bass": 0.10, "phrase": 0.10},
        energy_profile="groove_based",
    )
)

_register(
    _profile(
        "indobounce",
        "Indobounce",
        "Hard bass / bounce Indonesia. Strong drop, repetitive rhythm.",
        (125.0, 140.0),
        [
            SectionType.INTRO,
            SectionType.BUILD,
            SectionType.DROP,
            SectionType.BREAKDOWN,
            SectionType.SECOND_DROP,
            SectionType.OUTRO,
        ],
        cue_strategy=dict(STANDARD_CUE_STRATEGY),
        detection_weights={"energy": 0.35, "vocal": 0.20, "rhythm": 0.25, "bass": 0.10, "phrase": 0.10},
        energy_profile="build_drop",
    )
)

_register(
    _profile(
        "progressive_house",
        "Progressive House",
        "Long transitions, gradual layering, long breakdowns. 16-32 bar phrases.",
        (122.0, 130.0),
        [
            SectionType.INTRO,
            SectionType.BUILD,
            SectionType.DROP,
            SectionType.BREAKDOWN,
            SectionType.BUILD,
            SectionType.SECOND_DROP,
            SectionType.OUTRO,
        ],
        cue_strategy={
            1: CueRole.START,
            2: CueRole.INTRO,
            3: CueRole.MELODY,
            4: CueRole.BREAKDOWN,
            5: CueRole.BUILD,
            6: CueRole.DROP,
            7: CueRole.SECOND_DROP,
            8: CueRole.OUTRO,
        },
        detection_weights={"energy": 0.25, "vocal": 0.10, "rhythm": 0.15, "bass": 0.15, "phrase": 0.35},
        energy_profile="melodic_breakdown",
        preferred_phrase_bars=[16, 32],
    )
)
_register(
    _profile(
        "rnb",
        "R&B",
        "Vocal, chorus, instrumental focus. Avoid excessive cueing.",
        (65.0, 115.0),
        [
            SectionType.INTRO,
            SectionType.VOCAL,
            SectionType.CHORUS,
            SectionType.VOCAL,
            SectionType.CHORUS,
            SectionType.BREAKDOWN,
            SectionType.OUTRO,
        ],
        cue_strategy={
            1: CueRole.START,
            2: CueRole.MIX_IN,
            3: CueRole.VOCAL,
            4: CueRole.CHORUS,
            5: CueRole.HOOK,
            6: CueRole.BREAKDOWN,
            7: CueRole.SECOND_DROP,
            8: CueRole.MIX_OUT,
        },
        detection_weights={"energy": 0.10, "vocal": 0.50, "rhythm": 0.10, "bass": 0.10, "phrase": 0.20},
        energy_profile="vocal_based",
    )
)

_register(
    _profile(
        "jersey_club",
        "Jersey Club",
        "Kick patterns, stutter, vocal chops, short energetic sections.",
        (130.0, 140.0),
        [
            SectionType.INTRO,
            SectionType.VOCAL,
            SectionType.VOCAL,
            SectionType.BUILD,
            SectionType.DROP,
            SectionType.BREAKDOWN,
            SectionType.SECOND_DROP,
            SectionType.OUTRO,
        ],
        cue_strategy={
            1: CueRole.START,
            2: CueRole.INTRO,
            3: CueRole.VOCAL,
            4: CueRole.PRE_DROP,
            5: CueRole.DROP,
            6: CueRole.BREAKDOWN,
            7: CueRole.SECOND_DROP,
            8: CueRole.OUTRO,
        },
        detection_weights={"energy": 0.35, "vocal": 0.25, "rhythm": 0.30, "bass": 0.05, "phrase": 0.05},
        energy_profile="energy_peak",
    )
)

_register(
    _profile(
        "hipdut",
        "Hipdut",
        "Hybrid hip-hop + dangdut + electronic. Double-time perception.",
        (80.0, 110.0),
        [
            SectionType.INTRO,
            SectionType.VOCAL,
            SectionType.CHORUS,
            SectionType.VOCAL,
            SectionType.CHORUS,
            SectionType.OUTRO,
        ],
        cue_strategy={
            1: CueRole.START,
            2: CueRole.INTRO,
            3: CueRole.VOCAL,
            4: CueRole.HOOK,
            5: CueRole.DROP,
            6: CueRole.BREAKDOWN,
            7: CueRole.SECOND_DROP,
            8: CueRole.OUTRO,
        },
        detection_weights={"energy": 0.25, "vocal": 0.35, "rhythm": 0.20, "bass": 0.10, "phrase": 0.10},
        energy_profile="vocal_based",
    )
)

_register(
    _profile(
        "pop",
        "Pop",
        "Mainstream pop. Verse/chorus, melodic breakdown.",
        (90.0, 130.0),
        [
            SectionType.INTRO,
            SectionType.VOCAL,
            SectionType.CHORUS,
            SectionType.VOCAL,
            SectionType.CHORUS,
            SectionType.BREAKDOWN,
            SectionType.CHORUS,
            SectionType.OUTRO,
        ],
        cue_strategy={
            1: CueRole.START,
            2: CueRole.INTRO,
            3: CueRole.VOCAL,
            4: CueRole.HOOK,
            5: CueRole.HOOK,
            6: CueRole.BREAKDOWN,
            7: CueRole.SECOND_DROP,
            8: CueRole.OUTRO,
        },
        detection_weights={"energy": 0.20, "vocal": 0.40, "rhythm": 0.10, "bass": 0.10, "phrase": 0.20},
        energy_profile="vocal_based",
    )
)

_register(
    _profile(
        "bassline_bounce",
        "Bassline Bounce",
        "Heavy bass, bounce groove. Weight to low-frequency energy & kick-bass.",
        (130.0, 140.0),
        [
            SectionType.INTRO,
            SectionType.BUILD,
            SectionType.DROP,
            SectionType.BREAKDOWN,
            SectionType.SECOND_DROP,
            SectionType.OUTRO,
        ],
        cue_strategy={
            1: CueRole.START,
            2: CueRole.INTRO,
            3: CueRole.VOCAL,
            4: CueRole.BUILD,
            5: CueRole.DROP,
            6: CueRole.BREAKDOWN,
            7: CueRole.SECOND_DROP,
            8: CueRole.OUTRO,
        },
        detection_weights={"energy": 0.25, "vocal": 0.15, "rhythm": 0.20, "bass": 0.30, "phrase": 0.10},
        energy_profile="bass_bounce",
    )
)

_register(
    _profile(
        "breakbeat",
        "Breakbeat",
        "Not always four-on-the-floor. Onset & rhythmic periodicity focused.",
        (125.0, 145.0),
        [
            SectionType.INTRO,
            SectionType.VOCAL,
            SectionType.DROP,
            SectionType.BREAKDOWN,
            SectionType.SECOND_DROP,
            SectionType.OUTRO,
        ],
        cue_strategy={
            1: CueRole.START,
            2: CueRole.INTRO,
            3: CueRole.VOCAL,
            4: CueRole.TRANSITION,
            5: CueRole.DROP,
            6: CueRole.BREAKDOWN,
            7: CueRole.SECOND_DROP,
            8: CueRole.OUTRO,
        },
        detection_weights={"energy": 0.30, "vocal": 0.15, "rhythm": 0.35, "bass": 0.10, "phrase": 0.10},
        energy_profile="breakbeat",
    )
)

_register(
    _profile(
        "future_house",
        "Future House",
        "Bouncy bass, pop vocals, groove-driven drops.",
        (124.0, 128.0),
        [
            SectionType.INTRO,
            SectionType.BUILD,
            SectionType.DROP,
            SectionType.BREAKDOWN,
            SectionType.SECOND_DROP,
            SectionType.OUTRO,
        ],
        cue_strategy=dict(STANDARD_CUE_STRATEGY),
        detection_weights={"energy": 0.30, "vocal": 0.20, "rhythm": 0.20, "bass": 0.20, "phrase": 0.10},
        energy_profile="build_drop",
    )
)

_register(
    _profile(
        "bass_house",
        "Bass House",
        "Heavy bass weight, punchy kick, minimal melody.",
        (124.0, 132.0),
        [
            SectionType.INTRO,
            SectionType.BUILD,
            SectionType.DROP,
            SectionType.BREAKDOWN,
            SectionType.SECOND_DROP,
            SectionType.OUTRO,
        ],
        cue_strategy=dict(STANDARD_CUE_STRATEGY),
        detection_weights={"energy": 0.25, "vocal": 0.10, "rhythm": 0.25, "bass": 0.30, "phrase": 0.10},
        energy_profile="bass_bounce",
    )
)

_register(
    _profile(
        "electro_house",
        "Electro House",
        "Aggressive saws, heavy bass, big builds.",
        (126.0, 132.0),
        [
            SectionType.INTRO,
            SectionType.BUILD,
            SectionType.DROP,
            SectionType.BREAKDOWN,
            SectionType.SECOND_DROP,
            SectionType.OUTRO,
        ],
        cue_strategy=dict(STANDARD_CUE_STRATEGY),
        detection_weights={"energy": 0.35, "vocal": 0.10, "rhythm": 0.20, "bass": 0.20, "phrase": 0.15},
        energy_profile="build_drop",
    )
)

_register(
    _profile(
        "big_room",
        "Big Room",
        "Festival mainstage, massive drops, minimal breakdowns.",
        (126.0, 132.0),
        [
            SectionType.INTRO,
            SectionType.BUILD,
            SectionType.DROP,
            SectionType.BREAKDOWN,
            SectionType.SECOND_DROP,
            SectionType.OUTRO,
        ],
        cue_strategy=dict(STANDARD_CUE_STRATEGY),
        detection_weights={"energy": 0.40, "vocal": 0.05, "rhythm": 0.20, "bass": 0.15, "phrase": 0.20},
        energy_profile="build_drop",
    )
)

_register(
    _profile(
        "trance",
        "Trance",
        "Melodic, long breakdowns, euphoric drops.",
        (128.0, 150.0),
        [
            SectionType.INTRO,
            SectionType.BUILD,
            SectionType.DROP,
            SectionType.BREAKDOWN,
            SectionType.SECOND_DROP,
            SectionType.OUTRO,
        ],
        cue_strategy={
            1: CueRole.START,
            2: CueRole.INTRO,
            3: CueRole.MELODY,
            4: CueRole.BREAKDOWN,
            5: CueRole.BUILD,
            6: CueRole.DROP,
            7: CueRole.SECOND_DROP,
            8: CueRole.OUTRO,
        },
        detection_weights={"energy": 0.30, "vocal": 0.15, "rhythm": 0.15, "bass": 0.10, "phrase": 0.30},
        energy_profile="melodic_breakdown",
        preferred_phrase_bars=[16, 32],
    )
)

_register(
    _profile(
        "drum_and_bass",
        "Drum & Bass",
        "Breakbeat drums, rolling bass, half-time feel. 160-180 BPM.",
        (160.0, 180.0),
        [
            SectionType.INTRO,
            SectionType.BREAKDOWN,
            SectionType.DROP,
            SectionType.BREAKDOWN,
            SectionType.SECOND_DROP,
            SectionType.OUTRO,
        ],
        cue_strategy={
            1: CueRole.START,
            2: CueRole.INTRO,
            3: CueRole.VOCAL,
            4: CueRole.BUILD,
            5: CueRole.DROP,
            6: CueRole.BREAKDOWN,
            7: CueRole.SECOND_DROP,
            8: CueRole.OUTRO,
        },
        detection_weights={"energy": 0.25, "vocal": 0.10, "rhythm": 0.40, "bass": 0.10, "phrase": 0.15},
        energy_profile="breakbeat",
    )
)

_register(
    _profile(
        "dubstep",
        "Dubstep",
        "140 BPM half-time feel, wobble bass, drops.",
        (138.0, 142.0),
        [
            SectionType.INTRO,
            SectionType.BUILD,
            SectionType.DROP,
            SectionType.BREAKDOWN,
            SectionType.SECOND_DROP,
            SectionType.OUTRO,
        ],
        cue_strategy={
            1: CueRole.START,
            2: CueRole.INTRO,
            3: CueRole.VOCAL,
            4: CueRole.PRE_DROP,
            5: CueRole.DROP,
            6: CueRole.BREAKDOWN,
            7: CueRole.SECOND_DROP,
            8: CueRole.OUTRO,
        },
        detection_weights={"energy": 0.30, "vocal": 0.05, "rhythm": 0.20, "bass": 0.30, "phrase": 0.15},
        energy_profile="bass_bounce",
    )
)

_register(
    _profile(
        "trap",
        "Trap",
        "808 bass, hi-hat rolls, half-time 65-75 BPM feel.",
        (130.0, 160.0),
        [
            SectionType.INTRO,
            SectionType.VOCAL,
            SectionType.BUILD,
            SectionType.DROP,
            SectionType.BREAKDOWN,
            SectionType.SECOND_DROP,
            SectionType.OUTRO,
        ],
        cue_strategy={
            1: CueRole.START,
            2: CueRole.INTRO,
            3: CueRole.VOCAL,
            4: CueRole.BUILD,
            5: CueRole.DROP,
            6: CueRole.BREAKDOWN,
            7: CueRole.SECOND_DROP,
            8: CueRole.OUTRO,
        },
        detection_weights={"energy": 0.20, "vocal": 0.20, "rhythm": 0.25, "bass": 0.25, "phrase": 0.10},
        energy_profile="808_based",
    )
)

_register(
    _profile(
        "amapiano",
        "Amapiano",
        "Log drums, percussion, piano melodies. 110-115 BPM.",
        (108.0, 115.0),
        [
            SectionType.INTRO,
            SectionType.VOCAL,
            SectionType.CHORUS,
            SectionType.VOCAL,
            SectionType.CHORUS,
            SectionType.OUTRO,
        ],
        cue_strategy={
            1: CueRole.START,
            2: CueRole.GROOVE,
            3: CueRole.VOCAL,
            4: CueRole.PRE_GROOVE,
            5: CueRole.GROOVE,
            6: CueRole.BREAKDOWN,
            7: CueRole.SECOND_DROP,
            8: CueRole.OUTRO,
        },
        detection_weights={"energy": 0.20, "vocal": 0.20, "rhythm": 0.30, "bass": 0.15, "phrase": 0.15},
        energy_profile="groove_based",
    )
)

_register(
    _profile(
        "afrobeats",
        "Afrobeats",
        "Percussion, vocal, chorus, groove.",
        (100.0, 115.0),
        [
            SectionType.INTRO,
            SectionType.VOCAL,
            SectionType.CHORUS,
            SectionType.VOCAL,
            SectionType.CHORUS,
            SectionType.OUTRO,
        ],
        cue_strategy={
            1: CueRole.START,
            2: CueRole.GROOVE,
            3: CueRole.VOCAL,
            4: CueRole.PRE_GROOVE,
            5: CueRole.HOOK,
            6: CueRole.BREAKDOWN,
            7: CueRole.SECOND_DROP,
            8: CueRole.OUTRO,
        },
        detection_weights={"energy": 0.20, "vocal": 0.30, "rhythm": 0.25, "bass": 0.10, "phrase": 0.15},
        energy_profile="groove_based",
    )
)

_register(
    _profile(
        "dancehall",
        "Dancehall",
        "Riddim, vocal, hook, break.",
        (90.0, 108.0),
        [
            SectionType.INTRO,
            SectionType.VOCAL,
            SectionType.CHORUS,
            SectionType.VOCAL,
            SectionType.CHORUS,
            SectionType.OUTRO,
        ],
        cue_strategy={
            1: CueRole.START,
            2: CueRole.GROOVE,
            3: CueRole.VOCAL,
            4: CueRole.PRE_GROOVE,
            5: CueRole.HOOK,
            6: CueRole.BREAKDOWN,
            7: CueRole.SECOND_DROP,
            8: CueRole.OUTRO,
        },
        detection_weights={"energy": 0.20, "vocal": 0.30, "rhythm": 0.25, "bass": 0.10, "phrase": 0.15},
        energy_profile="groove_based",
    )
)

_register(
    _profile(
        "reggaeton",
        "Reggaeton",
        "Dembow rhythm, vocal, hook, rhythm changes.",
        (85.0, 100.0),
        [
            SectionType.INTRO,
            SectionType.VOCAL,
            SectionType.CHORUS,
            SectionType.VOCAL,
            SectionType.CHORUS,
            SectionType.OUTRO,
        ],
        cue_strategy={
            1: CueRole.START,
            2: CueRole.GROOVE,
            3: CueRole.VOCAL,
            4: CueRole.PRE_GROOVE,
            5: CueRole.HOOK,
            6: CueRole.BREAKDOWN,
            7: CueRole.SECOND_DROP,
            8: CueRole.OUTRO,
        },
        detection_weights={"energy": 0.20, "vocal": 0.30, "rhythm": 0.25, "bass": 0.10, "phrase": 0.15},
        energy_profile="dembow",
    )
)

_register(
    _profile(
        "uk_garage",
        "UK Garage",
        "2-step rhythm, shuffle, bass, vocal.",
        (130.0, 140.0),
        [
            SectionType.INTRO,
            SectionType.VOCAL,
            SectionType.CHORUS,
            SectionType.VOCAL,
            SectionType.CHORUS,
            SectionType.OUTRO,
        ],
        cue_strategy={
            1: CueRole.START,
            2: CueRole.INTRO,
            3: CueRole.VOCAL,
            4: CueRole.TRANSITION,
            5: CueRole.HOOK,
            6: CueRole.BREAKDOWN,
            7: CueRole.SECOND_DROP,
            8: CueRole.OUTRO,
        },
        detection_weights={"energy": 0.20, "vocal": 0.20, "rhythm": 0.30, "bass": 0.15, "phrase": 0.15},
        energy_profile="2step",
    )
)

_register(
    _profile(
        "speed_garage",
        "Speed Garage",
        "Fast 2-step, shuffled bass, garage energy.",
        (130.0, 140.0),
        [
            SectionType.INTRO,
            SectionType.VOCAL,
            SectionType.CHORUS,
            SectionType.VOCAL,
            SectionType.CHORUS,
            SectionType.OUTRO,
        ],
        cue_strategy={
            1: CueRole.START,
            2: CueRole.INTRO,
            3: CueRole.VOCAL,
            4: CueRole.TRANSITION,
            5: CueRole.HOOK,
            6: CueRole.BREAKDOWN,
            7: CueRole.SECOND_DROP,
            8: CueRole.OUTRO,
        },
        detection_weights={"energy": 0.25, "vocal": 0.20, "rhythm": 0.30, "bass": 0.10, "phrase": 0.15},
        energy_profile="2step",
    )
)

_register(
    _profile(
        "hardstyle",
        "Hardstyle",
        "Hard kicks, climax drops, second climax.",
        (150.0, 160.0),
        [
            SectionType.INTRO,
            SectionType.BUILD,
            SectionType.DROP,
            SectionType.BREAKDOWN,
            SectionType.SECOND_DROP,
            SectionType.OUTRO,
        ],
        cue_strategy={
            1: CueRole.START,
            2: CueRole.INTRO,
            3: CueRole.VOCAL,
            4: CueRole.BUILD,
            5: CueRole.CLIMAX,
            6: CueRole.BREAKDOWN,
            7: CueRole.CLIMAX,
            8: CueRole.OUTRO,
        },
        detection_weights={"energy": 0.35, "vocal": 0.10, "rhythm": 0.25, "bass": 0.20, "phrase": 0.10},
        energy_profile="climax",
    )
)

_register(
    _profile(
        "jungle",
        "Jungle",
        "Breakbeats, heavy bass, ragga vibes.",
        (160.0, 180.0),
        [
            SectionType.INTRO,
            SectionType.BREAKDOWN,
            SectionType.DROP,
            SectionType.BREAKDOWN,
            SectionType.SECOND_DROP,
            SectionType.OUTRO,
        ],
        cue_strategy={
            1: CueRole.START,
            2: CueRole.INTRO,
            3: CueRole.VOCAL,
            4: CueRole.TRANSITION,
            5: CueRole.DROP,
            6: CueRole.BREAKDOWN,
            7: CueRole.SECOND_DROP,
            8: CueRole.OUTRO,
        },
        detection_weights={"energy": 0.25, "vocal": 0.15, "rhythm": 0.35, "bass": 0.15, "phrase": 0.10},
        energy_profile="breakbeat",
    )
)

_register(
    _profile(
        "open_format",
        "Open Format",
        "Mixed bag of genres. Flexible mapping, prioritize vocal/hook/energy.",
        (80.0, 150.0),
        [
            SectionType.INTRO,
            SectionType.VOCAL,
            SectionType.BUILD,
            SectionType.DROP,
            SectionType.BREAKDOWN,
            SectionType.SECOND_DROP,
            SectionType.OUTRO,
        ],
        cue_strategy={
            1: CueRole.START,
            2: CueRole.MIX_IN,
            3: CueRole.VOCAL,
            4: CueRole.TRANSITION,
            5: CueRole.HOOK,
            6: CueRole.BREAKDOWN,
            7: CueRole.SECOND_DROP,
            8: CueRole.MIX_OUT,
        },
        detection_weights={"energy": 0.25, "vocal": 0.30, "rhythm": 0.15, "bass": 0.10, "phrase": 0.20},
        energy_profile="flexible",
    )
)


DEFAULT_PROFILE = "open_format"

# Aliases so DJ-friendly shorthand resolves to the right profile.
# Note: keys below are AFTER get_profile normalization (lowercase,
# "-" and " " -> "_", "&" -> "and").
_ALIASES: dict[str, str] = {
    "blb": "bassline_bounce",
    "bassline": "bassline_bounce",
    "dnb": "drum_and_bass",
    "prog_house": "progressive_house",
    "progressive": "progressive_house",
    "hiphop": "hip_hop",
    "r_and_b": "rnb",
    "randb": "rnb",
    "openformat": "open_format",
    "jersey": "jersey_club",
    "ukg": "uk_garage",
}


def get_profile(name: str) -> GenreProfile:
    """Return a built-in genre profile, or the default if unknown."""
    key = name.strip().lower().replace("-", "_").replace(" ", "_").replace("&", "and")
    key = _ALIASES.get(key, key)
    return PROFILES.get(key, PROFILES[DEFAULT_PROFILE])


def list_profiles() -> list[str]:
    return sorted(PROFILES)
