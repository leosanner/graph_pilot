"""LazyDocs wordmark: block letters with a lime → cyan wash."""

from rich.style import Style
from rich.text import Text

LIME = "#A3FF60"
GREEN = "#4ADE80"
TEAL = "#5EEAD4"
CYAN = "#22D3EE"
SELECT = "#0F766E"
MUTED = "#5B6B73"
RED = "#FF6B6B"
SHADOW = "#1B2B22"
DARK = "#0F172A"

LOGO_WORD = "LAZYDOCS"
LOGO_HEIGHT = 5
LOGO_FONT: dict[str, tuple[str, ...]] = {
    "L": ("█    ", "█    ", "█    ", "█    ", "█████"),
    "A": (" ███ ", "█   █", "█████", "█   █", "█   █"),
    "Z": ("█████", "   █ ", "  █  ", " █   ", "█████"),
    "Y": ("█   █", " █ █ ", "  █  ", "  █  ", "  █  "),
    "D": ("████ ", "█   █", "█   █", "█   █", "████ "),
    "O": (" ███ ", "█   █", "█   █", "█   █", " ███ "),
    "C": (" ████", "█    ", "█    ", "█    ", " ████"),
    "S": (" ████", "█    ", " ███ ", "    █", "████ "),
}

_GRADIENT_STOPS = (
    (0xA3, 0xFF, 0x60),
    (0x4A, 0xDE, 0x80),
    (0x5E, 0xEA, 0xD4),
    (0x22, 0xD3, 0xEE),
)


def logo_width() -> int:
    letter_count = len(LOGO_WORD)
    return letter_count * 5 + (letter_count - 1)


def _lerp(a: int, b: int, t: float) -> int:
    return int(a + (b - a) * t + 0.5)


def _rgb_hex(color: tuple[int, int, int]) -> str:
    return f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}"


def _gradient_color(frac: float) -> str:
    if frac <= 0:
        return _rgb_hex(_GRADIENT_STOPS[0])
    if frac >= 1:
        return _rgb_hex(_GRADIENT_STOPS[-1])

    scaled = frac * (len(_GRADIENT_STOPS) - 1)
    index = int(scaled)
    t = scaled - index
    start = _GRADIENT_STOPS[index]
    end = _GRADIENT_STOPS[index + 1]
    return _rgb_hex(
        (
            _lerp(start[0], end[0], t),
            _lerp(start[1], end[1], t),
            _lerp(start[2], end[2], t),
        )
    )


def render_block_logo(max_width: int = 0) -> Text | None:
    """Return the 6-line wordmark, or None when it cannot fit."""
    total = logo_width()
    if max_width > 0 and total > max_width:
        return None

    rows: list[list[str]] = [[] for _ in range(LOGO_HEIGHT)]
    last_index = len(LOGO_WORD) - 1
    for index, letter in enumerate(LOGO_WORD):
        glyph = LOGO_FONT[letter]
        for row, slice_ in enumerate(glyph):
            rows[row].extend(slice_)
            if index < last_index:
                rows[row].append(" ")

    filled_cols = [
        any(rows[row][col] == "█" for row in range(LOGO_HEIGHT))
        for col in range(len(rows[0]))
    ]
    denom = max(total - 1, 1)
    logo = Text()
    for row in rows:
        for col, char in enumerate(row):
            if char == "█":
                logo.append("█", Style(color=_gradient_color(col / denom)))
            else:
                logo.append(" ")
        logo.append("\n")

    shadow_style = Style(color=SHADOW)
    for filled in filled_cols:
        logo.append("▀" if filled else " ", shadow_style)
    return logo


def render_title(max_width: int) -> Text:
    logo = render_block_logo(max_width)
    if logo is not None:
        return logo
    return Text("LAZYDOCS", style=Style(color=LIME, bold=True))
