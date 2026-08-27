from v1.tui.logo import logo_width, render_block_logo, render_title


def test_logo_fits_and_draws_block_letters():
    logo = render_block_logo(80)

    assert logo is not None
    plain = logo.plain
    assert plain.count("\n") == 5
    assert "█" in plain
    assert "▀" in plain
    assert logo_width() == 47


def test_logo_falls_back_when_the_terminal_is_narrow():
    assert render_block_logo(20) is None
    assert render_title(20).plain == "LAZYDOCS"
