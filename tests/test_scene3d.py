from robotnav.scene3d import write_viewer
from robotnav.terrain import solvable_map


def _page(tmp_path, **kw):
    t = solvable_map("grassland", seed=3, size=48)
    return write_viewer([t], tmp_path / "docs" / "index.html", scenes=[{"key": "grassland"}], **kw).read_text()


def test_viewer_is_a_full_standards_mode_document(tmp_path):
    page = _page(tmp_path)
    assert page.startswith("<!doctype html>\n")
    assert '<meta charset="utf-8">' in page[:200]               # the page has symbols like × and ⇄
    assert '"key":"grassland"' in page                          # the scene data went in
    assert "og:image" not in page                               # no link-preview tags unless it has a home


def test_hosted_viewer_has_link_previews(tmp_path):
    page = _page(tmp_path, site_url="https://example.github.io/demo/")
    head = page[:2500]
    assert '<meta property="og:image" content="https://example.github.io/demo/social.png">' in head
    assert '<meta name="twitter:card" content="summary_large_image">' in head
    assert '<link rel="canonical" href="https://example.github.io/demo/">' in head


def test_viewer_has_the_console_look(tmp_path):
    page = _page(tmp_path)
    assert "family=Inter" in page and "JetBrains+Mono" in page and "Orbitron" not in page   # words and numbers
    assert 'id="hud-compass-cv"' in page and 'id="hud-alt-cv"' in page               # the drone view's read-outs
    assert "M 155,150 C 155,50 115,30 100,30 85,30 45,50 45,150" in page             # MIL-STD-2525 friendly air frame
    assert "M 100,28 L 172,100 100,172 28,100 Z" in page                              # and the hostile ground one
    assert "UAV-01" in page and "TWR-" in page and "TRACKING" in page                # callsign tags


def test_viewer_has_the_operator_console(tmp_path):
    page = _page(tmp_path)
    for part in ("ops-ents", "ops-tasks", "ops-tl", "ops-log", "ops-pal", "ops-aar"):   # lists, timeline, log, command bar, report
        assert f'id="{part}"' in page
    assert '{ key: "balanced", name: "Balanced", w: 2 }' in page                # the three plans' weights
    for status in ("SENT", "RUNNING", "DONE_OK", "DONE_NOT_OK"):                 # a task's life
        assert f'"{status}"' in page
    assert "re-planned: roadblock at" in page and "getUTCHours" in page          # the log's wording and its UTC clock
    assert 'w === "ops"' in page                                                 # the #ops link


def test_intro_art_is_inlined(tmp_path):
    page = _page(tmp_path)
    assert "__FLAG__" not in page and "__BISON_URI__" not in page   # both placeholders were filled
    assert page.count('<svg class="intro-flag') == 2                 # flown flat, and hung for portrait screens
    assert 'src="data:image/webp;base64,' in page                  # the bison travels inside the page
