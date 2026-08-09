#!/usr/bin/env python3
"""Validate rendered navigation state for representative documentation
pages."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

TOP_LEVEL_NAVIGATION = (
    "Get started",
    "Models",
    "Train",
    "Optimize",
)


@dataclass(frozen=True)
class RouteExpectation:
    title: str
    active_link: str
    expanded_branches: tuple[str, ...]


REPRESENTATIVE_ROUTES = {
    "index.html":
    RouteExpectation(
        title="VoiceHub",
        active_link="Overview",
        expanded_branches=("Get started", ),
    ),
    "getting-started/installation/index.html":
    RouteExpectation(
        title="Installation",
        active_link="Installation",
        expanded_branches=("Get started", ),
    ),
    "getting-started/quickstart/index.html":
    RouteExpectation(
        title="Quickstart",
        active_link="Quickstart",
        expanded_branches=("Get started", ),
    ),
    "guides/inference/index.html":
    RouteExpectation(
        title="Inference",
        active_link="Inference",
        expanded_branches=("Get started", ),
    ),
    "models/providers/index.html":
    RouteExpectation(
        title="Model list",
        active_link="Model list",
        expanded_branches=("Models", ),
    ),
    "models/providers/speecht5/index.html":
    RouteExpectation(
        title="SpeechT5",
        active_link="SpeechT5",
        expanded_branches=("Models", "Text to speech"),
    ),
    "guides/trainer/index.html":
    RouteExpectation(
        title="Trainer",
        active_link="Trainer overview",
        expanded_branches=("Train", ),
    ),
    "guides/optimization-overview/index.html":
    RouteExpectation(
        title="Optimization overview",
        active_link="Overview",
        expanded_branches=("Optimize", ),
    ),
    "optimizations/compile/index.html":
    RouteExpectation(
        title="Torch compile",
        active_link="Torch compile",
        expanded_branches=("Optimize", ),
    ),
    "project/adding-a-model/index.html":
    RouteExpectation(
        title="Add a model",
        active_link="Add a model",
        expanded_branches=("Models", ),
    ),
    "reference/models/index.html":
    RouteExpectation(
        title="Models",
        active_link="Models API",
        expanded_branches=("Models", ),
    ),
}


class DocumentationDOMError(RuntimeError):
    """Raised when the rendered site contradicts its navigation contract."""


class _Node:

    def __init__(self, tag: str, attrs: list[tuple[str, str | None]], parent: _Node | None = None):
        self.tag = tag
        self.attrs = {name: value if value is not None else "" for name, value in attrs}
        self.parent = parent
        self.children: list[_Node | str] = []

    @property
    def classes(self) -> set[str]:
        return set(self.attrs.get("class", "").split())

    @property
    def text(self) -> str:
        content = "".join(child.text if isinstance(child, _Node) else child for child in self.children)
        return re.sub(r"\s+", " ", content).strip().removesuffix("¶").strip()

    def descendants(self) -> list[_Node]:
        nodes: list[_Node] = []
        for child in self.children:
            if not isinstance(child, _Node):
                continue
            nodes.append(child)
            nodes.extend(child.descendants())
        return nodes

    def direct_children(self, tag: str) -> list[_Node]:
        return [child for child in self.children if isinstance(child, _Node) and child.tag == tag]


class _TreeParser(HTMLParser):

    _VOID_ELEMENTS = {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = _Node("document", [])
        self._stack = [self.root]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = _Node(tag, attrs, parent=self._stack[-1])
        self._stack[-1].children.append(node)
        if tag not in self._VOID_ELEMENTS:
            self._stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in self._VOID_ELEMENTS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self._stack) - 1, 0, -1):
            if self._stack[index].tag == tag:
                del self._stack[index:]
                return

    def handle_data(self, data: str) -> None:
        self._stack[-1].children.append(data)


def _matches(node: _Node, tag: str, class_name: str | None = None) -> bool:
    return node.tag == tag and (class_name is None or class_name in node.classes)


def _find_all(root: _Node, tag: str, class_name: str | None = None) -> list[_Node]:
    return [node for node in root.descendants() if _matches(node, tag, class_name)]


def _find_one(root: _Node, tag: str, class_name: str | None = None) -> _Node:
    matches = _find_all(root, tag, class_name)
    if len(matches) != 1:
        description = f"{tag}.{class_name}" if class_name else tag
        raise DocumentationDOMError(f"Expected one {description}, found {len(matches)}.")
    return matches[0]


def _label_for_toggle(primary_sidebar: _Node, toggle_id: str) -> _Node:
    labels = [
        node for node in _find_all(primary_sidebar, "label", "md-nav__link")
        if node.attrs.get("for") == toggle_id
    ]
    if len(labels) != 1:
        raise DocumentationDOMError(
            f"Expected one navigation label for toggle {toggle_id!r}, found {len(labels)}.")
    return labels[0]


def _validate_top_level_navigation(primary_sidebar: _Node) -> None:
    navigation = _find_one(primary_sidebar, "nav", "md-nav--primary")
    lists = [child for child in navigation.direct_children("ul") if "md-nav__list" in child.classes]
    if len(lists) != 1:
        raise DocumentationDOMError(f"Expected one top-level navigation list, found {len(lists)}.")
    labels: list[str] = []
    for item in lists[0].direct_children("li"):
        controls = [
            child for child in item.children
            if isinstance(child, _Node) and child.tag == "label" and "md-nav__link" in child.classes
        ]
        if len(controls) == 1:
            labels.append(controls[0].text)
    if tuple(labels) != TOP_LEVEL_NAVIGATION:
        raise DocumentationDOMError(
            f"Top-level navigation is {tuple(labels)!r}, expected {TOP_LEVEL_NAVIGATION!r}.")


def _validate_route(path: Path, expectation: RouteExpectation) -> None:
    parser = _TreeParser()
    parser.feed(path.read_text(encoding="utf-8"))

    html = _find_one(parser.root, "html")
    if html.attrs.get("lang") != "en":
        raise DocumentationDOMError(f"{path}: expected lang='en', found {html.attrs.get('lang')!r}.")

    article = _find_one(parser.root, "article", "md-content__inner")
    heading = _find_one(article, "h1").text
    if heading != expectation.title:
        raise DocumentationDOMError(f"{path}: title is {heading!r}, expected {expectation.title!r}.")

    primary_sidebar = _find_one(parser.root, "div", "md-sidebar--primary")
    _validate_top_level_navigation(primary_sidebar)

    active_links = [
        node for node in _find_all(primary_sidebar, "a", "md-nav__link")
        if "md-nav__link--active" in node.classes
    ]
    active_labels = tuple(node.text for node in active_links)
    if active_labels != (expectation.active_link, ):
        raise DocumentationDOMError(
            f"{path}: active links are {active_labels!r}, expected {(expectation.active_link,)!r}.")

    toggles = [
        toggle for toggle in _find_all(primary_sidebar, "input", "md-nav__toggle")
        if toggle.attrs.get("id", "").startswith("__nav_")
    ]
    checked_toggles = [toggle for toggle in toggles if "checked" in toggle.attrs]
    expanded_labels = tuple(
        _label_for_toggle(primary_sidebar, toggle.attrs.get("id", "")).text for toggle in checked_toggles)
    if expanded_labels != expectation.expanded_branches:
        raise DocumentationDOMError(
            f"{path}: expanded branches are {expanded_labels!r}, "
            f"expected {expectation.expanded_branches!r}.")

    panels = {
        node.attrs.get("aria-labelledby"): node
        for node in _find_all(primary_sidebar, "nav", "md-nav") if node.attrs.get("aria-labelledby")
    }
    for toggle in toggles:
        toggle_id = toggle.attrs.get("id", "")
        label = _label_for_toggle(primary_sidebar, toggle_id)
        if "tabindex" not in label.attrs:
            raise DocumentationDOMError(f"{path}: branch {label.text!r} is missing tabindex.")
        panel = panels.get(f"{toggle_id}_label")
        if panel is None:
            raise DocumentationDOMError(f"{path}: toggle {toggle_id!r} has no labelled navigation panel.")
        expected_expanded = "true" if "checked" in toggle.attrs else "false"
        if panel.attrs.get("aria-expanded") != expected_expanded:
            raise DocumentationDOMError(
                f"{path}: branch {label.text!r} reports aria-expanded="
                f"{panel.attrs.get('aria-expanded')!r}, expected {expected_expanded!r}.")


def validate_site(site_directory: Path) -> dict[str, object]:
    if not site_directory.is_dir():
        raise DocumentationDOMError(
            f"Rendered site directory does not exist: {site_directory}. Run mkdocs build first.")
    for relative_path, expectation in REPRESENTATIVE_ROUTES.items():
        path = site_directory / relative_path
        if not path.is_file():
            raise DocumentationDOMError(f"Representative page is missing: {path}.")
        _validate_route(path, expectation)
    return {
        "representative_routes": len(REPRESENTATIVE_ROUTES),
        "top_level_navigation": list(TOP_LEVEL_NAVIGATION),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "site_directory",
        nargs="?",
        type=Path,
        default=Path("site"),
        help="MkDocs output directory (default: site)",
    )
    args = parser.parse_args()
    result = validate_site(args.site_directory)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
