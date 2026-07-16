#!/usr/bin/env bash
# Build multi-version ioailab documentation with mdBook.
#
# Produces one book/<version>/ subtree per version, a versions.json manifest
# consumed by the docs/theme/head.hbs version switcher, and a root redirect to
# the current latest documentation. Run on the host (requires `mdbook` and
# `git`).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="book"
# Subpath the docs are served under. Defaults to /ioailab for production
# deploys; set DOCS_SITE_BASE="" to serve book/ directly at a host root.
SITE_BASE="${DOCS_SITE_BASE-/ioailab}"

# Released, mdBook-era tags to publish, newest first. The 0.0.1 tag predates the
# mdBook migration (it used MkDocs) and is intentionally excluded. Add new
# releases to the front of this list.
RELEASED_TAGS=()

# The current working-tree docs ship under this path/label.
CURRENT_PATH="latest"
CURRENT_LABEL="latest"

# The docs root redirects to the current working-tree documentation.
LATEST="$CURRENT_PATH"

command -v mdbook >/dev/null || {
  echo "error: mdbook not found on PATH" >&2
  exit 1
}
command -v git >/dev/null || {
  echo "error: git not found on PATH" >&2
  exit 1
}

echo "Cleaning ${OUT}/ ..."
rm -rf "$OUT"
mkdir -p "$OUT"

# 1. Build the current docs from the working tree.
echo "Building ${CURRENT_LABEL} -> ${OUT}/${CURRENT_PATH}"
MDBOOK_OUTPUT__HTML__SITE_URL="${SITE_BASE}/${CURRENT_PATH}/" \
  mdbook build -d "${OUT}/${CURRENT_PATH}"

# 2. Build each released tag from an isolated worktree, overlaying the current
#    docs theme so older checkouts also render the version switcher.
for tag in "${RELEASED_TAGS[@]}"; do
  echo "Building ${tag} -> ${OUT}/${tag}"
  tmp="$(mktemp -d)"
  worktree="${tmp}/wt"
  git worktree add --quiet --detach "$worktree" "$tag"
  rm -rf "$worktree/docs/theme" "$worktree/theme"
  mkdir -p "$worktree/docs"
  cp -r "$ROOT/docs/theme" "$worktree/docs/theme"
  if ! grep -q '^theme = "docs/theme"$' "$worktree/book.toml"; then
    sed -i '/^\[output\.html\]$/a theme = "docs/theme"' "$worktree/book.toml"
  fi
  (
    cd "$worktree"
    MDBOOK_OUTPUT__HTML__SITE_URL="${SITE_BASE}/${tag}/" \
      mdbook build -d "$ROOT/${OUT}/${tag}"
  )
  git worktree remove --force "$worktree"
  rm -rf "$tmp"
done

# 3. Write the versions manifest (newest first) and mirror it into each version
#    directory so the switcher's relative fetch resolves.
manifest="${OUT}/versions.json"
{
  printf '{\n'
  printf '  "latest": "%s",\n' "$LATEST"
  printf '  "versions": [\n'
  printf '    { "id": "%s", "label": "%s", "path": "%s" }' \
    "$CURRENT_PATH" "$CURRENT_LABEL" "$CURRENT_PATH"
  for tag in "${RELEASED_TAGS[@]}"; do
    label="${tag#v}"
    printf ',\n    { "id": "%s", "label": "%s", "path": "%s" }' \
      "$tag" "$label" "$tag"
  done
  printf '\n  ]\n}\n'
} >"$manifest"

for v in "$CURRENT_PATH" "${RELEASED_TAGS[@]}"; do
  cp "$manifest" "${OUT}/${v}/versions.json"
done

# 4. Redirect the docs root to the newest released version (relative href so it
#    works under any mount point).
cat >"${OUT}/index.html" <<HTML
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta http-equiv="refresh" content="0; url=${LATEST}/index.html" />
    <link rel="canonical" href="${LATEST}/index.html" />
    <title>ioailab documentation</title>
  </head>
  <body>
    <p>Redirecting to the <a href="${LATEST}/index.html">latest ioailab documentation</a>.</p>
  </body>
</html>
HTML

echo "Done. Open ${OUT}/index.html or serve ${OUT}/ under ${SITE_BASE}/."
