# Repository Guidelines

## Project Overview

This is a Hugo Blox academic CV / personal website built with Hugo modules and
Tailwind assets. Most site customization should happen in YAML front matter and
Markdown content files rather than theme internals.

## Key Files

- `config/_default/hugo.yaml`: Hugo site settings such as `baseURL`, language,
  and build options.
- `config/_default/params.yaml`: Hugo Blox identity, theme, typography, header,
  footer, and content behavior.
- `config/_default/menus.yaml`: top navigation items.
- `data/authors/me.yaml`: owner profile data used by the biography, education,
  interests, social links, and other profile blocks.
- `content/_index.md`: homepage landing page sections.
- `content/blog/`, `content/projects/`, and `content/friends/`: the currently
  published section content and their list pages.
- `assets/media/authors/me.png`: profile image.
- `assets/media/icons/custom/cc98.svg`: custom profile icon.

## Common Commands

- `pnpm run dev`: start the local Hugo development server.
- `pnpm run build`: build the production site into `public/` and generate
  Pagefind search index.
- `hugo server --disableFastRender`: direct Hugo dev command used by the npm
  script.

The `public/` and `resources/` directories, `hugo_stats.json`, and
`.hugo_build.lock` are generated locally and are intentionally not kept in the
working tree. Run the build again when a fresh production output is needed.

## Editing Notes

- Prefer editing Markdown and YAML content files over modifying generated or
  third-party theme code.
- Keep user profile information centralized in `data/authors/me.yaml` whenever
  a block references `username: me`.
- For a new project, create a folder under `content/projects/<slug>/` with an
  `index.md` and optional `featured.png` or `featured.jpg`.
- For a new blog post, create `content/blog/<slug>/index.md` and place its
  images in the same folder.
- For a friend link, create `content/friends/<slug>/index.md` with the friend's
  title, optional summary, and website URL under `links`. Place its cover image
  in the same folder as `featured.jpg` or `featured.png`.
- Keep page-specific images beside their content file, such as
  `content/blog/<slug>/featured.jpg` or
  `content/projects/<slug>/featured.png`.
- Do not remove or overwrite existing local changes unless the user explicitly
  asks for that.

## Typst to Markdown Conversion

- When converting a Typst note into a blog post, preserve the source prose,
  section hierarchy, lists, formulas, and links. Remove only Typst-only layout
  directives such as `#set`, `#show`, `#page`, `#outline`, `#pagebreak`, and
  spacing commands such as `#v`.
- Convert Typst headings (`=`, `==`, `===`, and so on) to the corresponding
  Markdown heading levels. Convert `#link("url")[text]` to a Markdown link.
- The site renders mathematics with KaTeX, so Typst math syntax must be
  converted to LaTeX. Common conversions include `RR` to `\\mathbb{R}`,
  `in` to `\\in`, `subset` to `\\subset`, named Greek letters to their
  backslash commands, `sum_` to `\\sum`, and `cases(...)` to a LaTeX
  `\\begin{cases}...\\end{cases}` block. Avoid nested constructs such as
  `\\text{\\operatorname{OPT}}`; use `\\operatorname{OPT}` or
  `\\mathrm{OPT}` instead.
- Keep formulas embedded in prose as inline math using `$...$`. Convert
  formulas occupying their own paragraph, including multi-line formulas, to
  `$$...$$` so KaTeX displays them centered.
- After conversion, search for remaining Typst commands and invalid nested
  KaTeX commands, run `git diff --check`, and run `pnpm run build` to verify
  that the page and its table of contents render successfully. Set `toc: true`
  in the page front matter when the article should show the desktop sidebar
  table of contents.
