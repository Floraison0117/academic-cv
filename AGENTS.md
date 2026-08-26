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
- Keep page-specific images beside their content file, such as
  `content/blog/<slug>/featured.jpg` or
  `content/projects/<slug>/featured.png`.
- Do not remove or overwrite existing local changes unless the user explicitly
  asks for that.
