# Git Commit Conventions

Git commit message specification based on GitHub community best practices and Conventional Commits.

## Format

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

## Types

### Core Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation changes |
| `ref` | Code refactoring |
| `test` | Test-related changes |
| `bump` | Version updates |

### Extended Types

| Type | Description |
|------|-------------|
| `style` | Code formatting |
| `perf` | Performance improvements |
| `build` | Build system changes |
| `ci` | CI/CD changes |
| `chore` | Maintenance tasks |
| `revert` | Revert previous commit |

## Scope (Optional)

Identifies the module or component affected by the change:

```bash
feat(chat): add streaming response support
fix(models): resolve model loading timeout
docs(api): update endpoint documentation
ref(utils): extract common helper functions
```

## Description Rules

- **Length**: 50 characters max (72 hard limit)
- **Style**: Imperative mood ("add" not "added" or "adds")
- **Case**: Lowercase first letter
- **Punctuation**: No period at the end
- **Language**: English preferred

## Examples

### Basic Examples

```bash
feat: add streaming response support
fix: resolve memory leak in model loading
docs: update API documentation
ref: extract common validation logic
bump: update mlx-lm to 0.24.1
```

### With Scope

```bash
feat(chat): implement streaming response support
fix(models): resolve model loading timeout
perf(embeddings): optimize vector computation
test(images): add integration test coverage
bump(deps): update mlx-lm to 0.24.1
build(pyproject): update project metadata
```

### Complete Format

```bash
feat(chat): add real-time notifications

Implement WebSocket-based notifications for instant
message delivery. Improves user experience by eliminating
manual page refresh requirement.

Closes #234
```

### Breaking Changes

```bash
feat(api)!: change authentication method

BREAKING CHANGE: Authentication endpoint now requires
OAuth2 instead of basic auth. Update client code to
use new authentication flow.
```

### Version Updates

```bash
bump(deps): update mlx-lm to 0.24.1
bump(dev): update pytest to 8.3.4
bump(pyproject): version 0.4.4
bump(lock): refresh uv.lock dependencies
```

### Reverts

```bash
revert: "feat(auth): add OAuth2 login support"

This reverts commit 1234567. OAuth2 implementation
caused compatibility issues with existing clients.
```

## PR Title Convention

PR titles should follow the same format as commit messages:

```bash
feat(auth): implement OAuth2 login support
fix(ui): resolve mobile layout issues
ref(api): extract common validation logic
bump(patch): update dependencies for security fixes
```

## LLM Generation Template

For AI assistants to generate compliant commit messages:

```markdown
Generate Git commit message following this specification:

Format: <type>(<scope>): <description>

Types:
- feat: new feature
- fix: bug fix
- docs: documentation
- ref: refactoring
- test: testing
- bump: version/dependency updates

Rules:
1. Use imperative mood ("add" not "added")
2. 50 characters max, lowercase first letter, no period
3. Scope optional but recommended
4. Use English

Examples:
feat(chat): add streaming response support
fix(models): resolve model loading timeout
ref(utils): extract common validation logic
bump(deps): update mlx-lm to 0.24.1
test(chat): add integration test coverage
build(pyproject): update project metadata
```

## Semantic Versioning Integration

Commit types automatically determine version bumps:

- `feat:` → MINOR version (1.1.0)
- `fix:` → PATCH version (1.0.1)
- `BREAKING CHANGE:` → MAJOR version (2.0.0)

## Best Practices

1. **Atomic commits**: One logical change per commit
2. **Clear descriptions**: Explain what and why, not how
3. **Consistent formatting**: Follow the specification strictly
4. **English language**: Use English for international collaboration
5. **Timely commits**: Commit frequently with focused changes

---

*Based on [Conventional Commits](https://www.conventionalcommits.org/) specification and GitHub community practices.*
