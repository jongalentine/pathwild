# Cursor Keyboard Shortcuts for PathWild

## AI Features (Most Important)

| Shortcut | Action | Use Case |
|----------|--------|----------|
| `Cmd+K` | Inline AI edit | "Add error handling", "Refactor this" |
| `Cmd+L` | Open AI chat | Ask questions about code/concepts |
| `Tab` | Accept AI suggestion | Accept Cursor Tab completion |
| `Cmd+Shift+K` | Generate code | Describe what you want, Cursor writes it |

## Code Navigation

| Shortcut | Action | Use Case |
|----------|--------|----------|
| `Cmd+P` | Quick file open | Jump to any file quickly |
| `Cmd+Shift+F` | Search all files | Find code across project |
| `Cmd+Click` | Go to definition | Jump to function/class definition |
| `Cmd+Shift+O` | Go to symbol | Jump to function in current file |
| `F12` | Go to definition | Alternative to Cmd+Click |
| `Shift+F12` | Find all references | See where function is used |

## Editing

| Shortcut | Action | Use Case |
|----------|--------|----------|
| `Cmd+D` | Select next occurrence | Multi-cursor editing |
| `Cmd+Shift+L` | Select all occurrences | Change all instances |
| `Option+↑/↓` | Move line up/down | Reorder code |
| `Cmd+/` | Toggle comment | Comment/uncomment lines |
| `Shift+Option+↓` | Duplicate line | Copy line |

## Terminal & Running

| Shortcut | Action | Use Case |
|----------|--------|----------|
| `Ctrl+\`` | Toggle terminal | Open/close integrated terminal |
| `Shift+Enter` | Run cell (notebook) | Execute Jupyter cell |
| `Cmd+Enter` | Run cell, stay on it | Execute without moving |

## Git

| Shortcut | Action | Use Case |
|----------|--------|----------|
| `Ctrl+Shift+G` | Source control panel | View git changes |
| `Cmd+K Cmd+C` | Commit changes | Quick commit |

## General

| Shortcut | Action | Use Case |
|----------|--------|----------|
| `Cmd+Shift+P` | Command palette | Access any command |
| `Cmd+,` | Settings | Open settings |
| `Cmd+B` | Toggle sidebar | Show/hide file explorer |
| `Cmd+J` | Toggle panel | Show/hide terminal/output |

## PathWild-Specific Workflows

### "I need to add a feature"
1. `Cmd+K` on relevant code
2. Describe feature in plain English
3. Review and accept/modify suggestion

### "I don't understand this code"
1. Highlight code
2. `Cmd+L` to open chat
3. Ask: "Explain this code to me"

### "I have an error"
1. Highlight error message
2. `Cmd+K`
3. Type: "Fix this error"

### "I want to refactor"
1. Highlight code
2. `Cmd+K`
3. Type: "Refactor this into separate functions"
